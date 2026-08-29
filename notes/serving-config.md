# Serving config & fixes — SM120 single-outlet inference

## GLM-5.3-Flash — current (validated 2026-08-29)

Deployed via k8s-gitops `stage3/apps/vllm.yaml`; image
`registry.ocnr.org/infra/vllm:0.29.0-sm120-cu130@sha256:29005b58…` built from
the fork's `0.29` branch (`d1fc212696`), which includes the
hardware-verified SM120 GLM-5.3 port (fp8 + FlashInfer NoPE sparse MLA) and
the 2026-08-28 kv-offload fix series.

```
vllm serve /models/zai-org/GLM-5.3-Flash \
  --served-model-name GLM-5.3-Flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --disable-custom-all-reduce \
  --enable-expert-parallel \
  --max-model-len auto \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.97 \
  --kv-cache-dtype fp8 \
  --kv-offloading-size 100 \
  --kv-offloading-backend native \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --default-chat-template-kwargs '{"thinking": true}'
```

Env: `HF_HUB_OFFLINE=1`, `NCCL_P2P_LEVEL=NODE`, `RAYON_NUM_THREADS=4`,
`OMP_NUM_THREADS=4`, `MAX_JOBS=32`,
`VLLM_FLASHINFER_AUTOTUNE_PROCESS_GROUP=1`, `VLLM_KV_OFFLOAD_COLLECTIVE_BARRIER=1`).
Keep CUDA graph memory profiling enabled (v0.21+ default, don't set
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` — see *Allocator flush-retry
warnings* below).

### Auto-fit memory model (measured, 4× RTX PRO 6000, fp8 KV, TP4)

`--max-model-len auto` binary-searches the largest context that fits the
profiled KV budget (`kv_cache_utils.py:_auto_fit_max_model_len`) and logs one
of:

- `Auto-fit max_model_len: full model context length 1048576 fits in available GPU memory`
- `Auto-fit max_model_len: reduced from 1048576 to NNNN to fit in available GPU memory (… GiB)`

Per-GPU levers (1% of gmu ≈ 0.93 GiB on the 97,887 MiB cards):

| Lever | KV effect |
|---|---|
| 1M tokens costs | ≈ 7.6 GiB/GPU (≈7.6 KiB/token) |
| gmu 0.94 → 0.97 | +2.8 GiB |
| mbt 2048 → 8192 (profiled prefill activations) | −0.74 GiB |
| Vision stack resident (image: 1) | −0.33 GiB |
| CUDA graph reserve (profiling enabled) | −0.95 GiB vs profiling disabled |

Final config `0.97 + mbt 8192` → 7.95 GiB → **1,100,441 tokens, full 1M with
`image: 1` (1.05x concurrency)**. A 1M request consumes ~95% of GPU KV;
concurrent overflow spills to the 100 GiB offload tier. Forcing
`--max-model-len 1048576` explicitly does *not* bypass the fit check — it
raises a hard ValueError at startup while memory is short.

### Allocator flush-retry warnings (benign, know the signature)

```
[W829 HH:MM:SS CUDACachingAllocator.cpp:3933] memory allocation failed with OOM
on device N while trying to allocate X bytes (free: Y, total: 102014189568)
```

These are the caching allocator's first-attempt-missed, flush-cached-blocks-
and-retry path — warning level, self-healing, and expected once per new
runtime shape. The hard-fail variant to
watch for is `RuntimeError: CUDA out of memory. Tried to allocate …` — that
kills the engine; fall back gmu → 0.965 (mbt 8192) or mbt → 4096 (gmu 0.97),
both of which still hold 1M.

### Benchmarks (2026-08-29, 16 prompts, concurrency 4, zero failures)

| Input | Output | Decode (tok/s) | p50 TTFT | p50 ITL |
|---|---|---|---|---|
| 2048 | 256 | 182 | 677ms | 19ms |
| 8192 | 1024 | 181 | 1716ms | 19ms |
| 32768 | 4096 | 184 | 6467ms | 19ms |
| 131072 | 8192 | 150 | 20395ms | 20ms |

Mid-run Triton/TileLang JIT compiles (`_count_expert_num_tokens`,
`_kpool_tail_seed_kernel`, `mhc_pre_big_fuse_with_norm_tilelang`) still fire on
first hit of uncovered shapes — one-off latency spikes, warmup-coverage fix
pending.

### kv-offload status

**Enabled** (`--kv-offloading-size 100 --kv-offloading-backend native`) after
the 2026-08-28 fix series (`d1fc212696`, one fix per assert surface):
GLM-5.3's kpool-tail KV group (4-token blocks, opts out of
prefix caching) broke every offload-connector invariant that assumes all
groups are hash-chained; the series makes non-participating groups fully
GPU-resident across config, store/load, lookup/match, and alloc accounting.
The deadlock below was the pre-series state on the M3/0.25.1 build.

## MiniMax-M3-NVFP4 — historical (2026-08, superseded by GLM-5.3-Flash)

Served at gmu 0.97 on the `0.25.1-sm120-cu131` image (full flag list in git
history); full 1M context auto-fit at 1,136,384 KV tokens. **Do NOT add
kv-offloading on this config** — see `incident-kv-offload-deadlock.md`.
Superseded for GLM-5.3-Flash, where offloading is enabled after the
2026-08-28 fix series (above).

Durable lessons from that bring-up:

- Nested VL models can hide MoE experts from ModelOpt quant resolution (the
  nested weight mapper strips the `language_model.` prefix) → experts load
  bf16 and OOM. Fix lives in `_quantized_layer_prefix_candidates`.
- TRTLLM MoE needs downloaded cubins — unusable offline; FlashInfer-CUTLASS
  was the only SM120 backend honoring M3's clamped SwiGLU.
- FlashInfer runtime JIT needs `cuda-nvrtc-dev` in the *runtime* image, not
  just the build image.
- Hybrid/sparse attention reconciles block sizes across groups; a mismatched
  `--block-size` fails KV init ("No common block size").
- PCIe-only multi-GPU: `NCCL_P2P_LEVEL=NODE`, custom all-reduce auto-disables
  beyond 2 GPUs, and P2P should be functionally verified before enabling
  fused AR+RMSNorm (vllm PR #47544).
- vLLM renamed the reasoning field `reasoning_content` → `reasoning`; clients
  reading only the old name see "missing reasoning".
- Tool-call streaming: per-delta regexes can't see namespace markers that
  arrived in earlier deltas — parser fixes must compose normalize →
  boundary-collapse → tolerant-close (M3's stack: #47001 + ocnr fixes,
  cargo-tested).
