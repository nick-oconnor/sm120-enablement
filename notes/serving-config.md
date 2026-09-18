# Serving config & fixes — SM120 single-outlet inference

## GLM-5.3-Flash — current (2026-09-18 0.30 rebuild; native KV offload)

Deployed via k8s-gitops `stage3/apps/vllm.yaml`; image
`registry.ocnr.org/infra/vllm:0.30.0-sm120-cu130@sha256:11190e94…` built from
the `0.30` branch (2026-09-18 re-cut onto upstream vLLM `main`
past the v0.30.0rc1 fork — GLM-5.3-Flash model support is native upstream
since vllm-project #53906, the ZJY0516 fork is retired). On top of upstream:
the ocnr SM120 NoPE sparse-MLA port (fp8 + FlashInfer zero-pad, backend
priority, buffer pin), the b12x PCIe oneshot allreduce integration (b12x
1.3.0, CuTe DSL — no native extension build), and the hybrid-state
prefix-cache fix #55601 (still open upstream). KV offloading is enabled on
upstream's native backend (see *kv-offload status* below).

```
vllm serve /models/zai-org/GLM-5.3-Flash \
  --served-model-name GLM-5.3-Flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len auto \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.97 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --kv-offloading-size 100 \
  --kv-offloading-backend native \
  --enable-chunked-prefill \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --default-chat-template-kwargs '{"thinking": true}'
```

Env: `HF_HUB_OFFLINE=1`, `NCCL_P2P_LEVEL=NODE`, `RAYON_NUM_THREADS=4`,
`OMP_NUM_THREADS=4`, `MAX_JOBS=32`,
`VLLM_FLASHINFER_AUTOTUNE_PROCESS_GROUP=1`,
`VLLM_ENABLE_PCIE_ALLREDUCE=1` (b12x oneshot replaces NCCL-SHM for decode-size
all-reduces; >8 MiB prefill collectives stay on NCCL).
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

Final config `0.97 + mbt 8192` → 7.91 GiB → **1,095,931 tokens, full 1M with
`image: 1` (1.05x concurrency)** (identical on the 2026-09-09 and 2026-09-11
rebuild boots; was 7.95 GiB / 1,100,441 tokens on the fork build). The
offload-enabled boot on the pre-0.30 image held the same full 1M
("full model context length 1048576 fits", 2026-09-18 01:29 UTC). The
2026-09-18 **0.30** boot regressed: only **3.81 GiB** profiled for KV →
**518,311 tokens**, and auto-fit **cut max_model_len from 1,048,576 to
516,096** — prompts above 516K are rejected; the offload tier does NOT
extend the schedulable context in this build. Consumed memory (weights +
non-torch) came in at 82.27 GiB/GPU (~5 GiB above the pre-0.30 boots, which
ran identical offload args; the CUDAGraph reserve also fell 0.95 → 0.06 GiB)
— the delta is in the 0.30 build, attribution not yet bisected. Forcing
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

### b12x PCIe oneshot allreduce (deployed 2026-09-02)

`CustomAllreduce` gains a b12x backend when `VLLM_ENABLE_PCIE_ALLREDUCE=1` and
b12x ≥ 1.3.0 is installed: the oneshot pool replaces the legacy custom-AR path
at TP>2 PCIe-only (the stock `world_size > 2` gate is bypassed), routing by
size (≤ `max_size` → oneshot, else NCCL). Capture uses `single_channel=True` —
one semantic channel for the whole vLLM graph-capture phase; distributed
multi-channel mode demands per-graph channel ids vLLM does not plumb.
`PCIeAllReduce.should_allreduce` on b12x delegates to a pool method that does
not exist — route by size, do not call it. Kernels are CuTe DSL, compiled at
first use (first requests after boot pay ~50 ms ITL once, self-heals).

### Benchmarks (2026-09-11, no-offload build, b12x oneshot — concurrency 1, 16 prompts per cell, zero failures)

| Input | Output | Decode (tok/s) | Median TTFT | Median ITL | TPOT |
|---|---|---|---|---|---|
| 2048 | 256 | 89.04 | 241ms | 10.33ms | 10.33 |
| 8192 | 1024 | 89.40 | 877ms | 10.34ms | 10.34 |
| 32768 | 4096 | 89.71 | 3300ms | 10.34ms | 10.34 |
| 131072 | 8192 | 82.09 | 14270ms | 10.48ms | 10.47 |

vs the 2026-08-29 c=4 baseline (decode 150–184 tok/s, ITL 19–20 ms): the c=1
per-stream rate at long context (82 tok/s at 128K) is ~2× the c=4 per-stream
rate, and production (c=1) ITL improved 12.5 → 10.8 ms (−13.6%) post-deploy.
Direct peer reads measured 53 GB/s (PCIe 5.0 x16 line rate) on driver 610
without any P2P registry overrides. Mid-run Triton/TileLang JIT compiles
(`_count_expert_num_tokens`, `_kpool_tail_seed_kernel`,
`mhc_pre_big_fuse_with_norm_tilelang`) still fire on first hit of uncovered
shapes — one-off latency spikes, warmup-coverage fix pending.

### kv-offload status

**Re-enabled in production 2026-09-17** (`--kv-offloading-size 100
--kv-offloading-backend native`, image `d6b7d2c6`, gitops `91b39a43`).
Requires **dshm 120Gi** — the native CPUOffloadingSpec mmaps its 100 GiB pool
in `/dev/shm` (`vllm_offload_*.mmap`); the 8Gi dshm set in `128c387a` (when
offload was disabled) starves it and was reverted in the same commit. The
2026-09-10 corruption was **not** caused by offloading — the root cause is
upstream #55600 (KDA recurrent-state slot seeded in the wrong units on *any*
prefix-cache hit, local or external); it recurred on the no-offload build at
a 97% local hit rate. Post-deploy validation: cold vs warm answers byte
identical on a 12,388-token prompt (≥3 mamba blocks); external-hit soak
pending. See `incident-kv-offload-cache-corruption.md`.

The `0.30` branch (2026-09-18 re-cut onto upstream main past
v0.30.0rc1) carries what re-enabling offloading needs on GLM-5.3-Flash:

- #55601 — seed hybrid mamba state index with `mamba_block_size` (the
  corruption fix; 1-line, `mamba_hybrid.py`) — still open upstream, carried
- #54743 — scope offload group configs to prefix-cacheable groups: NOT
  needed on 0.30 — upstream's native offloader selects eligible groups
  itself (`get_offloading_group_ids` → `prefix_cacheable_group_ids`),
  verified booting the multi-group kpool-tail config 2026-09-18 (the PR
  itself is still open upstream with conflicts)
- #55450 — retire mamba states across null gaps: merged upstream, in the
  0.30 base
- #50388 — fix ValueError on KV load failure with a hybrid KV cache:
  merged upstream, in the 0.30 base

The 0.28-era native KV fixes (#52771 zeroed hits under spec decode, #54288
final-sampled-token slot, #51787 recency) were verified already present in
the tree. Re-enable flags (previous production form):
`--kv-offloading-size 100 --kv-offloading-backend native`. Residual open
risk: #50454 (assert crash under multi-group external-hit + eviction
pressure, no upstream fix) — worst case is a crash, not silent corruption.
0.30 caveat: auto-fit cut `max_model_len` to 516,096 on the 2026-09-18 boot
(see the auto-fit section above) — the offload tier does not extend the
schedulable context in this build.

The deadlock below was the pre-series state on the M3/0.25.1 build.

## MiniMax-M3-NVFP4 — historical (2026-08, superseded by GLM-5.3-Flash)

Served at gmu 0.97 on the `0.25.1-sm120-cu131` image (full flag list in git
history); full 1M context auto-fit at 1,136,384 KV tokens. **Do NOT add
kv-offloading on this config** — see `incident-kv-offload-deadlock.md`.
Superseded by GLM-5.3-Flash (offloading carried on that build after the
2026-08-28 fix series, since removed — see *kv-offload status* above).

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
