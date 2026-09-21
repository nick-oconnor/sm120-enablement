# Serving config & fixes — SM120 single-outlet inference

## GLM-5.3-Flash — current (2026-09-20 0.30 re-cut; native KV offload)

Deployed via k8s-gitops `stage3/apps/vllm.yaml`; image
`registry.ocnr.org/infra/vllm:0.30.0-sm120-cu130` built from the `0.30`
branch (2026-09-20 re-cut onto upstream vLLM `main` at `4868312128` —
GLM-5.3-Flash model support is native upstream since vllm-project #53906,
the ZJY0516 fork is retired). On top of upstream: the ocnr SM120 NoPE
sparse-MLA port (fp8 + FlashInfer zero-pad, backend priority, buffer pin),
the b12x PCIe oneshot allreduce integration (b12x 1.3.0, CuTe DSL — no
native extension build), the hybrid-state prefix-cache fix #55601 and the
indexer-prefill-workspace right-size #55222 (both still open upstream).
KV offloading is enabled on upstream's native backend (see *kv-offload
status* below).

The 09-20 re-cut exists to fix the two 0.30 production defects — the lost
1M context and the silent KV-cache poisoning. Both are covered below under
*Auto-fit memory model* and *kpool tail seed poisoning*; the ocnr
`_use_cooperative_topk` override was dropped in the rebase because upstream
#57546 moved kpool top-k onto the shared `SparseIndexerTopk` dispatcher,
which already carries the SM12x exclusion.

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
("full model context length 1048576 fits", 2026-09-18 01:29 UTC).

### 0.30 context regression — root cause and fix (2026-09-20)

The 2026-09-18 **0.30** boot regressed: only **3.81 GiB** profiled for KV →
**518,311 tokens**, and auto-fit cut max_model_len from 1,048,576 to
**516,096** — prompts above 516K were rejected. (Forcing
`--max-model-len 1048576` does *not* bypass the fit check; it raises a hard
ValueError at startup while memory is short.)

Root cause: **the indexer prefill gather workspace is sized in tokens, but
this indexer's KV is pool-granular.**
`Indexer.__init__` (`vllm/models/glm5next/common/attention.py`) takes
`get_max_prefill_buffer_size(vllm_config)` = `max_model_len * 40` entries of
132 B verbatim, while the spec carries `compress_ratio == index_kpool == 4`.
At `max_model_len = 1,048,576` that reserves **5.16 GiB/GPU** for the K-gather
workspace instead of 1.29 GiB. `deepseek_v4/attention.py` already divides at
the same call site. Upstream issue **#55221**, PR **#55222** (open) — carried
on the branch.

Measured on this box (4x RTX PRO 6000, TP4, gmu 0.97, mbt 8192, fp8 KV,
offload on):

| build | consumed (weights+non-torch) | available KV | auto-fit |
|---|---|---|---|
| 0.30 09-18 image, `--max-model-len auto` | 82.27 GiB | 3.81 GiB | cut to 516,096 (518,311 tok) |
| same image, `--max-model-len 262144` (A/B: scales the same workspace by 1/4) | 78.40 GiB | 7.77 GiB | n/a (1,016,607 tok pool) |
| 0.30 09-20 re-cut with #55222 | **78.40 GiB** | **7.68 GiB** | **full 1,048,576 fits** (1,064,361 tok, 1.02x) |

The A/B row is the attribution: cutting `max_model_len` by 4x on the
*unfixed* image reproduces exactly the same 3.87 GiB saving that #55222
produces at the full 1M, which is the workspace and nothing else.

**Do not blame — or disable — the CUDA graph estimate.** The boot logs
`CUDA graph pool memory: 0.07 GiB (actual), 4.19 GiB (estimated)`, a 60x
overshoot that looks like the culprit and is not. `profile_cudagraph_memory`
runs the first forward pass that has a KV cache, so the one-time persistent
allocations it triggers (FlashInfer/TileLang workspaces, CuTe DSL, b12x
pools) are charged to "CUDA graph memory"; the real capture afterwards is
cheap because those buffers already exist. The reserve corresponds to real
resident memory — steady-state `nvidia-smi` is ~91.6 GiB/GPU against
`consumed + KV + actual-cudagraph` of ~86 GiB. Setting
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` double-spends that memory into
the KV pool and OOMs; the 2026-09-19 attempts to disable the estimate
(k8s-gitops `a0822ae4`) and to pin `--kv-cache-memory` (`99b6c822`,
`486cd09b`) were both reverted for that reason.

### kpool tail seed poisoning (silent KV corruption; fixed 2026-09-20)

The second 0.30 production defect: **every prefill wrote raw indexer K and
gate scores into unrelated indexer cache blocks**, and never seeded the
request's own tail block.

`get_kv_cache_config_from_groups` (`vllm/v1/core/kv_cache_utils.py`) aliases
each layer's **tail** tensor onto its **indexer** tensor at the same offset,
so the tail view inherits the indexer's block stride. Probed on this box by
printing `tail_kv_cache.stride()` at the first prefill of the 09-18 image:

```
OCNR-TAIL-LAYOUT shape=(N, 2, 4, 128) stride=(38016, 512, 128, 1) dense_block_elems=1024
```

`stride(0)` is **38016** elements; the NVIDIA `_kpool_tail_seed_kernel`
computed `base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM`, i.e. a dense
**1024**. Consequences, per prefill, per layer, per in-flight request:

- the 128-element K write and the 128-element score write land at
  `blk * 1024`, which for `blk > 0` is inside indexer block `blk // 37` —
  pooled indexer keys of an unrelated block are overwritten with raw
  (unpooled, unquantized) values;
- the request's own tail block is left holding whatever the previous tenant
  of that indexer region wrote, so the boundary pool that decode compresses
  is seeded from garbage.

It is silent: the stray writes stay inside the shared KV allocation, so
there is no OOB fault, no assert and no log line. The damage lands in
prefix-cached indexer blocks, so it **persists across requests and
accumulates with block reuse/eviction** — matching the observed profile of
a session that is fine for a while and then degenerates into repeated-token
word salad, cleared only by a restart. The AMD kernel never had the bug
(it already addressed by stride).

Fixed upstream by **#57477** (`TAIL_BLOCK_ELEMS = tail.stride(0)`,
`KPOOL_HEAD = tail.stride(1)`), merged and in the 09-20 base. Regression
test `tests/kernels/test_kpool_decode_update_batched.py::test_prefill_seed_honors_padded_tail_block_stride`
now runs on every platform; run against both images it reports:

```
09-18 image: tail K seeded False / score False / no stray write False -> FAIL
09-20 image: tail K seeded True  / score True  / no stray write True  -> PASS
```

This is a *different* defect from #55600/#55601 (the KDA recurrent-state
seed units), which was already carried. Both had to be fixed to make the
prefix-cache path honest.

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

### Benchmarks (2026-09-21, 0.30 production build, offload on, b12x oneshot — concurrency 1, 16 prompts per cell, zero failures)

`vllm bench serve` (same protocol as the 09-11 baseline below) against the
production deployment — statefulset `apps/vllm`, image
`0.30.0-sm120-cu130@sha256:d15260ba`, auto-fit full 1M, native KV offload on:

| Input | Output | Decode (tok/s) | Median TTFT | Median ITL | TPOT |
|---|---|---|---|---|---|
| 2048 | 256 | 86.15 | 237ms | 10.73ms | 10.72 |
| 8192 | 1024 | 86.59 | 850ms | 10.73ms | 10.73 |
| 32768 | 4096 | 86.81 | 3217ms | 10.74ms | 10.73 |
| 131072 | 8192 | 82.47 | 10471ms | 10.86ms | 10.85 |

vs the 09-11 no-offload 0.29 baseline: long-context prefill improved on the
0.30 re-cut — median TTFT 14270ms → 10471ms at 128K (−27%) and 3300ms →
3217ms at 32K — and the 128K decode rate holds (82.09 → 82.47 tok/s).
Short-context decode eases ~3% (89.0-89.7 → 86.2-86.8 tok/s; median ITL
10.33-10.34 → 10.73-10.74ms, +0.4ms). TTFT percentiles stay tight at every
length (P99 within ~25ms of P50); no failed requests, no preemptions. This
is the current production profile — the table below is the no-offload
reference build.

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
identical on a 12,388-token prompt (≥3 mamba blocks). The external-hit soak
was run on 2026-09-20 — see the soak results at the end of this section.
See `incident-kv-offload-cache-corruption.md`.

The `0.30` branch (2026-09-20 re-cut onto upstream main `4868312128`)
carries what re-enabling offloading needs on GLM-5.3-Flash:

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
The offload tier does not extend the *schedulable* context: `max_model_len`
is bounded by the GPU KV pool alone, so the 09-18 context regression had to
be fixed on the GPU side (#55222, see the auto-fit section above).

**2026-09-20 eviction + external-hit soak** (09-20 re-cut, full 1M auto-fit,
offload on): six distinct ~300K-token sessions run cold and then revisited
after the whole 1.06M-token GPU pool had turned over — 12/12 needles
correct, 0 preemptions, 0 errors, 3,594,240 external (CPU) prefix-cache hit
tokens served, revisit latency 1.3-2.7 s against 33-35 s cold. Also
verified: a 1,017,549-token prompt retrieves its needle cold (136 s) and
warm (2.6 s), and a 716,924-token prompt likewise — both well past the old
516,096 ceiling. This is the external-hit soak that was listed as pending
above; it now also covers the #57477 tail-seed path.

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
