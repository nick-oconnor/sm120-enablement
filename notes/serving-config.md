# Serving config & fixes — SM120 single-outlet inference

## GLM-5.3-Flash NVFP4 + MTP3 — current (2026-09-27)

Deployed via k8s-gitops `stage3/apps/vllm.yaml`; image
`registry.ocnr.org/infra/vllm:0.30.0-sm120-cu130@sha256:3399d4fc` — the
2026-10-01 CI rebuild of the `0.30` branch (`0.30` rebased onto upstream
main `924707f1bf`, ~202 commits: FlashInfer **0.7.0** (#58069), the
GLM-5.3-Flash corruption-hunt fixes #58454 (kpool pool selection with spec
decode) and #58368 (prompt-tail prefix-cache hits with MTP), the
profiling-allocator fix #58430, the GLM5.3 metadata-op optimization
#58450). The branch carries #55222 (both halves, one squashed commit),
#55601, #57635 (per-rank FlashInfer autotune cache), two ocnr
autotune/scheduler fixes and the SM120 NoPE sparse-MLA port — full list in
`vllm/SM120-GLM53-FLASH.md`. The rebuild re-pins the 09-27 tree
(`87feab60`) after a one-day rebase onto upstream `2eaa3bc5ac` (built as
`7a7d3d10`) was reverted as broken — same tree; the digest moved only via
the commit-timestamp build stamp (symptom:
`incident-recut-kv-poisoning.md`). Pushed to Docker Hub 2026-10-01. Checkpoint stayed on
`nvidia/GLM-5.3-Flash-NVFP4` (ModelOpt recipe
`nvfp4_experts_dense_mlp-kv_fp8_cast`: experts + dense MLP W4A4, attention /
router / norms / lm_head / MTP head at source precision, fp8-cast KV recipe)
with MTP speculative decoding k=3. Served model name is still
`GLM-5.3-Flash`. Weights 49.17 GiB/GPU (vs 76.87 fp8); the freed VRAM pays
for the MTP draft head *and* a ~24 GiB/GPU reservation for comfyui, which
shares GPUs 0 and 3.

```
vllm serve /models/nvidia/GLM-5.3-Flash-NVFP4 \
  --served-model-name GLM-5.3-Flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len auto \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.69 \
  --moe-backend flashinfer_cutlass \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --kv-offloading-size 100 \
  --kv-offloading-backend native \
  --enable-chunked-prefill \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image": 20, "video": 0}' \
  --default-chat-template-kwargs '{"thinking": true}'
```

Env carries over from the fp8 era (`HF_HUB_OFFLINE=1`,
`NCCL_P2P_LEVEL=NODE`, `RAYON_NUM_THREADS=4`, `OMP_NUM_THREADS=4`,
`MAX_JOBS=32`, `VLLM_ENABLE_PCIE_ALLREDUCE=1`); dshm stays 120Gi for the
100 GiB offload mmap. `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS` is **gone** —
see *EP autotune deadlock* below.

### Boot (2026-09-26, f8cdce48, NVFP4+MTP3)

Attention stays `FLASHINFER_MLA_SPARSE_SM120` + `fp8_ds_mla` — the checkpoint
quantizes experts + dense MLP only, so the SM120 sparse-MLA port is untouched.
MoE resolves to `FLASHINFER_CUTLASS` NVFP4 (pinned via `--moe-backend`), and
the bf16 MTP draft experts route through the unquantized FlashInfer-CUTLASS
path. Weights load 49.17 GiB/GPU in ~85s (vs ~265s at fp8). Profiled budget
at gmu 0.69: consumed 50.71 GiB (weights + non-torch), peak activation
6.34 GiB, CUDA graph 0.26 GiB, **KV 8.50 GiB → 1,064,126 tokens (1.01x
concurrency over 1,048,576)**; auto-fit holds the full 1M.

### gmu 0.69 memory model (measured)

gmu is sized from measured *resident* peak, not the boot accounting: fp4
kernel workspaces and the 1M-shape indexer allocations land outside vLLM's
accounted budget, so resident runs ~4.8 GiB above `requested`. At 0.69 a full
1M prefill leaves 24.5 GiB free on GPU 0 and 25.1 GiB on the rest; 0.70 drops
GPU 0 to 23.7 GiB. Little room above that: KV headroom over 1,048,576 tokens
is only 1.5%.

### MTP3 tuning

Acceptance 66% overall — 79.9 / 59.3 / 41.8% by draft position, 1.81 accepted
tokens/step. k=3 over k=2: k=2 costs the same KV (8.47 GiB) and decodes
slower. k=3 over k=1 (136-139 tok/s): the +19% outweighs shrinking the 1M
headroom from 1.05x to 1.01x. Decode at c=1: 89 → 157-165 tok/s
(ITL 10.3 → 6.0-6.4 ms).

### EP autotune deadlock (fixed by the #57635 + leader-only-warmup carries)

FlashInfer autotunes the trtllm `fused_moe` tactics even though CUTLASS is
the selected NVFP4 MoE backend, and the EP ranks fell out of phase and
deadlocked: rank 0 completed while ranks 1-3 sat at 0% forever, visible only
as repeated `shm_broadcast "No available shared memory broadcast block"`
from EngineCore.

- 2026-09-26: `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=trtllm::fused_moe::gemm1,gemm2`
  dodged it (fp4_gemm autotune still ran).
- 2026-09-27, on FlashInfer 0.7.0, `fp4_gemm` deadlocked the same way; the
  skip list grew (one boot) and then went away entirely. Two carried fixes
  are the actual resolution: **#57635** persists the autotune cache per rank
  (a warm cache root could no longer desync a later boot), and the ocnr
  leader-only fix (vllm `022c5085fa`) stops the SM120 sparse-MLA warmup from
  tuning leader-only — on the NVFP4 checkpoint it tuned the linear and MoE
  GEMMs too, so rank 0 hit its live in-process cache and skipped the
  per-tactic all-reduce while ranks 1-3 blocked in it. Broadcasting the file
  cannot equalize the ranks: FlashInfer consults the live `profiling_cache`
  before file configs, and MoE entries are keyed by tp/ep rank.
- Verified on 4x RTX PRO 6000 (TP4+EP) with NVFP4 + MTP3 + FlashInfer 0.7.0
  and **no skip-ops env**: boots to ready; `fp4_gemm` and
  `trtllm::fused_moe::gemm1` tune to completion on every rank; MTP
  acceptance 83%; decode 170-186 tok/s (ITL 5.4-5.9 ms) vs 161-172 under the
  skip-ops workaround — faster, because these ops are now actually tuned.
  Trade-off: sparse-MLA decode shapes from the mixed-batch warmup fall back
  to FlashInfer's tactic heuristic (no measurable decode regression).

### Boot (2026-09-27, 87feab60, NVFP4+MTP3, FlashInfer 0.7.0)

Same boot shape as 09-26 (attention `FLASHINFER_MLA_SPARSE_SM120` +
`fp8_ds_mla`, MoE `FLASHINFER_CUTLASS` NVFP4); the rebase adds the upstream
corruption fixes #58454/#58368 and profiling fix #58430 to the base. Second
boot with the persistent cache root completes autotune cleanly (the #57635
per-rank cache).

### Boot (2026-10-01, 3399d4fc — rebuild of the 09-27 tree)

Same boot shape as 09-27: backend `FLASHINFER_MLA_SPARSE_SM120` +
`fp8_ds_mla`, auto-fit full 1M, clean startup (no ERROR/assert lines),
serving 200 OKs. No new bench run — the image content is identical to the
`87feab60` build.

### Checkpoint prerequisite (host-side, not in git)

NVIDIA ships the MTP module (layer 45) in bf16 — no `weight_scale` tensors,
while every real layer has them — but omits it from `hf_quant_config.json`'s
quant exclusion list, so vLLM builds NVFP4 experts for the draft model and
startup dies with
`The size of tensor a (1024) must match the size of tensor b (2048)` (fp4
packs two values per byte). Fix on home-0: edit the checkpoint's
`config.json` and `hf_quant_config.json` to declare layer 45 unquantized,
listing **both** `model.layers.45*` and `model.language_model.layers.45*`:
`apply_vllm_mapper` rewrites exclusions through the target model's
WeightsMapper (`model.language_model.` → `language_model.model.`), but the
MTP head is built as a text-only model whose modules are `model.layers.45.*`,
so only the unmapped spelling actually matches. Originals kept alongside as
`*.nvidia-orig`; **re-apply after any re-download of the checkpoint**.

### Validation (2026-09-26)

1,017,549-token needle found cold (141s) and warm (2.8s); tool calling and
CJK/emoji clean (this checkpoint does not reproduce upstream #54150's invalid
UTF-8 on SM120); 0 preemptions, 0 errors.

### Benchmarks (2026-09-26, NVFP4+MTP3, offload on, b12x oneshot — c=1, 16 prompts per cell, zero failures)

`vllm bench serve` (same protocol as the 09-21/09-11 tables) against the
NVFP4+MTP deployment — job `vllm-bench-manual-spawn-muipn6n7-0o010`, pod
`-dcdrk`, 18:15-18:43 UTC:

| Input | Output | Decode (tok/s) | Median TTFT | Median ITL | Median TPOT | Accept len |
|---|---|---|---|---|---|---|
| 2048 | 256 | 129.38 | 189ms | 17.33ms | 7.07 | 2.48 |
| 8192 | 1024 | 129.08 | 327ms | 17.40ms | 7.52 | 2.34 |
| 32768 | 4096 | 134.45 | 325ms | 17.45ms | 7.35 | 2.37 |
| 131072 | 8192 | 134.60 | 681ms | 17.65ms | 7.29 | 2.47 |

- **32K/128K TTFT are warm external-prefix-hit latencies, not cold prefill.**
  The bench's fixed-seed random prompts were resident in the 100 GiB offload
  pool from earlier runs the same day (GPU prefix hits 72-79%, external hits
  climbing 11%→36% over the run). Cold prefill on the NVFP4 stack measured
  ~10.1K tok/s in the 17:46 probe run; the fp8-era comparators (3217ms/10471ms)
  are cold and not comparable to this row.
- Decode is the MTP output rate: acceptance 2.34-2.48 tokens/step on the
  random dataset (per-position ~66-74 / 42-48 / 27-32%). Median ITL
  17.3-17.7ms is the step cadence (bursts of ~2.4-2.5 tokens per step);
  median TPOT 7.07-7.52ms is the per-token spacing. The cutover validation's
  natural-text numbers were higher (2.81 tokens/step → 157-165 tok/s, ITL
  6.0-6.4ms) — the random dataset accepts less.
- PSU over the bench window: 238W idle floor, 1.18kW avg, 1.27kW peak
  (fp8 era: 1.28kW avg / 1.76kW peak — NVFP4 + gmu 0.69 draws substantially
  less).

### Benchmarks (2026-09-27, 87feab60 build, NVFP4+MTP3, offload on, b12x oneshot — c=1, 16 prompts per cell, zero failures)

`vllm bench serve` (same protocol) against the 87feab60 deployment with the
autotune fix (vllm `022c5085fa`) and no skip-ops env — job
`vllm-bench-manual-spawn-mukcrykg-53obk`, pod `-q44mv`, 21:51-22:20 UTC, on
a fresh boot: empty GPU cache and offload pool, so **every TTFT is cold
prefill** (the 09-26 run served warm prefix hits and is not TTFT-comparable).

| Input | Output | Decode (tok/s) | Median TTFT | Median ITL | Median TPOT | Accept len |
|---|---|---|---|---|---|---|
| 2048 | 256 | 129.43 | 189ms | 16.47ms | 6.93 | 2.36 |
| 8192 | 1024 | 127.36 | 702ms | 16.54ms | 7.17 | 2.31 |
| 32768 | 4096 | 130.75 | 2468ms | 16.62ms | 7.21 | 2.36 |
| 131072 | 8192 | 128.92 | 9302ms | 16.79ms | 6.78 | 2.53 |

- Decode holds 127-131 tok/s at every length — within noise of the 09-26
  skip-ops build (129-135 warm) — and the leader-only-tuning removal costs
  nothing measurable while the tuned MoE ops raise peak power: PSU over the
  window 262W idle floor, 1.15kW avg, **1.68kW peak** (09-26 warm run
  peaked at 1.27kW; this run prefills cold at 128K).
- Cold prefill rates (input/TTFT): ~10.9K tok/s at 2K, 11.7K at 8K, 13.3K at
  32K, 14.1K at 128K — the chunked-prefill amortization curve on the NVFP4
  stack; the fp8-era cold comparator was 12.5K tok/s at 128K (10471ms).

## GLM-5.3-Flash fp8 era — historical (2026-09-23 0.30 rebase; superseded by NVFP4+MTP 2026-09-26)

Deployed via k8s-gitops `stage3/apps/vllm.yaml`; image
`registry.ocnr.org/infra/vllm:0.30.0-sm120-cu130@sha256:f8cdce48` built from
the `0.30` branch (rebased 2026-09-23 onto upstream vLLM `main` at
`9f07d023d0`, past v0.30.0 — GLM-5.3-Flash model support is native upstream
since vllm-project #53906, the ZJY0516 fork is retired). On top of upstream:
the ocnr SM120 NoPE sparse-MLA port (fp8 + FlashInfer zero-pad, backend
priority, buffer pin), the b12x PCIe oneshot allreduce integration (b12x
1.3.0, CuTe DSL — no native extension build), the hybrid-state prefix-cache
fix #55601 and the indexer-prefill-workspace right-size #55222 in **both
halves** (the glm5next call-site fix plus the chunker-budget commit — still
open upstream). The FlashInfer autotune-sync ocnr commit was **dropped** in
the 2026-09-22 rebase: upstream now syncs autotune tactics natively (leader-only
autotune + result broadcast, both in the generic warmup and the SM120
sparse-MLA decode warmup), and `VLLM_FLASHINFER_AUTOTUNE_PROCESS_GROUP` is
no longer set. KV offloading is enabled on upstream's native backend (see
*kv-offload status* below).

Boot verification of the 2026-09-23 rebase (f8cdce48) on the fp8 config was
completed 2026-09-26 (backend `FLASHINFER_MLA_SPARSE_SM120` + `fp8_ds_mla`,
available KV 7.68 GiB, auto-fit full 1M at 1,064,361 tokens, autotune + CUDA
graphs FULL 3/3 / PIECEWISE 4/4 clean, health 200 OK, 1,677 successful
requests / 0 errors); the config was superseded by the NVFP4+MTP cutover the
same day. Previous build (a9725bcb), boot-verified 2026-09-22
(pod `vllm-0`, uid `404ab979`, after raising `--limit-mm-per-prompt` to 20
images): backend
`FLASHINFER_MLA_SPARSE_SM120` + `fp8_ds_mla`, available KV 7.68 GiB, auto-fit
`full model context length 1048576 fits`, autotune + CUDA graphs (FULL 3/3,
PIECEWISE 4/4) clean, chat completions 200 OK.

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
  --limit-mm-per-prompt '{"image": 20, "video": 0}' \
  --default-chat-template-kwargs '{"thinking": true}'
```

Env: `HF_HUB_OFFLINE=1`, `NCCL_P2P_LEVEL=NODE`, `RAYON_NUM_THREADS=4`,
`OMP_NUM_THREADS=4`, `MAX_JOBS=32`,
`VLLM_ENABLE_PCIE_ALLREDUCE=1` (b12x oneshot replaces NCCL-SHM for decode-size
all-reduces; >8 MiB prefill collectives stay on NCCL). FlashInfer autotune
stays enabled; tactic consistency across TP ranks is upstream-native
(leader autotunes, results broadcast) — the old
`VLLM_FLASHINFER_AUTOTUNE_PROCESS_GROUP` opt-in is gone.
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
("full model context length 1048576 fits", 2026-09-18 01:29 UTC). The 0.30
production boot raises the limit to `image: 20` and still auto-fits full 1M
at the same 7.68 GiB (2026-09-22).

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
on the branch in both halves: the call-site fix plus the chunker budget
sized in compressed rows (without the chunker half, a step can admit up to
`compress_ratio`x more rows than the workspace holds at ≥64 requests near
max-model-len; not reachable at the production `--max-num-seqs 4`).

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
was the production profile until the 2026-09-26 NVFP4+MTP cutover — the
table below is the no-offload reference build.

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
