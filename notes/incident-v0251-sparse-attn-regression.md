# Incident: v0.25.1 upgrade crash — MiniMax-M3 sparse-attn indexer (PR #47502) — RESOLVED

## Status

| Date | Event |
| --- | --- |
| 2026-07-12 | Rebased the fork onto upstream `v0.25.1` (branch `next`, `0.25.1+sm120.cu131`). Boots clean, then crashes on the first real batched prefill in the NVFP4 MoE `gemm2`. |
| 2026-07-13 | Rolled production back to the working `0.24.0` build. Ruled out deps, rebase corruption, stale JIT cache, FlashInfer autotune, and PR #47631. |
| 2026-07-13 | Root-caused by local `docker` A/B bisection to upstream **PR #47502**; reverted on `next` (`7a94c181a`). Validated: same repro that crashed every time now serves 22+ concurrent requests clean. |

Production stayed on `0.24.0` throughout; `next`/`0.25.1` was an in-progress upgrade.

## Symptom
`next` (`0.25.1+sm120.cu131`) starts **cleanly** — model loads `modelopt_mixed`
(NVFP4 experts resolve, not bf16), CUDA graphs capture, server listens on :8000 —
then dies on the **first multi-sequence prefill batch**, all 4 ranks:

```
[TensorRT-LLM][ERROR] Assertion failed: Failed to initialize cutlass TMA WS grouped gemm.
  .../flashinfer/0.6.13/120f/.../cutlass_kernel_file_gemm_grouped_sm120_M256_BS_group0.generated.cu
Error: Failed to initialize the TMA descriptor 700   (CUDA 700 = illegal memory access)
  TMA desc: globalDim (128,256)  gmem_address 0  globalStrides (0,0,0,0,0)
→ EngineDeadError
```

The fault is reported in the NVFP4 FlashInfer-CUTLASS MoE `gemm2` (down-proj)
grouped GEMM with a null-pointer TMA descriptor — which is why it *looked* like a
MoE / FlashInfer / autotune bug for several rounds. It is not: the null descriptor
is a **downstream report**. The corruption happens earlier in the same forward
(instrumenting `gemm2` showed all its input tensors non-null, and a `bincount` on
`topk_ids` placed *before* the `gemm2` call itself faulted with the illegal
access — so an earlier kernel in that forward already corrupted the context).

## Trigger
A prefill batch containing **≥2 sequences**. Reproduced deterministically with 6
concurrent chat requests. A **single** request — even a full 8192-token prefill —
does **not** trip it, and neither does the dummy-profiling routing at startup
(only ~4 experts selected). This is why single-request smoke tests passed and the
crash only showed under real agent concurrency (opencode/hermes).

## Root cause — upstream PR #47502
`cf3df3368` "[Minimax-M3] Using tok_sparse_select from MSA instead of triton
kernels", new in `v0.25.1`. It flips the shared `topk_indices_buffer` from
head-major `[num_index_heads, total_q, topk]` to **token-major**
`[total_q, num_index_heads, topk]` and rewrites the sparse-attention indexer
(`index_topk`, `indexer_msa`, `sparse_attn`) to write its native
`[token, head, topk]` top-k. On **SM120 / EP=4** with a multi-sequence batch that
new indexer path corrupts memory; the corruption surfaces two subsystems later in
the MoE `gemm2` descriptor build.

MSA (`fmha_sm100`) is **SM100-only** and unused on SM120 (the Triton
`_topk_index_kernel` / `_gqa_sparse_fwd_kernel` path runs here — see the Xid-69
doc), so #47502's intended benefit never applied to this hardware anyway.

## How it was root-caused (local docker bisection)
GPUs were freed and `next` built locally, so the crash could be driven directly
instead of waiting on deploys:

1. **Reproduced**: 6 concurrent requests → crash every time; a single request
   (incl. 8192 tokens) → clean. Multi-sequence is the trigger.
2. **Instrumented** `flashinfer_cutlass_moe.py::apply` (bind-mounted over the
   image, no rebuild): gemm2's weights/scales/activations/output were all
   non-null with correct shapes; the pre-gemm2 `bincount` faulted → corruption is
   upstream of the MoE.
3. **Decisive A/B** (identical repro, identical FlashInfer 0.6.13 / CUTLASS
   `e8ecfad`):
   - `0.24.0` image: 22+ concurrent requests, **no crash**.
   - `0.25.1` image: crashes on the first ≥2-seq batch, **every time**.
   - `0.25.1` + bind-mounted **pre-#47502 revert**: 22+ concurrent requests,
     **no crash**, coherent output.

## Fix (on `next`, `7a94c181a`)
Revert #47502's M3 changes — restore the pre-#47502 (`b71218107`) Triton indexer
(`common/indexer.py`, `common/ops/index_topk.py`, `common/ops/sparse_attn.py`,
`common/ops/__init__.py`, `nvidia/indexer_msa.py`, `nvidia/sparse_attention_msa.py`)
and the head-major `topk_indices_buffer` in `nvidia/model.py`.
`fmha_sm100.cmake` is left at `v0.25.1` (SM100-only, not built for arch `12.0a`).

## Ruled out (each takes several rounds off the suspect list)
- **Dependency bump** — FlashInfer `2c0d595f` (=0.6.13), CUTLASS `e8ecfad`,
  torch 2.11.0, CUDA 13.1.1 are all identical between the `0.24.0` and `0.25.1`
  builds. The crashing kernel is byte-identical on both.
- **Rebase corruption** — `next` == `v0.25.1` + only the intended ocnr files; the
  whole flashinfer_cutlass / modular_kernel / routed_experts / EP prepare-finalize
  path is pristine `v0.25.1`.
- **Stale FlashInfer JIT cache** — crashes on a cleared cache too (the PVC-persisted
  `/home/vllm/.cache/flashinfer/0.6.13/120f` is version/arch-keyed, not
  commit-keyed, so it *was* a real hazard — just not this bug).
- **FlashInfer autotune** — crashes identically with `--no-enable-flashinfer-autotune`
  (a different CUTLASS tactic, `M128_BS` vs `M256_BS`; the `group0/group2` in the
  kernel filenames are tactic variants, not expert indices).
- **PR #47631** (cross-layer allreduce-norm fusion) — the other new M3 change in
  range. Its `reduce_results=True` workaround did **not** fix the crash, and under
  EP the MoE all-reduce goes through the all2all combine so #47631's deferral is a
  no-op for MoE layers. Left intact.

## Relationship to the Xid-69 incident
Same subsystem: the **M3 sparse-attention indexer** on SM120. The Xid-69 doc
flagged the Triton `_topk_index_kernel` path as the SM120 code path and still
open; #47502 rewrote exactly that indexer. This incident is a clean upstream
regression in that indexer under multi-sequence batching — reverting to the
pre-#47502 Triton path is the fix here, and keeps SM120 on the indexer the Xid-69
work was characterizing.
