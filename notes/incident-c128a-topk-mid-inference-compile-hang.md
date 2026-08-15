# Incident: rank-0 C128A topk-metadata JIT recompile hang at long context — RESOLVED

## Status

- **2026-08-15** — Jammed resuming a 632,932-token conversation; `dump-jam-state.sh`
  showed a rank-0 C128A metadata-build stall (worker TP0 wedged inside the Triton
  kernel launch while TP1–3 sat in `dequeue` and the engine waited for the step).
- **2026-08-15** — Fix landed in `infra/vllm` on `main`/`0.27`
  (`6841036cc`, "ocnr: fix C128A topk metadata kernel mid-inference JIT recompile
  hang"): `do_not_specialize` on the kernel's runtime scalar args + a builder
  `__init__` warmup. Validation on this box: **pending** — need one >524K request
  on the rebuilt image to confirm no mid-inference compile.

## Symptom

A single >500K-context resume **hung** the engine (did not self-recover, manual pod
restart). The `dump-jam-state.sh` capture (`/home/vllm/jam-20260815T204729Z-vllm-0/`):

- **Worker_TP0** (GPU 0): MainThread in
  `execute_model → _build_attn_group_metadata → DeepseekV4FlashMLAMetadataBuilder.build
  → _build_c128a_metadata → build_c128a_topk_metadata →
  _build_c128a_topk_metadata_kernel[...]` — i.e. inside the Triton kernel
  launch/compile wrapper (`triton/language/core.py _unwrap_if_constexpr`).
- **Worker_TP1/2/3**: all back in `input_broadcaster.dequeue()` (shm_broadcast
  `acquire_read`) — they already finished the batch.
- **EngineCore**: waiting in `get_response` for the step.
- All 4 GPUs ~99% util (workers busy/collective-spinning around the stalled rank).

So rank 0 never returned its step while the other three moved on → engine wedge.

## Root cause

`_build_c128a_topk_metadata_kernel` (DeepSeek-V4 C128A sparse-MLA topk metadata)
takes its bounds/strides as **plain scalar int args, which Triton specializes by
value into the JIT compile key**. Three of them track `active_topk_width`
(`max_compressed_tokens`, `global_decode_stride`, `prefill_local_stride`), which
jumps at each power-of-two boundary of `max_seq_len//128`; `num_decode_tokens`
varies every step; `block_table_stride` differs from any warmup dummy. Because these
are baked into the key, a long context that pushes `active_topk_width` to the
`c128a_max_compressed` cap (8192) — first reached at `max_seq_len > 128·4096 =
524,288`; the jam was at 632,932 — triggered a **fresh mid-inference JIT recompile
that warmup never covered**, on one rank only. The other ranks had already compiled
it (or loaded it from the shared persistent `/home/vllm/.triton` cache), so rank 0
stalled alone → the rest desynced and the engine hung.

This is the same "per-rank divergence" family as the KV-offload deadlock and the
Xid-69 autotune crash, but at a **Triton JIT-compile** site rather than an
NCCL/FlashInfer site. Notably **different from both prior incidents**: the stuck rank
here is rank 0 / GPU 0 (`01:00`), not rank 1 / GPU 1 (`21:00`).

### Evidence

- The fresh (restarted) pod logs the exact mechanism in `jit_monitor`:
  ```
  (Worker_TP0) WARNING [jit_monitor.py:135] Triton kernel JIT compilation during
  inference: _build_c128a_topk_metadata_kernel. This causes a latency spike; consider
  extending warmup to cover this shape/config.
  ```
- The 632,932-token context maps precisely: `632932 // 128 = 4944 →
  next_power_of_2 = 8192` (the cap, never warmed).

## Fix (deployed commit `6841036cc`, `main` + `0.27`)

In `vllm/models/deepseek_v4/sparse_mla.py`:

1. **`@triton.jit(do_not_specialize=[...])`** on `_build_c128a_topk_metadata_kernel`
   for the runtime scalars (`global_decode_stride`, `prefill_local_stride`,
   `max_compressed_tokens`, `num_decode_tokens`, `block_table_stride`). All are used
   as runtime values inside the kernel, so this is semantically a no-op — it just
   gives the kernel a **single stable compile key** (only pointer dtypes +
   `BLOCK_SIZE` constexpr) so it never recompiles at the long-context width.
2. **`_warmup_c128a_topk_kernel()`** in `DeepseekV4FlashMLAMetadataBuilder.__init__`
   compiles the kernel once at the maximum width (dtype-matched to the real call
   sites: positions/slot_mapping int64, block_table/req-index/topk buffers int32),
   so it is a startup cost rather than a runtime hang.

### Why not the other two fixes' shape

A collective barrier (KV-offload fix) would not help — rank 0 is genuinely stuck
compiling, not racing into a collective. PR #3187-style autotune process-group sync
(Xid-69 fix) targets FlashInfer autotune; this is a plain `@triton.jit` kernel with
no autotune, so the Triton-side fix is `do_not_specialize` + warmup.

## Verification

- `triton.jit(do_not_specialize=[...])` accepted by triton 3.7.1.
- A mirror kernel ran correctly on GPU across differing scalar values (limit 8/32/100,
  stride 1/5) and the on-disk Triton cache did **not** grow → single compile, no
  recompile.
- **Pending on hardware**: rebuild the image, bump the digest in
  `k8s-gitops/stage3/apps/vllm.yaml`, roll `vllm-0`, then drive one >524K request.
  Set `jit_monitor_mode=error` (currently `warn`) for the confirmation run — it
  raises with the kernel name + exact compile key on any mid-inference compile, with
  none of the throughput hit of `CUDA_LAUNCH_BLOCKING`. If it aborts on
  `_build_c128a_topk_metadata_kernel`, that confirms the pre-fix failure and that the
  fix eliminated it.

## Notes for next time

- Long-context hang where **one** worker is wedged in a Triton launch wrapper
  (`_unwrap_if_constexpr`) while the others sit in `dequeue` ⇒ suspect a
  mid-inference JIT recompile of a width-dependent kernel, not a collective deadlock.
- GPU-0- vs GPU-1-rank attribution differs from the earlier two incidents — don't
  assume the same physical card/rank each time.
- `jit_monitor_mode=error` is the cheap way to pin these; reserve
  `CUDA_LAUNCH_BLOCKING=1` for async-CUDA attribution (huge throughput hit).
