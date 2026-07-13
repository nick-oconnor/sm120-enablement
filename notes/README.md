# ocnr notes — MiniMax-M3-NVFP4 on SM120

Operational deep-dive for serving **`nvidia/MiniMax-M3-NVFP4`** on the ocnr vLLM
fork (`registry.ocnr.org/infra/vllm`) — the build/deploy config and
production-incident post-mortems behind the top-level [writeup](../README.md).

## Hardware / platform

| | |
|---|---|
| GPUs | 4× **RTX PRO 6000 Blackwell Max-Q Workstation Edition** (GB202GL, device `2bb4`) |
| Compute capability | **SM120 / cc 12.0** (Blackwell) |
| Interconnect | **PCIe 5.0 only — no NVLink** (custom all-reduce auto-disabled → PYNCCL) |
| Board | Gigabyte **TRX50 AI TOP**, AMD Threadripper |
| Driver / CUDA | 590.48.01 / CUDA 13.1 |
| vLLM | fork `registry.ocnr.org/infra/vllm`, version `0.24.0+sm120.cu131` |
| Model | `nvidia/MiniMax-M3-NVFP4` — VL, `MiniMaxM3SparseForConditionalGeneration`, MIXED_PRECISION (NVFP4 experts + MXFP8), 128 experts, block-sparse attention + lightning indexer, 1,048,576 max context |

### GPU ↔ PCI ↔ physical slot map

| CUDA idx | PCI (host) | Root complex | Physical slot | Serial |
|---|---|---|---|---|
| 0 | `0000:01:00.0` | `0000:00` | PCIE3 | …71748 |
| **1** | **`0000:21:00.0`** | `0000:20` | **PCIE4** | **…70908** |
| 2 | `0000:81:00.0` | `0000:80` | *(MCIO/SlimSAS — not in SMBIOS slot table)* | …71758 |
| 3 | `0000:c1:00.0` | `0000:c0` | PCIE2 | …70981 |

> GPU 1 (PCIE4, serial …70908) is the card that recurs in the production
> incidents below. As of this writing its hardware has been **cleared** (see
> the Xid-69 incident doc) — the recurring failures point at software.

## Index

- [`serving-config.md`](serving-config.md) — the validated launch config, the
  upstream PR stack, and every ocnr code fix (build, quantization, tool parser,
  reasoning) needed to serve M3 NVFP4 on SM120.
- [`incident-kv-offload-deadlock.md`](incident-kv-offload-deadlock.md) — host-RAM
  KV offloading deadlocks at long context under TP/EP + async scheduling.
  **Resolved** by stream-side ordering + TP barrier in `OffloadingConnector`;
  offload flags re-enabled in production 2026-07-11.
- [`incident-longcontext-xid69.md`](incident-longcontext-xid69.md) — hard CUDA
  `unspecified launch failure` (Xid 69) on long-context requests. Hardware
  fully eliminated; **open**, pointing at a software kernel path. Fix candidate
  (FlashInfer PR #3187 sync) deployed 2026-07-11; attribution repro
  (`CUDA_LAUNCH_BLOCKING=1`) still required to confirm site.
- [`incident-v0251-sparse-attn-regression.md`](incident-v0251-sparse-attn-regression.md) —
  the `v0.25.1` upgrade (`next`) crashes on the first multi-sequence prefill in
  the NVFP4 MoE `gemm2` (null-pointer TMA descriptor, CUDA 700). **Resolved** —
  root-caused by local docker bisection to upstream **PR #47502** (M3 sparse-attn
  indexer / token-major `topk_indices_buffer`); reverted on `next`. The MoE crash
  was a downstream report; corruption originated in the sparse-attention indexer.

## TL;DR current status

- **Serving works** (on `0.24.0`, current production): full 1M context, fp8 KV +
  Triton attn + EP + flashinfer_cutlass NVFP4 MoE, tool calls + reasoning
  validated. See `serving-config.md`.
- **`v0.25.1` upgrade** (`next`): blocked then **fixed**. Upstream PR #47502 (M3
  sparse-attn indexer) crashed multi-sequence prefills in the NVFP4 MoE `gemm2`;
  reverted on `next` and validated locally (22+ concurrent, clean). Not yet
  deployed. See `incident-v0251-sparse-attn-regression.md`.
- **KV offloading**: **deployed**. Offload flags re-enabled in
  `stage3/apps/vllm.yaml` (`56d81ffc`, 2026-07-11) with the rank-desync
  fix activated via `VLLM_KV_OFFLOAD_COLLECTIVE_BARRIER=1`. See
  `incident-kv-offload-deadlock.md` for the timeline and validation status.
- **Xid-69 crash**: open — **fix candidate deployed**. FlashInfer pin past
  PR #3187 (`2c0d595f`) + `VLLM_FLASHINFER_AUTOTUNE_PROCESS_GROUP=1` wire-up
  shipped 2026-07-10/11. Caveat: only addresses the FlashInfer-autotune
  failure path; the Triton `_topk_index_kernel` path remains unaddressed.
  Repro (CUDA_LAUNCH_BLOCKING=1) still required to confirm which autotuner
  is the actual fault site. See `incident-longcontext-xid69.md`.
- All hardware checks (DCGM `-r 3`, ECC, PCIe Gen5, ASPM) remain clean.
