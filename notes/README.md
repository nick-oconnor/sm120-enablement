# ocnr notes — vLLM on SM120

Operational deep-dive behind the top-level [writeup](../README.md): the
current serving config (GLM-5.3-Flash, see
[`serving-config.md`](serving-config.md)) and production-incident
post-mortems. The M3-era sections below (hardware table as of 2026-08,
vLLM `0.25.1+sm120.cu131`) are historical.

## Hardware / platform

| | |
|---|---|
| GPUs | 4× **RTX PRO 6000 Blackwell Max-Q Workstation Edition** |
| Compute capability | **SM120 / cc 12.0** (Blackwell) |
| Interconnect | **PCIe 5.0 only — no NVLink** |
| Board | Gigabyte **MH53-G40**, AMD Threadripper |
| Driver / CUDA | 590.48.01 / CUDA 13.1 |
| vLLM | ocnr fork, version `0.25.1+sm120.cu131` |
| Model | `nvidia/MiniMax-M3-NVFP4` |

- GPU die: (GB202GL, device `2bb4`).
- Interconnect: (custom all-reduce auto-disabled → PYNCCL).
- Model: VL `MiniMaxM3SparseForConditionalGeneration`, MIXED_PRECISION
  (NVFP4 experts + MXFP8), 128 experts, block-sparse attention + lightning
  indexer, 1,048,576 max context.

### GPU ↔ PCI ↔ physical slot map

| CUDA idx | PCI (host) | Root complex | Physical slot | Serial |
|---|---|---|---|---|
| 0 | `0000:01:00.0` | `0000:00` | PCIE3 | …71748 |
| **1** | **`0000:21:00.0`** | `0000:20` | **PCIE4** | **…70908** |
| 2 | `0000:81:00.0` | `0000:80` | | …71758 |
| 3 | `0000:c1:00.0` | `0000:c0` | PCIE2 | …70981 |

> Slot for GPU 2 is MCIO/SlimSAS — not in SMBIOS slot table.
>
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
- [`incident-v0251-sparse-attn-regression.md`](incident-v0251-sparse-attn-regression.md)
  — the `v0.25.1` upgrade (`next`) crashes on the first multi-sequence prefill in
  the NVFP4 MoE `gemm2` (null-pointer TMA descriptor, CUDA 700). **Resolved** —
  root-caused by local docker bisection to upstream **PR #47502** (M3 sparse-attn
  indexer / token-major `topk_indices_buffer`); reverted on `next`. The MoE crash
  was a downstream report; corruption originated in the sparse-attention indexer.
- [`incident-kv-offload-cache-corruption.md`](incident-kv-offload-cache-corruption.md)
  — silent progressive KV-cache corruption on GLM-5.3-Flash with APC + native
  offloading (garbled output, then empty responses; all finishes reported as
  clean `stop`). **Resolved** — offload patches dropped from the branch and
  flags removed in production 2026-09-11; fingerprint matches upstream #50454.

## TL;DR current status

- **Serving (current)**: GLM-5.3-Flash on `0.29.0-sm120-cu130` (image
  `94a85a96`, branch `bea5f74795`), full 1M context, fp8 KV, b12x PCIe
  one-shot all-reduce. See `serving-config.md`.
- **KV offloading**: **removed** (2026-09-11) after silent cache corruption on
  the previous build. See `incident-kv-offload-cache-corruption.md`; the M3-era
  deadlock note below covers the earlier 2026-07 episode.
- **Xid-69 crash (M3-era)**: open — **fix candidate deployed**. FlashInfer pin past
  PR #3187 (`2c0d595f`) + `VLLM_FLASHINFER_AUTOTUNE_PROCESS_GROUP=1` wire-up
  shipped 2026-07-10/11. Caveat: only addresses the FlashInfer-autotune
  failure path; the Triton `_topk_index_kernel` path remains unaddressed.
  Repro (CUDA_LAUNCH_BLOCKING=1) still required to confirm which autotuner
  is the actual fault site. See `incident-longcontext-xid69.md`.
- All hardware checks (DCGM `-r 3`, ECC, PCIe Gen5, ASPM) remain clean.
