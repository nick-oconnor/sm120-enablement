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
  **Resolved** by disabling offloading.
- [`incident-longcontext-xid69.md`](incident-longcontext-xid69.md) — hard CUDA
  `unspecified launch failure` (Xid 69) on long-context requests. Hardware
  fully eliminated; **open**, pointing at a software kernel path on rank 1.

## TL;DR current status

- **Serving works**: full 1M context, fp8 KV + Triton attn + EP + flashinfer_cutlass
  NVFP4 MoE, tool calls + reasoning validated. See `serving-config.md`.
- **KV offloading**: disabled in production. **Patch available locally**
  (`VLLM_KV_OFFLOAD_COLLECTIVE_BARRIER=1`, opt-in, off by default) — combines
  stream-side ordering + a host-side `dist.barrier()` after `start_load_kv` to
  prevent the TP/EP rank desync documented in `incident-kv-offload-deadlock.md`.
  Offloading remains unused because 1M context fits in GPU KV at 0.97 util.
- **Xid-69 crash**: open. Upstream candidate identified — **FlashInfer
  `2c0d595f` (PR #3187)** adds `set_autotune_process_group()` which fixes the
  same bug family (per-rank async autotune tactic pick → desync → illegal
  kernel launch) on identical hardware (RTX PRO 6000 SM120, author's repro
  box). Wire-up needed in vLLM warmup (~10 lines, try-import guarded). See
  `incident-longcontext-xid69.md` for the full patch plan. Both pieces required:
  confirm faulting site with `CUDA_LAUNCH_BLOCKING=1`, then bump FlashInfer
  past `2c0d595f` and wire the setter in warmup.
- All hardware checks (DCGM `-r 3`, ECC, PCIe Gen5, ASPM) remain clean.
