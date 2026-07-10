# Incident: long-context crash — CUDA unspecified launch failure (Xid 69) — OPEN

## Symptom
Hard crash (not a hang) on long-context requests. A worker throws a
context-corrupting CUDA error → the worker dies → EngineCore dies → pod restart.

```
Autotuning kernel _topk_index_kernel ... best config BLOCK_SIZE_K: 64  (immediately before)
WorkerProc hit an exception. (Worker_TP1_EP1)
  torch.AcceleratorError: CUDA error: unspecified launch failure   (cudaErrorLaunchFailure)
  at gpu_input_batch.py:1043  update_async_output_token_ids → async_copy_ready_event.synchronize()
[rank1] ProcessGroupNCCL watchdog terminated: CUDA error: unspecified launch failure
→ EngineCore fatal → EngineDeadError → shutdown
```

Kernel log:
```
NVRM: Xid (PCI:0000:21:00): 69, pid=..., name=python3, Class Error:
      channel 0x9, Class 0000cec0, Offset 0x184, Data ffffffff, ErrorCode 0x4
NVRM: GPU Board Serial Number: 1791526070908
```

## Key facts
- **Xid 69** = "Graphics Engine class error" — a compute kernel submitted an
  invalid op to the engine (illegal kernel operation), *not* an ECC/memory or
  bus Xid.
- The CUDA error is **async** — reported at the next `synchronize()`, so the
  crash site (`update_async_output_token_ids`) is *not* the faulting kernel. The
  last GPU activity logged was the **`_topk_index_kernel` autotune** — the
  MiniMax-M3 **lightning-indexer top-k block selector** for block-sparse attention
  (`vllm/models/minimax_m3/common/ops/index_topk.py`). Prime suspect, but async
  attribution means it isn't proven.
- Triggered by **long context** (seen at ~51K and ~166K tokens); short requests
  are fine → this is why it's "random" (only long opencode sessions hit it).
- Consistently **rank 1 / physical GPU 1** (PCI `0000:21:00.0`, serial …70908,
  PCIE4).

## Hardware elimination (all clean)

| Check | Result | Conclusion |
|---|---|---|
| `dcgmi diag -r 3 -i 1` | **Pass** — software, memory, diagnostic, nvbandwidth, pcie, targeted_stress, targeted_power | GPU 1 die/VRAM/compute/link healthy under sustained load |
| ECC | Enabled on **all 4** GPUs; clean | no VRAM bit-flips (also a live detector now) |
| PCIe link speed | idle 2.5 GT/s (Gen1) → **32 GT/s (Gen5) x16 under load** | link trains to full spec; idle Gen1 is benign dynamic speed-scaling, not a fault |
| ASPM | `LnkCtl: ASPM Disabled` on endpoint **and** root port; L1 substates off | no link sleep/wake path to glitch — ASPM ruled out |

**Caveat on DCGM/gpu-burn:** they run *generic* sustained-load kernels (link pinned
at Gen5), so they don't exercise (a) the M3 sparse-attention kernels specifically,
nor (b) idle→wake transitions. They make gross hardware unlikely but can't fully
exclude a rare intermittent fault under the real workload.

## Current conclusion
Hardware, VRAM, PCIe link, and power management are all **eliminated**. The Xid 69
class error is a **software** illegal-kernel-op, in the M3 sparse-attention indexer /
long-context path executed on rank 1.

## Next step (to get a fix)
Reproduce once with launch-blocking to convert the async failure into a precise
kernel/line attribution:
```yaml
env:
- name: CUDA_LAUNCH_BLOCKING
  value: "1"          # TEMPORARY — serializes all kernels, big throughput hit
```
Roll `vllm-0`, drive one long-context request until it faults, capture the
traceback (expected: `index_topk` / sparse-attention). Optionally add
`compute-sanitizer` for the exact illegal address. Then patch the kernel + add a
test. Revert the env var after capture.

Note: the indexer's K-tile load *is* bounds-masked (`i+off_k < valid_blocks`), so
it's not a naive overflow — the fault is subtler (e.g. `valid_blocks` vs the score
buffer's `max_block`, a bitonic-merge indexing edge, or a different kernel in the
same step). Don't guess without the launch-blocking/sanitizer attribution.

## Diagnostic playbook (for next time)
- No host `dcgmi`/`py-spy`/`docker` on the node; containerd only. Runtimes:
  `crictl` (k8s.io namespace), `ctr`. GPU test images available:
  `registry.ocnr.org/infra/nvidia-devel`, the vllm image (has torch+CUDA).
- **py-spy** (static binary, run as the pod's `vllm` user): dumps all workers —
  the fastest way to see whether it's a collective deadlock vs a crash.
- **Xid**: `sudo dmesg -T | grep -iE "xid|nvrm"`. Xid PCI address is the
  container-index-independent ground truth for *which physical card*.
  Timestamps: dmesg is local (PDT), container logs are UTC (−7h).
- **GPU utilization pattern**: 3×100% + 1×0% ⇒ collective deadlock (one rank out).
- **Rank-vs-card** discriminator (staged, not yet run): set
  `CUDA_VISIBLE_DEVICES=1,0,2,3` so logical rank 1 runs on a *different* physical
  card; if the next Xid follows PCI `21:00` it's the card, if it follows rank 1
  it's software. (Hardware is already cleared, so this is now optional.)
- **PCIe link under load**: `nvidia-smi --query-gpu=index,pcie.link.gen.gpucurrent,pcie.link.gen.max --format=csv`.
