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

## Upstream candidate fix (FlashInfer PR #3187, ranked the most plausible cause)

**Strong match — same bug family as the KV-offload deadlock** (per-rank async
decisions before TP/EP collectives), at a different code site. Pinned
`flashinfer-ai/flashinfer` is `b5ac097e`; commit `2c0d595f` (Thanhhao, 2026-07-10,
PR #3187) adds `set_autotune_process_group(group)` which `all_reduce`s measured
kernel timings across a process group before the autotuner's `argmin` — every
rank therefore picks the same tactic.

Quoting the PR description verbatim, the failure mode:
> *Different tactics chosen in step 1 produce different scratch shapes in step 2,
> and `ncclCommWindowRegister` requires identical sizes across ranks or it
> deadlocks.*

This matches our incident shape exactly:
1. `_topk_index_kernel` autotune runs per-rank with locally-measured timings →
   different ranks may pick different configs under JIT/J-cache variance
2. Next op (NCCL collective in MoE all-to-all, or the next Triton kernel in
   attention) sees mismatched state on at least one rank
3. Async CUDA error surfaces at the next `synchronize()` as
   `cudaErrorLaunchFailure` → Xid 69

Validation on hardware identical to ours: FlashInfer PR #3614 author explicitly
ran the reproducer on *"This box is SM120 (RTX PRO 6000)"*.

### Wire-up needed in vLLM (try-import guarded)

```python
from flashinfer.autotuner import set_autotune_process_group
set_autotune_process_group(tp_group.cpu_group)   # gloo subgroup; safe
try:
    with autotune(True):
        run_warmup_forwards()
finally:
    set_autotune_process_group(None)
```

Without the wire-up, bumping FlashInfer alone is half the fix. The wire-up is
~10 lines around the warmup autotune block. Combined with a FlashInfer pin bump
past `2c0d595f` (next release tag is `f2f9646e` v0.6.15), this is the path to
turning the Xid-69 incident from "open" to "fixed."

### Related FlashInfer fixes (newer than pin, take with the bump)

| SHA | PR | What | Relevance |
|---|---|---|---|
| `2c0d595f` | #3187 | `set_autotune_process_group` | **primary fix for this incident** |
| `59b1c4d7` | #3614 | MXFP8 cutlass MoE profiler autotune crash (null SF pointer) | defense in depth — same family, different kernel |
| `596d1a06` | #3840 | fail loud on missing `(numExperts, topK)` routing tier | M3 (128/8) is covered; cheap insurance |
| `ce52279d` | #3863 | clamp `CTA_TILE_Q` for SM120 99 KiB smem cap | not in topk path, but useful to know about SM120 limit |
| `783914f9` | #3687 | autotuner memory leak | hygiene |

### Why we still need CUDA_LAUNCH_BLOCKING attribution first

PR #3187 fixes the autotune-tactic-divergence mechanism. The Xid-69 doc's
"primacy suspicion" of `index_topk.py:_topk_index_kernel` is consistent with
that mechanism but unproven — the crash is async, the launch-blocking/sanitizer
attribution is what converts "plausible mechanism" into "confirmed site."
Both pieces are needed: confirm the faulting site is in the autotune-affected
path, then take PR #3187 to prevent it.

### ⚠ Caveat — does PR #3187 target the right autotuner?

`_topk_index_kernel` is decorated with **Triton autotune** (`@triton.autotune`,
`key=["BLOCK_SIZE_T"]` — `BLOCK_SIZE_T=next_power_of_2(topk)`, M3 default
`sparse_topk_blocks=16` → BLOCK_SIZE_T=16), not FlashInfer autotune.

PR #3187 syncs **FlashInfer's** autotuner only. Triton autotune is a separate
subsystem with no equivalent process-group sync concept. If the actual
faulting site is the Triton `_topk_index_kernel` autotune (and not just
"the last console output before the async error surfaced"), PR #3187
won't fix it — the symptom pattern matches but the mechanism is in the
wrong autotuner.

**Repro path to disambiguate:**
1. Run with `CUDA_LAUNCH_BLOCKING=1` per the diagnostic playbook.
2. If the faulting kernel reports as `index_topk.py:_topk_index_kernel` →
   it's the Triton autotune; PR #3187 is not the right fix; we'd need
   either (a) pre-warm + pin the Triton config, or (b) a Triton-side
   sync (much harder — Triton has no `set_autotune_process_group`).
3. If the faulting kernel reports as a FlashInfer MoE op
   (`trtllm_fp4_block_scale_moe`, `cutlass_fused_moe`, etc.) → PR #3187
   is the right fix.

Until (1)-(3) are done, this is *necessary but unproven* mitigation.
PR #3187 still has value (it's the right fix for the FlashInfer autotune
case which is the more likely root cause per the incident shape — async
error after a tuning pass), but it should be treated as "applied because
the shape matches, not because we've confirmed the site."

## Diagnostic playbook (for next time)
- No host `dcgmi`/`py-spy`/`docker` on the node; containerd only. Runtimes:
  `crictl` (k8s.io namespace), `ctr`. GPU test images available:
  `registry.ocnr.org/infra/nvidia-devel`, the vllm image (has torch+CUDA).
- **`dump-jam-state.sh`** (pre-installed in the vllm image, writes to
  `/home/vllm/jam-<UTC>-<hostshort>/`): py-spy dump for every vllm/EngineCore/
  Worker PID + full `nvidia-smi` + per-GPU util + `/proc/<pid>/{cmdline,status,
  wchan,stack}` + last 50 dmesg Xid/NVRM lines. The single command to run from
  a jammed pod when you don't have time to remember the recipe — see
  `vllm/tools/dump_jam_state.sh` and the FlashInfer pin bump in
  `vllm/docker/Dockerfile`.
- **Xid**: `sudo dmesg -T | grep -iE "xid|nvrm"`. Xid PCI address is the
  container-index-independent ground truth for *which physical card*.
  Timestamps: dmesg is local (PDT), container logs are UTC (−7h).
- **GPU utilization pattern**: 3×100% + 1×0% ⇒ collective deadlock (one rank out).
- **Rank-vs-card** discriminator (staged, not yet run): set
  `CUDA_VISIBLE_DEVICES=1,0,2,3` so logical rank 1 runs on a *different* physical
  card; if the next Xid follows PCI `21:00` it's the card, if it follows rank 1
  it's software. (Hardware is already cleared, so this is now optional.)
- **PCIe link under load**: `nvidia-smi --query-gpu=index,pcie.link.gen.gpucurrent,pcie.link.gen.max --format=csv`.
