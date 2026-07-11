# Incident: KV-offloading deadlock at long context — RESOLVED, fix deployed

## Status

| Date | Event |
| --- | --- |
| 2026-07-03 | Offload flags deployed (`33f5a354`); deadlock fires within hours at ~166K tokens |
| 2026-07-03 | Offload flags removed (`d9736d40`); 1M-context fits in GPU KV, so offload wasn't needed anyway |
| 2026-07-10 | Local fix landed in `infra/vllm` fork (`f256368d`-era commits, see "Patch" below): stream-side ordering + TP barrier in `OffloadingConnector.start_load_kv`, gated behind `VLLM_KV_OFFLOAD_COLLECTIVE_BARRIER=1` |
| 2026-07-11 | Offload flags re-enabled + barrier env var set in production (`k8s-gitops` `56d81ffc`); awaiting long-context load to validate |

## Symptom
Random inference **hangs** that did not self-recover and required a manual pod
restart. The engine would eventually die with:

```
EngineCore encountered a fatal error.
TimeoutError: RPC call to sample_tokens timed out.
shm_broadcast: No available shared memory broadcast block found in 60 seconds
  (repeated every 60s for ~5 min before the RPC timeout)
→ EngineDeadError
```

## Trigger
Originally enabled by the k8s-gitops commit *"vllm: enable KV cache offloading to host RAM"* (`33f5a354`), which added:
```
--kv-offloading-size 100
--kv-offloading-backend native
```
Occurred on **long-context** requests (observed at ~166K computed tokens),
`num_running_reqs=1`, KV usage only ~15% (not a memory-pressure issue).

## Root cause
Host-RAM KV offloading + TP/EP + async scheduling. When offloaded KV blocks are
loaded back from host RAM, the async loads resolve **inconsistently across TP/EP
ranks**; the ranks desync and a collective deadlocks.

**Evidence (py-spy during the hang):**
- All four workers' MainThread wedged at
  `gpu_input_batch.py:update_async_output_token_ids → async_copy_ready_event.synchronize()`
  — the async-scheduling sample path, which the code itself annotates with a
  *"tokens discarded after a kv-load failure"* branch (offloading is baked into
  that exact site).
- `nvidia-smi`: GPUs 0/2/3 at **100%** (spinning in an NCCL collective), GPU 1 at
  **0%** (diverged — never entered it). Classic one-rank-desync deadlock.

## Fix (deployed 2026-07-11)

The offload flags are back on, paired with `VLLM_KV_OFFLOAD_COLLECTIVE_BARRIER=1`
in `stage3/apps/vllm.yaml` (`k8s-gitops` `56d81ffc`). The barrier is the
opt-in part of the patch; without it, the flags alone re-trigger the deadlock.
Image bump: `4dfecc7...` → `9b61dbd...`.

Validation status: **pending**. The re-enabled offload buffer hasn't been
exercised yet (live traffic has been ≤131K tokens; offload triggers only
above the GPU KV budget, currently 1.14M tokens at 0.97 util). Need to drive
a single >200K-token request through the live pod and confirm no hang. Use
`dump-jam-state.sh` if it jams — that script is pre-installed in the vllm
image (`/usr/local/bin/dump-jam-state.sh`, writes to `/home/vllm/jam-*/`).

## Why disabling was the right call at the time
- The **full 1M context already fits in GPU KV** (1,136,384 tokens at 0.97 util),
  so offloading buys nothing for single long conversations (the real workload).
- It only helped *concurrent* long sequences whose combined KV exceeds aggregate
  GPU KV — not worth a class of hangs that hard-kill the engine.

## Is there a code fix?
Yes — landed in the `infra/vllm` fork on 2026-07-10. Two composable shapes,
both gated behind the same `VLLM_KV_OFFLOAD_COLLECTIVE_BARRIER=1` env var:

1. **Stream-side ordering** — in `vllm/v1/kv_offload/cpu/gpu_worker.py`, after
   each CPU→GPU load's `end_event` on the dedicated transfer stream,
   `compute_stream.wait_event(end_event)` so subsequent compute ops
   (attention reads, collectives) are auto-ordered after the load on this
   rank. Fixes in-rank ordering.
2. **Collective barrier** — `OffloadingConnector.start_load_kv` calls
   `wait_for_pending_loads()` (new on `OffloadingConnectorWorker`, blocks on
   this rank's load events) then `tp_group.barrier()` -- all ranks enter
   the forward together. Fixes cross-rank desync.

Both are needed: (1) alone doesn't fix the desync; (2) alone would need
the host-side load sync anyway. Cost is one ~10-20us barrier per step
when sync is enabled; zero when disabled.

This is the same shape as FlashInfer PR #3187's `set_autotune_process_group`
for the Xid-69 autotune crash (same bug family — per-rank async decisions
before collectives).

## Upstream status
No upstream vLLM commit addresses our TP/EP-rank desync. Upstream PR #45388
(closed) / open PR #45406 target a *scheduler* deadlock under KV-cache
**pressure** (multi-request queue-head starvation) — a different failure.
vLLM PR #44560 (`3b3d5287f`, "Resolve multiple async kv load deadlock")
fixed the *admission-control wedge* (two async loads together consuming all
blocks; neither can complete) — also a different failure.

## Caveat / relationship to the Xid-69 incident
Both this hang and the later Xid-69 crash localized to **rank 1 / GPU 1**. We
initially wondered whether a marginal PCIE4 link (offloading = max PCIe DMA) could
be the shared cause, but GPU 1's link/hardware was subsequently cleared (DCGM
`-r 3`, PCIe Gen5 x16 under load, ASPM disabled). So this hang is best explained
by the offloading software desync, and the Xid-69 crash is tracked separately —
though both share the same underlying family: **per-rank async decisions before
TP/EP collectives**.
