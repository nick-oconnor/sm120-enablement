# Incident: KV-offloading deadlock at long context — RESOLVED

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
Enabled by the k8s-gitops commit *"vllm: enable KV cache offloading to host RAM"*,
which added:
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

## Fix
Removed both flags. Commits:
- k8s-gitops `stage3/apps/vllm.yaml` — `d9736d40` (reverts the offloading change)
- sm120-enablement README — `4b89c07`

## Why disabling is the right call (not just a workaround)
- The **full 1M context already fits in GPU KV** (1,136,384 tokens at 0.97 util),
  so offloading buys nothing for single long conversations (the real workload).
- It only helped *concurrent* long sequences whose combined KV exceeds aggregate
  GPU KV — not worth a class of hangs that hard-kill the engine.

## Is there a code fix?
Not in upstream vLLM as of the current fork pin. Upstream #45388 (closed) / open
PR #45406 target a *scheduler* deadlock under KV-cache **pressure** (multi-request
queue-head starvation) — a **different** failure than our single-request,
long-context, worker-collective desync. vLLM PR #44560 (`3b3d5287f`, "Resolve
multiple async kv load deadlock") fixed the *admission-control wedge* (two async
loads together consuming all blocks; neither can complete) — also a different
failure. As of 2026-07, no upstream vLLM commit addresses our TP/EP-rank desync.

A real fix has to make KV-load completion a **collective** decision: all ranks
agree they have finished loading before any rank enters the forward's TP/EP
collectives. Two composable shapes, both proposed in incident-doc history:

1. **Stream-side ordering** — record the load's `end_event` on the
   compute/NCCL stream so the forward's collectives are automatically ordered
   after the load by CUDA's stream graph. Fixes in-rank ordering but does not
   cross-rank desync.
2. **Collective barrier** — `start_load_kv` blocks on each rank's local load
   events, then a host-side `dist.barrier()` (or `all_reduce(empty)`) gates the
   forward. Adds ~10–20 µs/step but eliminates the desync.

Either alone is insufficient; **(1)+(2) together** is the actual fix. This is the
same shape as FlashInfer PR #3187's `set_autotune_process_group` for the Xid-69
autotune crash (same bug family — per-rank async decisions before collectives).

## Patch (implemented locally, not yet deployed)

`VLLM_KV_OFFLOAD_COLLECTIVE_BARRIER=1` opt-in env var, off by default, lives in
`vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` (barrier
side) and `vllm/v1/kv offload/cpu/gpu_worker.py` (stream side). The barrier waits
on this rank's load events via `OffloadingConnectorWorker.wait_for_pending_loads()`
(added on `_load_jobs`), then `dist.barrier()` on the default group before
returning from `start_load_kv`. The stream-side change makes the CPU→GPU
load's `end_event` visible to the compute stream (`current_stream().wait_event`)
so subsequent ops — including collectives — are ordered after the load on this
rank.

Validation path:
- The offloading connector is disabled in production (1M context fits in GPU
  KV at 0.97 util, so offloading is unused). The patch is unreachable without
  re-enabling `--kv-offloading-size`.
- To test: spin up a vllm pod with `--kv-offloading-size 1 --kv-offloading-backend
  native --enable-async-output-processing`, drive a single >100K-token request,
  and confirm no hang at 100–200K tokens (the failure window).
- No commit needed until validation passes on SM120 with a long-context request.

## Why we kept offloading off in production (status quo)
- The **full 1M context already fits in GPU KV** (1,136,384 tokens at 0.97 util),
  so offloading buys nothing for single long conversations (the real workload).
- It only helped *concurrent* long sequences whose combined KV exceeds aggregate
  GPU KV — not worth a class of hangs that hard-kill the engine.
- The local patch turns the failure mode from "guaranteed hang" to "untested
  throughput regression" — which is a much better trade, but still needs
  validation before flipping the production flag.

## Caveat / relationship to the Xid-69 incident
Both this hang and the later Xid-69 crash localized to **rank 1 / GPU 1**. We
initially wondered whether a marginal PCIE4 link (offloading = max PCIe DMA) could
be the shared cause, but GPU 1's link/hardware was subsequently cleared (DCGM
`-r 3`, PCIe Gen5 x16 under load, ASPM disabled). So this hang is best explained
by the offloading software desync, and the Xid-69 crash is tracked separately —
though both share the same underlying family: **per-rank async decisions before
TP/EP collectives**.
