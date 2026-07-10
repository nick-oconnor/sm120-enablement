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
Not a ready-made one. Upstream #45388 (closed) / open PR #45406 target a *scheduler*
deadlock under KV-cache **pressure** (multi-request queue-head starvation) — a
**different** failure than our single-request, long-context, worker-collective
desync. A real fix would make the offloading connector's KV-load completion a
**collective** decision (all ranks barrier/agree on ready blocks before entering
the forward's collectives) instead of each rank proceeding on local state — a
non-trivial upstream change. Keep offloading off until such a fix exists and is
proven on SM120.

## Caveat / relationship to the Xid-69 incident
Both this hang and the later Xid-69 crash localized to **rank 1 / GPU 1**. We
initially wondered whether a marginal PCIE4 link (offloading = max PCIe DMA) could
be the shared cause, but GPU 1's link/hardware was subsequently cleared (DCGM
`-r 3`, PCIe Gen5 x16 under load, ASPM disabled). So this hang is best explained
by the offloading software desync, and the Xid-69 crash is tracked separately.
