# Incident: silent KV-cache corruption under offloading — RESOLVED, offload removed

## Status

- **2026-09-10 ~17:04 UTC** — `e593bc5a` rolled (0.29 upstream-main rebuild;
  rewritten `OffloadingConnector` scheduler, offload fix series re-ported onto
  the new upstream layout)
- **2026-09-10 22:56–23:57 UTC** — progressive silent corruption in production
  agent traffic; three distinct stages (below)
- **2026-09-10 23:58 UTC** — pod restart; corruption gone (state-bound, dies
  with the process)
- **2026-09-11** — offload patches dropped from the `0.29` branch, offload
  flags removed in production (`k8s-gitops` `d793fefe` + `6f0ff09a`); rebuilt
  image `94a85a96` (source `bea5f74795`) deployed

## Symptom

Progressive over ~1 h of agentic (long-context, tool-heavy) load, bound to the
engine process lifetime:

1. Repeated-token runs and degenerate text inside an ongoing session (a
   workspace path degrading to `/home/vu/vu/vu/…`, runaway
   `abs_path(abs_path(…` repetition, mangled tool calls)
2. Wrong-context recall: fluent replies referencing conversations, files, and
   workspaces that never existed (the model's memory of its own context was
   wrong, not its language)
3. Reasoning-only replies with no content, then immediate-EOS empty responses
   (HTTP 200, `stopReason: stop`, 0 output tokens)

vLLM reported every finish as a clean `stop` (573 `stop` / 4 `length` /
0 `error` over the pod lifetime) with zero warnings; KV usage peaked at 28%.
Restart cleared it; the same configuration on the previous build (`ff25e8dc`,
8 days up) never showed it.

## Trigger

Multi-group hybrid KV (sparse-MLA + KDA linear-attention state + kpool tail)
with APC and native CPU offloading under sustained eviction pressure. During
the degraded window the offloader served 1,094,400 external (CPU) prefix-hit
tokens; corruption scaled with cache-hit volume, consistent with upstream
#53912.

## Root cause (most likely)

Per-group external-hit allocation in the rewritten `OffloadingConnector`
scheduler: on an external (CPU) hit the recurrent (KDA) state group resumed
misaligned with the sparse-MLA group, and the poisoned recurrent state (the
model's compressed conversation memory) was served back on subsequent hits.
Fingerprint matches the open upstream class "multi-group hybrid KV cache +
connector external hit + eviction pressure" (#50454). Not proven to the line;
removing offloading eliminated the failure path and is the deployed mitigation.

Amplifier (not cause): litellm's redis response cache (TTL 1h) cached one
empty response; every client retry and the next session received the identical
response id until TTL expiry, making the outage look total.

## Evidence

- Session transcripts (dsh session store): the three stages with timestamps;
  the model itself reported "corrupted, degenerate text (repeated tokens)"
- VictoriaMetrics: `vllm:request_success_total{finished_reason="stop"}` 573,
  `error` 0; `vllm:kv_cache_usage_perc` max 0.28;
  `increase(vllm:external_prefix_cache_hits_total[7h])` = 1,094,400
- Pod log clean through the collapse (no mid-inference JIT compiles, no OOM,
  no aborts)
- Known-good vs regressed builds share args, backend
  (`FLASHINFER_MLA_SPARSE_SM120`), and KV geometry (1,095,931 tokens); the
  delta was confined to the upstream re-merge (model impl #53906, KV-cache
  layout refactor, scheduler rewrite) plus the re-ported ocnr patches

## References

- https://github.com/vllm-project/vllm/issues/50454 (multi-group + connector
  external hit + eviction pressure; open)
- https://github.com/vllm-project/vllm/issues/53912 (corruption scales with
  prefix-cache hit rate; reopened)
- https://github.com/vllm-project/vllm/issues/52735 (OffloadingConnector
  store/serve asymmetry; closed)
