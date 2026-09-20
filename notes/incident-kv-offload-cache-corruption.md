# Incident: silent KV-cache corruption under offloading — root cause found (offloading was an amplifier, not the cause)

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
- **2026-09-17** — corruption **recurred with offload disabled** (97.2–97.3%
  local prefix-cache hit rate). This falsified the offload-connector
  root-cause hypothesis in the sections below.
- **2026-09-17** — **actual root cause identified**: upstream #55600 —
  `MambaHybridModelState.add_request` seeds the KDA recurrent-state slot with
  `(num_computed_tokens - 1) // cache_config.block_size`, but
  `EngineCore._initialize_kv_caches` has by then lowered
  `cache_config.block_size` to the smallest prefix-cacheable group
  (GLM-5.3-Flash drafter/SWA group, 64) while mamba blocks are in KDA units
  (3584 at TP4) — so **every prefix-cache hit, local or external/offload,
  restores the wrong recurrent state**. Silent at small hit sizes; OOB
  (`Xid 31`) at ≥8 mamba blocks in single-GPU configs. Fix = upstream PR
  #55601 (open, 1-line: divide by `cache_config.mamba_block_size`), verified
  upstream on GLM-5.3-Flash (warm hits == cold answers, multi-needle 5/5,
  3 h soak). Offloading amplified exposure (external hits are hits too —
  1,094,400 external-hit tokens in the degraded window), which is why
  removing offload only reduced the rate instead of eliminating it.
- **2026-09-18** — fix carried on the `0.30` re-cut (upstream main past
  v0.30.0rc1; image
  `registry.ocnr.org/infra/vllm:0.30.0-sm120-cu130@sha256:11190e94…`,
  gitops `027a6c79`); offload live on upstream's native `CPUOffloadingSpec`
  (scopes offload configs to prefix-cacheable groups natively via
  `get_offloading_group_ids` — no #54743 carry needed). Production boot
  healthy: FLASHINFER_MLA_SPARSE_SM120 + fp8_ds_mla selected, no asserts,
  prefix hit 82% within minutes, one-time allocator flush-retry + one-time
  `_count_expert_num_tokens` JIT (both known-benign signatures). Stores
  flowing on eviction (up to 2.66 GB/10s); external-hit soak still
  pending — watch `vllm:external_prefix_cache_hits_total` and finish
  reasons over the first hours of agentic load.
- **2026-09-20** — a **second, independent** corruption source found on the
  same model and fixed: upstream **#57477**, the NVIDIA
  `_kpool_tail_seed_kernel`. The tail cache tensor is aliased onto the
  indexer tensor by `get_kv_cache_config_from_groups`, so it carries the
  indexer's block stride — probed live on the 09-18 image as
  `stride=(38016, 512, 128, 1)` against a dense `2 * kpool * head_dim =
  1024`. The kernel addressed blocks densely, so on **every prefill**, for
  every tail block `blk > 0`, the 128-element K and score writes landed
  inside indexer block `blk // 37`, overwriting pooled indexer keys with raw
  unpooled values — and the request's own tail block was never seeded, so
  decode compressed the boundary pool from whatever the previous tenant
  left. Silent (no OOB, no assert, no log), persistent (the damaged indexer
  blocks are prefix-cached and reused) and cumulative — exactly the
  "degenerates after a while, restart clears it" profile, and it accounts
  for the residue #55601 alone did not explain. Fixed in the 2026-09-20
  `0.30` re-cut; regression test
  `tests/kernels/test_kpool_decode_update_batched.py::test_prefill_seed_honors_padded_tail_block_stride`
  FAILs on the 09-18 image and PASSes on the re-cut. Post-fix soak: six
  ~300K-token sessions cycled through a full eviction of the 1.06M-token GPU
  pool, 12/12 needles correct, 3,594,240 external-hit tokens, 0 preemptions,
  0 errors. This closes the external-hit soak listed as pending above.

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

## Root cause (two, confirmed 2026-09-17 and 2026-09-20)

There were **two** independent defects producing the same fingerprint. The
first is below; the second — the kpool tail seed kernel writing at a dense
stride into a padded-stride view (upstream #57477) — is in the 2026-09-20
status entry. #55601 fixed the recurrent-state half; #57477 fixed the
indexer half. Neither alone was sufficient.

### 1. Hybrid mamba state-index units (#55600)

Upstream **#55600** (fix: PR #55601, cherry-picked as `be9492b657`): the KDA
recurrent-state slot seed on a prefix-cache hit is computed in the wrong
units. The bug fires on **any** prefix-cache hit — local APC or external
(offload) — which unifies both episodes: the 2026-09-10 offload-enabled
corruption (external hits) and the 2026-09-17 no-offload recurrence (local
hits at 97% hit rate). It also explains why the pre-0.29 fork build was
clean: the `gpu/model_states/mamba_hybrid.py` align-mode seeding arrived with
the 0.29 re-merge. The original "per-group external-hit allocation
misalignment" hypothesis below was unproven and is superseded; #50454
remains open upstream as a separate crash-class risk for offload, but is no
longer blamed for this corruption.

Superseded hypothesis: per-group external-hit allocation in the rewritten
`OffloadingConnector` scheduler resumed the recurrent (KDA) state group
misaligned with the sparse-MLA group.

Amplifier (not cause): litellm's redis response cache (TTL 1h) cached one
empty response; every client retry and the next session received the identical
response id until TTL expiry, making the outage look total. Flush redis on
any restart that follows a corruption window.

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

- https://github.com/vllm-project/vllm/issues/55600 (root cause: hybrid mamba
  state index seeded with the wrong block size on prefix-cache hits)
- https://github.com/vllm-project/vllm/pull/55601 (the 1-line fix; open —
  cherry-picked onto the ocnr `0.29` branch as `be9492b657`)
- https://github.com/vllm-project/vllm/pull/57477 (merged: address kpool tail
  blocks by the padded indexer stride in the NVIDIA prefill seed kernel —
  the second corruption source, found 2026-09-20)
- https://github.com/vllm-project/vllm/issues/50454 (multi-group + connector
  external hit + eviction pressure; open — separate crash-class risk)
- https://github.com/vllm-project/vllm/issues/56868 and
  https://github.com/vllm-project/vllm/issues/56605 (community reports of
  the same "long session degenerates into repeated-token word salad"
  profile on GLM-5.3-Flash; both predate #57477 and neither has been
  re-tested against it upstream)
- https://github.com/vllm-project/vllm/issues/53912 (corruption scales with
  prefix-cache hit rate; reopened — MTP-gated variant, we run no MTP)
- https://github.com/vllm-project/vllm/issues/52735 (OffloadingConnector
  store/serve asymmetry; closed)
- https://github.com/gitcommit90/glm-5.3-one-spark/issues/3 (independent
  reproduction + root-cause writeup on GLM-5.3-Flash, DGX Spark)
