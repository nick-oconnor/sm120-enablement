# Recut build: agent loop degeneration (2026-09-30 → 2026-10-01)

## What happened

The 2026-09-30 re-cut of the `0.30` branch onto upstream main `2eaa3bc5ac`
(image `0.30.0-sm120-cu130@sha256:7a7d3d10`, pipeline 55808) degraded the
served agent: the dsh agents' inference rides this vLLM (`litellm` →
`api_base: http://vllm.apps.svc.cluster.local/v1`), and the operator watched
the agent stall in tool-call loops twice during the night it served, plus
further degraded behavior the next morning. The build booted green and stayed
green — backend selection, auto-fit and the memory profile all nominal, zero
ERROR lines — while generation quality degenerated. Reverted the next
morning; the previous tree rebuilt as `3399d4fc` (content-identical to the
verified `87feab60`) and the symptom never returned.

Degenerate repetition under a long, prefix-cache-heavy session that only a
rollback clears is the classic KV-poisoning signature this box has seen twice
before (#57477 kpool tail seed, #55600/#55601 state-seed units).

## Observed artifacts (from the serving agent's own transcript)

| time (UTC) | artifact |
|---|---|
| 10-01 ≈03:45–04:15 | tool-call stall #1: the agent repeated one identical tool call until the harness flagged it ("repeating the exact same tool call with identical arguments") |
| 10-01 ≈04:20–04:45 | tool-call stall #2, same signature |
| 10-01 04:47:32 | vllm-0 rolled to the recut build |
| 10-01 ≈11:3x | corrupted token generation mid-window: a generated query carried spurious CJK characters inside an identifier, followed by a run of repeated failing/empty tool calls |
| 10-01 12:05:25 | vllm-0 recreated (still the recut build) |
| 10-01 ≈12:13 | operator reverted the branches to `b68c19aef6` |
| 10-01 ≈12:4x | digest re-pinned to `87feab60` |
| 10-01 15:05 | rolled to the `3399d4fc` rebuild; no stalls or corrupted output since (several turns of normal operation) |

The first two stalls end minutes before the recut pod's first boot — they sit
on the transition into the recut window; the corrupted-token episode is
unambiguously mid-window. All of them predate the revert; none recurred
after.

## Prime suspect: the #54076 scheduler carry swap

Against the verified 09-27 tree, the recut changed exactly one thing on the
mamba prefix-cache path: the ocnr one-line aligned-split fix (chunk stops on
the scheduler-resolved LCM grid, `self.block_size`) was replaced with the
full upstream carry of **#54076** — chunk stops derived from the mamba
group's own `MambaSpec.block_size`, plus *stop at every crossed state
boundary*.

The worker-side resume seeding (carried **#55601**, byte-identical to the PR
in both trees) divides by `cache_config.mamba_block_size`. The #55601 review
thread itself warns these two derivations must be tied together: if the grid
the scheduler publishes states and hashes on diverges from the grid the
worker seeds resumed requests by, every prefix-cache hit resumes from a
mis-indexed state row — silent, progressive, exactly the #55600 "units
disease" class, and exactly the observed symptom. This box runs an ~87%
prefix-cache hit rate, so hits dominate every agent session.

Unproven: the build was reverted within hours, no probe was captured, and
#54076's boot-time single-mamba-grid assert passed (so any divergence would
be `cache_config.mamba_block_size` vs the group spec, not group-vs-group).
The recut's other changes (#55222 swapped to upstream's merged version,
#57635 reworked onto #57197's per-stage autotune groups) do not touch the
state/resume path.

## Disposition

- **Resolved by revert** — branches back to `b68c19aef6` (the 09-27 tree);
  CI rebuilt it as `3399d4fc` (content-identical to the verified
  `87feab60`); deployed 2026-10-01 15:05 UTC and clean since.
- The recut commits survive only in local reflog (branch deleted
  2026-10-01). If the rebase is ever reattempted: #54076 is the prime
  suspect — verify the `cache_config.mamba_block_size` ↔
  `MambaSpec.block_size` tie on this layout before booting, and consider
  keeping the verified LCM one-liner for the scheduler side instead.
- If it reproduces on any build: capture before touching anything
  (needle-style prefix-cache A/B across a restart, `dump-jam-state.sh`),
  per the playbook in `SM120-GLM53-FLASH.md`.
