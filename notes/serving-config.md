# Serving config & fixes — MiniMax-M3-NVFP4 on SM120

## Validated launch config

Runs the full 1M context across 4× RTX PRO 6000 Blackwell (SM120, PCIe-only),
fp8 KV cache, expert parallelism, FlashInfer-CUTLASS NVFP4 MoE, Triton attention.
Validated end-to-end: boot → serve → chat → tool call (no `]<]minimax[>[` leak) →
reasoning.

```
vllm serve /models/nvidia/MiniMax-M3-NVFP4 \
  --served-model-name MiniMax-M3-NVFP4 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --trust-remote-code \
  --tokenizer-mode hf \
  --max-model-len auto \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.97 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --moe-backend flashinfer_cutlass \
  --block-size 128 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image":1,"video":0}'
```

Required env: `HF_HUB_OFFLINE=1`, `NCCL_P2P_LEVEL=NODE`, plus (to survive the
container PID limit during FlashInfer JIT + HF tokenization) `RAYON_NUM_THREADS=4`
and `MAX_JOBS=32`. Deployed via k8s-gitops `stage3/apps/vllm.yaml`.

At 0.97 util: full context auto-fits (`GPU KV cache size: 1,136,384 tokens`,
1.08× concurrency for a 1,048,576-token request). **Note:** ECC is now enabled
on all four GPUs (see Xid-69 incident) which trims ~6% VRAM — re-check the
auto-fit KV line; drop util to ~0.95 if boot gets tight.

> **Do NOT add `--kv-offloading-size` / `--kv-offloading-backend`** — see
> `incident-kv-offload-deadlock.md`.

## Why each non-obvious flag

- `--enable-expert-parallel` — 128 routed experts; EP works on 4 GPUs (DCP needs
  TP > num_kv_heads, which fails at 4, so EP is the route).
- `--kv-cache-dtype fp8` — fits the full 1M context in GPU KV.
- `--attention-backend TRITON_ATTN` — M3 sparse attention needs block size 128;
  FlashAttn's fp8 path is SM90-only, and FlashInfer caps block ≤64 on SM120.
  Triton reconciles both.
- `--moe-backend flashinfer_cutlass` — the only NVFP4 MoE backend on SM120 that
  applies M3's clamped SwiGLU-OAI (`swiglu_limit`). TRTLLM needs downloaded
  cubins (fails offline); it also maps to plain Swiglu and drops the clamp.
- `--block-size 128` — required by M3 sparse attention.

## The 5 original launch blockers (all fixed)

1. **OOM / experts loaded unquantized (bf16).** ModelOpt-mixed `_resolve_quant_algo`
   missed the experts because the nested `MiniMaxM3SparseForCausalLM` weight mapper
   strips the `language_model.` prefix from the shared quant config's keys while the
   live module prefix keeps it. Fix: add a `language_model.`-stripped candidate in
   `_quantized_layer_prefix_candidates` (`modelopt.py`).
2. **`FLASHINFER_TRTLLM` wants downloaded cubins** (unavailable offline) →
   `--moe-backend flashinfer_cutlass`.
3. **`nvrtc.h` missing** for FlashInfer runtime JIT → add
   `cuda-nvrtc-dev-${CUDA_VERSION_DASH}` to the Dockerfile final stage.
4. **KV block-size reconciliation fails** (`No common block size for 16`) →
   `--block-size 128`.
5. **Multi-GPU PCIe P2P**: `NCCL_P2P_LEVEL=NODE`; custom all-reduce auto-disables
   on >2 PCIe-only GPUs → PYNCCL. PR #47544 adds a functional-P2P verify guard.

## Build image

`registry.ocnr.org/infra/vllm:0.25.1-sm120-cu131` — CUDA 13.1.1, Ubuntu 24.04,
arch list `12.0`, CUTLASS pinned to HEAD `e8ecfad` (`GIT_SHALLOW FALSE`, commit
hash can't shallow-fetch), FlashInfer built from source (jit-cache wheel lacks
SM120 kernels), `cuda-nvrtc-dev` added.

## PR stack (fork `main`)

Rebased onto upstream `main` (`cc1d020d0`). Minimal validated set:

- **#45738** — NVFP4 clamped SwiGLU-OAI on FlashInfer-CUTLASS MoE
- **#47001** — MiniMax-M3 bugfix v2 (pure-Python tool parser primary, XML/param parsing)
- **#47544** — verify functional P2P before enabling MiniMax fused AR+RMSNorm
- **ocnr commit** — build config + the fixes below

Dropped as unnecessary (verified no dangling refs): #43814, #47577, #47392,
#47599, #47515.

## ocnr code fixes (in the squashed `ocnr:` commit)

- **Build**: CUDA 13.1.1 / Ubuntu 24.04 / sm120 arch / GitLab CI / `VERSION`.
- **CUTLASS** pinned to HEAD `e8ecfad` for SM120 FP4 kernels.
- **`cuda-nvrtc-dev`** for FlashInfer SM120 JIT.
- **`ckpt_names=("w1","w2","w3")`** passed to `FusedMoE` (nvidia/model.py) so
  ModelOpt NVFP4 resolves the fused experts (else unquantized → OOM).
- **NVFP4 quant resolution** for VL-nested MoE experts (`language_model.` prefix
  strip) in `modelopt.py`.
- **Tool parser** streaming fixes — see `tool-parser` section below.

## Tool parser (`minimax_m3`) — streaming namespace handling

The model wraps every structural tag with the `]<]minimax[>[` namespace marker.
Non-streaming uses a tolerant pure-Python parser (#47001). Streaming delegates to
the strict Rust parser, so three fixes compose as a pipeline:

1. **Normalize** (pr-47001, `_MISSING_NS_RE`): add the NS prefix to bare tags that
   arrive whole-in-delta.
2. **Boundary collapse** (ocnr, `minimax_m3_tool_parser.py`): the per-delta regex
   can't see an NS marker that arrived in an *earlier* delta (it's its own token),
   so it double-prefixes → `]<]minimax[>[]<]minimax[>[</tool_call>` which the Rust
   parser rejects. Drop the duplicate straddling the boundary.
3. **Rust `opt(NAMESPACE)` close** (ocnr, `rust/.../tool/minimax_m3.rs`): accept a
   close with 0 or 1 NS prefix, so a genuinely bare `</tool_call>` (model dropped
   the prefix, split across deltas so #1 can't repair it) still ends the block.
4. **Content strip** (ocnr): strip `]<]minimax[>[` from any streamed *content*
   delta (the Rust parser can surface it as content in malformed cases).

Contracts line up: #2 caps input at ≤1 NS; the Rust `opt` accepts 0 or 1. Verified
with cargo tests (`tolerates_bare_tool_call_close`) and a standalone regex repro of
the double-NS case.

## Reasoning (`minimax_m3`) — not a bug, just a field name

- vLLM emits reasoning under the message field **`reasoning`** (this version
  renamed `reasoning_content` → `reasoning`). A client reading `reasoning_content`
  sees nothing — that's the usual "reasoning missing" symptom.
- Chosen config: **adaptive** (no `--default-chat-template-kwargs`). The model
  decides per-turn whether to emit `<mm:think>…</mm:think>`; when it does, the
  parser splits `reasoning` + clean `content` and never leaks tags. When it
  doesn't, it writes an inline "**Reasoning:**" section into `content` (clean).
- `--default-chat-template-kwargs '{"thinking_mode":"enabled"}'` forces thinking
  every turn (incl. after each tool result) — declined for the per-tool-round-trip
  token cost. `thinking_mode` reaches the parser via `_effective_chat_template_kwargs`
  → `Parser.__init__`.
- opencode (bundled AI SDK) reads both `reasoning` and `reasoning_content`, so it
  displays reasoning fine.
