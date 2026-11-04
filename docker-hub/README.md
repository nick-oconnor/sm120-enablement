# vLLM for SM120

[vLLM](https://github.com/vllm-project/vllm) inference image tuned for
**NVIDIA Blackwell consumer GPUs (SM 12.0, RTX PRO 6000 Blackwell)**, serving
**GLM-5.3-Flash** (sparse-MLA MoE, 1M context, native vision) on the
**single-outlet inference** rig documented in
[sm120-enablement](https://github.com/nick-oconnor/sm120-enablement).

## Do you need this image?

**Probably not, as of vLLM 0.33.** Upstream `vllm/vllm-openai:v0.33.0` serves
GLM-5.3-Flash on SM120 correctly on its own - FlashInfer now ships a native
no-rope MLA kernel for this shape, upstream's default build already includes
arch 12.0, and it has a native PCIe IPC all-reduce for TP>2 without NVLink.
The reference rig switched to the stock upstream image on 2026-11-04.

This image is still a drop-in replacement, and it adds:

- **py-spy + `dump-jam-state.sh`** pre-installed for incident diagnostics
  (dumps py-spy traces, `nvidia-smi`, and dmesg Xid lines) - the main reason
  to prefer it
- a **single-arch build** (`TORCH_CUDA_ARCH_LIST="12.0"`), which cuts build
  time substantially; the resulting image is about the same size as upstream
  (35.2 GB vs 35.4 GB) but does less runtime JIT, so restarts are faster
  (~46 s vs ~164 s warm on the reference rig)
- **reproducible digests** - `SOURCE_DATE_EPOCH` is pinned to the last commit
  touching the build inputs, so an unchanged tree rebuilds to the same digest

Earlier tags (0.29-0.32) carried fixes that were genuinely required before
upstream absorbed them; see the
[serving notes](https://github.com/nick-oconnor/sm120-enablement/blob/main/notes/serving-config.md)
for the per-release history.

## Image Contents

- **vLLM** built from the `0.33` branch of
  [github.com/nick-oconnor/vllm](https://github.com/nick-oconnor/vllm), tagged
  `0.33.0-sm120-cu130` (upstream `v0.33.0` release tag `3a4d82f569`).
  GLM-5.3-Flash model support is native upstream since vllm-project #53906.
  The branch still carries an SM120 NoPE sparse-MLA port, a kpool
  index-width pin, an all-ranks warmup tune and a scheduler aligned-split
  fix, but **all of them are redundant on v0.33.0** - they are retained only
  so the branch history stays intact, and are verified to be no-ops against
  stock upstream.
- KV offloading is upstream native (`CPUOffloadingSpec`, scoped to
  prefix-cacheable groups), enabled in the quick start below.
- **CUDA 13.0** runtime + toolchain: DeepGEMM is compiled in-image and
  TileLang builds from sdist; kernels the FlashInfer / CUTLASS wheels don't
  cover JIT at server startup, so `cuda-nvrtc-dev` ships in the runtime layer,
  not just the build layer
- **FlashInfer 0.7.2.post1** via the `flashinfer-python` +
  `flashinfer-cubin` wheels from the flashinfer.ai index (upstream's own
  pin); autotune tactic sync and per-rank cache persistence across TP ranks
  are both upstream-native since 0.32 (vllm-project #57635)
- **CUTLASS** via the `nvidia-cutlass-dsl==4.7.1` PyPI wheel (SM120 GEMM
  kernels; upstream's requirement since the DSL 4.7 bump)
- **b12x** via the `b12x==1.3.0` PyPI wheel: SM12x CuTe DSL kernels. Upstream
  imports b12x (modelopt_block_quant, kernel_warmup, fused_moe) but never
  installs it. Installed `--no-deps` so it cannot downgrade cutlass-dsl below
  the 4.7.1 upstream requires
- `TORCH_CUDA_ARCH_LIST="12.0"`: only SM120 is built. Upstream's default
  list already includes 12.0, so this is a build-time and startup-JIT
  optimisation, not a capability difference

## Quick Start

- `--shm-size 120g`: NCCL bootstrap + engine IPC + the 100 GiB native
  KV-offload mmap
- `HF_HUB_OFFLINE=1`: weights come from the local `/models` mount, not the
  Hub
- `NCCL_P2P_LEVEL=NODE`: NCCL can't auto-detect P2P from inside the
  container; declare it manually
- `RAYON_NUM_THREADS=4`, `OMP_NUM_THREADS=4`: cap the Rayon thread pool
  (used by tokenizers/parquet)
- `MAX_JOBS=32`: cap JIT parallelism so container PIDs stay sane
- `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`: upstream's PCIe IPC all-reduce
  for decode-size collectives; the only AR path supporting TP>2 without
  NVLink, and ~27% faster than the b12x oneshot it replaced on 0.33
- `--tensor-parallel-size 4`: 4-way TP across the four GPUs
- `--max-model-len auto`: auto-fit; holds the full 1M (8.75 GiB KV /
  1,068,443 tokens, 1.02x)
- `--gpu-memory-utilization 0.644`: leaves ~25 GiB/GPU free; this rig shares
  two cards with an image-generation workload. Measured floor for the full 1M
  is 0.643; 0.642 lands 4,864 tokens short
- `--compilation-config '{"cudagraph_mode": "FULL"}'`: **required.** Under
  the upstream default `FULL_AND_PIECEWISE` the model emits token salad:
  `splitting_ops` is empty for this model, so "piecewise" captures the whole
  model for mixed prefill+decode batches, including the sparse-MLA prefill
  path, which is not CUDA-graph safe. `FULL` captures only uniform decode
  batches, at no measurable throughput cost
- `--moe-backend flashinfer_cutlass`: NVFP4 MoE; pinned, `auto` resolves
  here today but is free to drift
- `--kv-offloading-size 100 --kv-offloading-backend native`: native CPU KV
  offload: 100 GiB pool mmap'd in `/dev/shm`
- `--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'`:
  MTP speculative decoding, 3 draft tokens
- `--default-chat-template-kwargs '{"thinking": true}'`: force thinking mode
  on every turn

```bash
docker run --rm --gpus all --shm-size 120g \
  -v <host-models-path>:/models:ro \
  -v <host-cache-path>:/home/vllm \
  -p 8000:8000 \
  -e HF_HUB_OFFLINE=1 \
  -e NCCL_P2P_LEVEL=NODE \
  -e RAYON_NUM_THREADS=4 \
  -e OMP_NUM_THREADS=4 \
  -e MAX_JOBS=32 \
  -e VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 \
  ngpitt/vllm:0.33.0-sm120-cu130 \
    /models/nvidia/GLM-5.3-Flash-NVFP4 \
      --served-model-name GLM-5.3-Flash \
      --tensor-parallel-size 4 \
      --enable-expert-parallel \
      --trust-remote-code \
      --max-model-len auto \
      --max-num-seqs 4 \
      --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.644 \
      --compilation-config '{"cudagraph_mode": "FULL"}' \
      --moe-backend flashinfer_cutlass \
      --kv-cache-dtype fp8 \
      --enable-prefix-caching \
      --enable-prompt-tokens-details \
      --kv-offloading-size 100 \
      --kv-offloading-backend native \
      --enable-chunked-prefill \
      --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --enable-auto-tool-choice \
      --limit-mm-per-prompt '{"image": 999, "video": 0}' \
      --default-chat-template-kwargs '{"thinking": true}'
```
