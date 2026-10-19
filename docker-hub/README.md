# vLLM for SM120

Standalone [vLLM](https://github.com/nick-oconnor/vllm) inference image for the
**single-outlet inference** setup documented in
[sm120-enablement](https://github.com/nick-oconnor/sm120-enablement).

Targets **NVIDIA Blackwell consumer GPUs (SM 12.0, RTX PRO 6000 Blackwell)**
for serving **GLM-5.3-Flash** (sparse-MLA MoE, 1M context, native vision).

## Image Contents

- **vLLM** built from the `0.32` branch of
  [github.com/nick-oconnor/vllm](https://github.com/nick-oconnor/vllm), tagged
  `0.32.0-sm120-cu130` (upstream `v0.32.0` release tag `5df63a7203`).
  GLM-5.3-Flash model support is native upstream since vllm-project #53906
  — no fork overlay is needed. The image carries, vs upstream:

   - the SM120 NoPE sparse-MLA port (fp8 + FlashInfer zero-pad, backend
     priority, indexer buffer pin)
   - the b12x PCIe one-shot all-reduce backend (`VLLM_ENABLE_PCIE_ALLREDUCE`)
   - the SM120 image build (Blackwell consumer)
   - SM120 sparse-MLA warmup tunes all ranks; leader-only tuning left
     asymmetric tuner state that deadlocked the synchronized autotune
   - mamba aligned-split uses the scheduler-resolved block size;
     `cache_config.block_size` gets reassigned to the smallest
     prefix-cacheable group

  On 0.32 the overlay shrank: #55222, #55601, #56531 and #57635 all landed
  upstream and are no longer carried (#57635 upstream is a superset that also
  fingerprints the per-rank cache files), and #59504 (async KV-load zeroing)
  is in the base. Upstream also ships its own `flashinfer_mla_sparse_sm120.py`
  now, so the NoPE port is a delta on top of it. Upstream's #57477 (kpool
  tail-seed stride fix for the silent KV-cache poisoning) is in the base. KV offloading is upstream native
  (`CPUOffloadingSpec`, scoped to prefix-cacheable groups), enabled in the
  quick start below.

- **CUDA 13.0** runtime + toolchain — DeepGEMM is compiled in-image and
  TileLang builds from sdist; kernels the FlashInfer / CUTLASS wheels don't
  cover JIT at server startup, so `cuda-nvrtc-dev` ships in the runtime layer,
  not just the build layer
- **FlashInfer 0.7.0.post1** via the `flashinfer-python` +
  `flashinfer-cubin` wheels from the flashinfer.ai index (upstream's own
  pin); autotune tactic sync across TP ranks is upstream-native, with
  per-rank cache persistence carried on top
- **CUTLASS** via the `nvidia-cutlass-dsl==4.7.1` PyPI wheel (SM120 GEMM
  kernels; upstream's requirement since the DSL 4.7 bump)
- **b12x** via the `b12x==1.3.0` PyPI wheel — CuTe DSL PCIe one-shot all-reduce
  (`b12x.comm.pcie`); enabled per-deployment with `VLLM_ENABLE_PCIE_ALLREDUCE=1`
  (replaces NCCL-SHM for decode-size collectives at TP>2 on PCIe-only boxes)
- **py-spy + `dump-jam-state.sh`** pre-installed for incident diagnostics
  (dumps py-spy traces, `nvidia-smi`, and dmesg Xid lines)
- `TORCH_CUDA_ARCH_LIST="12.0"` — only SM120 gets built, so the image is
  smaller than the upstream `vllm/vllm` images that target every arch

## Quick Start

- `--shm-size 120g` — NCCL bootstrap + engine IPC + the 100 GiB native
  KV-offload mmap
- `HF_HUB_OFFLINE=1` — weights come from the local `/models` mount, not the
  Hub
- `NCCL_P2P_LEVEL=NODE` — NCCL can't auto-detect P2P from inside the
  container; declare it manually
- `RAYON_NUM_THREADS=4`, `OMP_NUM_THREADS=4` — cap the Rayon thread pool
  (used by tokenizers/parquet)
- `MAX_JOBS=32` — cap JIT parallelism so container PIDs stay sane
- `VLLM_ENABLE_PCIE_ALLREDUCE=1` — b12x PCIe one-shot all-reduce replaces
  NCCL-SHM for decode-size collectives
- `--tensor-parallel-size 4` — 4-way TP across the four GPUs
- `--max-model-len auto` — auto-fit; holds the full 1M (8.75 GiB KV /
  1,068,443 tokens, 1.02x)
- `--gpu-memory-utilization 0.644` — leaves ~25 GiB/GPU free; this rig shares
  two cards with an image-generation workload. Measured floor for the full 1M
  is 0.643; 0.642 lands 4,864 tokens short
- `--compilation-config '{"cudagraph_mode": "FULL"}'` — **required.** Under
  the upstream default `FULL_AND_PIECEWISE` the model emits token salad:
  `splitting_ops` is empty for this model, so "piecewise" captures the whole
  model for mixed prefill+decode batches, including the sparse-MLA prefill
  path, which is not CUDA-graph safe. `FULL` captures only uniform decode
  batches, at no measurable throughput cost
- `--moe-backend flashinfer_cutlass` — NVFP4 MoE; pinned, `auto` resolves
  here today but is free to drift
- `--kv-offloading-size 100 --kv-offloading-backend native` — native CPU KV
  offload: 100 GiB pool mmap'd in `/dev/shm`
- `--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'` —
  MTP speculative decoding, 3 draft tokens
- `--default-chat-template-kwargs '{"thinking": true}'` — force thinking mode
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
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  ngpitt/vllm:0.32.0-sm120-cu130 \
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
