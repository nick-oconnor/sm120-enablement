# vLLM for SM120

Standalone [vLLM](https://github.com/nick-oconnor/vllm) inference image for the
**single-outlet inference** setup documented in
[sm120-enablement](https://github.com/nick-oconnor/sm120-enablement).

Targets **NVIDIA Blackwell consumer GPUs (SM 12.0, RTX PRO 6000 Blackwell)**
for serving **GLM-5.3-Flash** (sparse-MLA MoE, 1M context, native vision).

## Image Contents

- **vLLM** built from the `0.30` branch of
  [github.com/nick-oconnor/vllm](https://github.com/nick-oconnor/vllm), tagged
  `0.30.0-sm120-cu130` (upstream `main` base `924707f1bf`). GLM-5.3-Flash model
  support is native upstream since vllm-project #53906 — no fork overlay is
  needed. The image carries, vs upstream:

   - the SM120 NoPE sparse-MLA port (fp8 + FlashInfer zero-pad, backend
     priority, indexer buffer pin)
   - the b12x PCIe one-shot all-reduce backend (`VLLM_ENABLE_PCIE_ALLREDUCE`)
   - the SM120 image build (Blackwell consumer)
   - #55222 — right-sizes the indexer prefill workspace and sizes the prefill
     chunk budget in compressed rows; without it, #55221 caps auto-fit
     `max_model_len` at 516K and loses the 1M context (merged upstream after
     this branch's base)
   - #55601 — seeds the hybrid mamba state index by `mamba_block_size`
     (the prefix-cache KV-corruption fix; open upstream)
   - #57635 — per-rank FlashInfer autotune cache files (open upstream)
   - SM120 sparse-MLA warmup tunes all ranks; leader-only tuning left
     asymmetric tuner state that deadlocked the synchronized autotune
   - mamba aligned-split uses the scheduler-resolved block size;
     `cache_config.block_size` gets reassigned to the smallest
     prefix-cacheable group

  Upstream's #57477 (kpool tail-seed stride fix for the silent KV-cache
  poisoning) is in the base. KV offloading is upstream native
  (`CPUOffloadingSpec`, scoped to prefix-cacheable groups), enabled in the
  quick start below.

- **CUDA 13.0** runtime + toolchain — DeepGEMM is compiled in-image and
  TileLang builds from sdist; kernels the FlashInfer / CUTLASS wheels don't
  cover JIT at server startup, so `cuda-nvrtc-dev` ships in the runtime layer,
  not just the build layer
- **FlashInfer 0.7.0** via the `flashinfer-python` + `flashinfer-cubin` wheels
  from the flashinfer.ai index (upstream's own pin); autotune tactic sync
  across TP ranks is upstream-native, with per-rank cache persistence carried
  on top
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

| Flag | Notes |
| --- | --- |
| `--shm-size` | NCCL bootstrap, engine IPC, 100 GiB KV-offload mmap |
| `HF_HUB_OFFLINE` | weights come from the local `/models` mount, not the Hub |
| `NCCL_P2P_LEVEL` | NCCL can't auto-detect P2P inside the container |
| `RAYON_NUM_THREADS` | cap the Rayon thread pool (tokenizers/parquet) |
| `OMP_NUM_THREADS` | cap the OpenMP thread pool |
| `MAX_JOBS` | cap JIT parallelism so container PIDs stay sane |
| `VLLM_ENABLE_PCIE_ALLREDUCE` | enables the b12x one-shot all-reduce backend |
| `--tensor-parallel-size` | 4-way TP across the four GPUs |
| `--max-model-len` | auto-fit; holds the full 1M via the #55222 right-size |
| `--gpu-memory-utilization` | leaves ~24 GiB/GPU free for the image workload |
| `--moe-backend` | NVFP4 MoE; pinned, `auto` may drift |
| `--kv-offloading-*` | native CPU KV offload: 100 GiB pool in `/dev/shm` |
| `--speculative-config` | MTP speculative decoding, 3 draft tokens |
| `--default-chat-template-kwargs` | force thinking mode on every turn |

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
  ngpitt/vllm:0.30.0-sm120-cu130 \
    /models/nvidia/GLM-5.3-Flash-NVFP4 \
      --served-model-name GLM-5.3-Flash \
      --tensor-parallel-size 4 \
      --enable-expert-parallel \
      --trust-remote-code \
      --max-model-len auto \
      --max-num-seqs 4 \
      --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.69 \
      --moe-backend flashinfer_cutlass \
      --kv-cache-dtype fp8 \
      --enable-prefix-caching \
      --kv-offloading-size 100 \
      --kv-offloading-backend native \
      --enable-chunked-prefill \
      --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --enable-auto-tool-choice \
      --limit-mm-per-prompt '{"image": 20, "video": 0}' \
      --default-chat-template-kwargs '{"thinking": true}'
```
