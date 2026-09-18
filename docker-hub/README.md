Standalone [vLLM‎](https://github.com/nick-oconnor/vllm) inference image for the **single-outlet inference** setup documented in [sm120-enablement‎](https://github.com/nick-oconnor/sm120-enablement).

Targets **NVIDIA Blackwell consumer GPUs (SM 12.0, RTX PRO 6000 Blackwell)** for serving **GLM-5.3-Flash** (sparse-MLA MoE, 1M context, native vision) as of the `0.30.0-sm120-cu130` build; **DeepSeek-V4-Flash-0731** (fp8 MoE) was the `0.27.1-sm120-cu133` target. DeepGEMM is compiled in-image and TileLang builds from sdist; FlashInfer ships as the `flashinfer-jit-cache` wheel and CUTLASS from the `nvidia-cutlass-dsl` wheel — kernels the cache doesn't cover JIT at server startup, so `cuda-nvrtc-dev` ships in the runtime layer.

#### Image Contents
- **vLLM** built from upstream `main` past the v0.30.0rc1 fork (re-cut
  2026-09-18 as the `0.30` branch, tagged `0.30.0-sm120-cu130`) —
  **GLM-5.3-Flash model support is native upstream** since vllm-project
  #53906, so no fork overlay is needed; the ocnr commits on top are the SM120
  NoPE sparse-MLA port (fp8 + FlashInfer zero-pad, backend priority, indexer
  buffer pin) and the hybrid mamba state-seed fix (#55601, still open
  upstream). KV offloading ships unpatched (upstream native
  `CPUOffloadingSpec`, scoped to prefix-cacheable groups) and is enabled
  per-deployment with `--kv-offloading-size 100 --kv-offloading-backend
  native` (needs `--shm-size` ≥ ~110g — the pool mmaps in `/dev/shm`)
- **CUDA 13.0** runtime + toolchain (so JIT kernels compile at server startup — `cuda-nvrtc-dev` is in the runtime layer, not just the build layer)
- **FlashInfer** via the `flashinfer-jit-cache==0.6.18.post1` wheel from the flashinfer.ai index (upstream's own pin; `0.6.18` is now published), plus `set_autotune_process_group` (Xid-69 fix)
- **CUTLASS** via the `nvidia-cutlass-dsl==4.7.1` PyPI wheel (SM120 GEMM kernels; upstream's requirement since the DSL 4.7 bump)
- **b12x** via the `b12x==1.3.0` PyPI wheel — CuTe DSL PCIe one-shot all-reduce
  (`b12x.comm.pcie`); enabled per-deployment with `VLLM_ENABLE_PCIE_ALLREDUCE=1`
  (replaces NCCL-SHM for decode-size collectives at TP>2 on PCIe-only boxes)
- **FlashInfer autotune wire-up** in warmup — `VLLM_FLASHINFER_AUTOTUNE_PROCESS_GROUP=1` forces every rank into the autotune context so the autotune all-reduce doesn't hang on missing peers at long context (see [`notes/incident-longcontext-xid69.md`](https://github.com/nick-oconnor/sm120-enablement/blob/main/notes/incident-longcontext-xid69.md))
- **py-spy + `dump-jam-state.sh`** pre-installed for incident diagnostics (dumps py-spy traces, `nvidia-smi`, and dmesg Xid lines)
- `TORCH_CUDA_ARCH_LIST="12.0"` — only SM120 gets built, so the image is smaller than the upstream `vllm/vllm` images that target every arch

#### Quick Start

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
  -e VLLM_FLASHINFER_AUTOTUNE_PROCESS_GROUP=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  vllm:0.30.0-sm120-cu130 \
    /models/zai-org/GLM-5.3-Flash \
      --served-model-name GLM-5.3-Flash \
      --tensor-parallel-size 4 \
      --enable-expert-parallel \
      --trust-remote-code \
      --max-model-len auto \
      --max-num-seqs 4 \
      --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.97 \
      --kv-cache-dtype fp8 \
      --enable-prefix-caching \
      --kv-offloading-size 100 \
      --kv-offloading-backend native \
      --enable-chunked-prefill \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --enable-auto-tool-choice \
      --limit-mm-per-prompt '{"image": 1, "video": 0}' \
      --default-chat-template-kwargs '{"thinking": true}'
```
