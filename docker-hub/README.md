Standalone [vLLM‎](https://github.com/nick-oconnor/vllm) inference image for the **single-outlet inference** setup documented in [sm120-enablement‎](https://github.com/nick-oconnor/sm120-enablement).

Targets **NVIDIA Blackwell consumer GPUs (SM 12.0, RTX PRO 6000 Blackwell)** for serving **DeepSeek-V4-Flash-0731** (fp8 MoE). The SM120 kernel paths (FlashInfer FP4/CUTLASS MoE, DeepGEMM, TileLang) are built from source — FlashInfer's stock wheels and tagged CUTLASS releases do not ship the SM120 paths. FlashInfer JITs the SM120 CUTLASS MoE kernels at server startup, so `cuda-nvrtc-dev` ships in the runtime layer.

#### Image Contents
- **vLLM** from the fork's `0.27` branch (built and tagged as `0.27.1-sm120-cu133`), rebased onto v0.27.1 which supports DeepSeek-V4 natively, with the ocnr SM120-specific config on top
- **CUDA 13.3** runtime + toolchain (so JIT MoE kernels compile at server startup — `cuda-nvrtc-dev` is in the runtime layer, not just the build layer)
- **FlashInfer** built from source at `v0.6.16.post3` for the SM120 FP4/CUTLASS MoE kernels, plus `set_autotune_process_group` (Xid-69 fix)
- **CUTLASS** via the `nvidia-cutlass-dsl==4.5.2` PyPI wheel (SM120 GEMM kernels)
- **KV-offload collective barrier** patch — `VLLM_KV_OFFLOAD_COLLECTIVE_BARRIER=1` keeps TP ranks in sync when KV is reloaded from host RAM (otherwise the offload path deadlocks; see [`notes/incident-kv-offload-deadlock.md`](https://github.com/nick-oconnor/sm120-enablement/blob/main/notes/incident-kv-offload-deadlock.md))
- **FlashInfer autotune wire-up** in warmup — `VLLM_FLASHINFER_AUTOTUNE_PROCESS_GROUP=1` forces every rank into the autotune context so the autotune all-reduce doesn't hang on missing peers at long context (see [`notes/incident-longcontext-xid69.md`](https://github.com/nick-oconnor/sm120-enablement/blob/main/notes/incident-longcontext-xid69.md))
- **py-spy + `dump-jam-state.sh`** pre-installed for incident diagnostics (dumps py-spy traces, `nvidia-smi`, and dmesg Xid lines)
- `TORCH_CUDA_ARCH_LIST="12.0"` — only SM120 gets built, so the image is smaller than the upstream `vllm/vllm` images that target every arch

#### Quick Start

```bash
docker run --rm --gpus all --shm-size 120g \
  -v <host-models-path>:/models:ro \
  -v <host-cache path>:/home/vllm \
  -p 8000:8000 \
  -e HF_HUB_OFFLINE=1 \
  -e NCCL_P2P_LEVEL=NODE \
  -e RAYON_NUM_THREADS=4 \
  -e OMP_NUM_THREADS=4 \
  -e MAX_JOBS=32 \
  -e VLLM_FLASHINFER_AUTOTUNE_PROCESS_GROUP=1 \
  -e VLLM_KV_OFFLOAD_COLLECTIVE_BARRIER=1 \
  vllm:0.27.1-sm120-cu133 \
    /models/deepseek-ai/DeepSeek-V4-Flash-0731 \
      --served-model-name DeepSeek-V4-Flash-0731 \
      --tensor-parallel-size 4 \
      --enable-expert-parallel \
      --trust-remote-code \
      --tokenizer-mode deepseek_v4 \
      --max-model-len auto \
      --max-num-seqs 4 \
      --max-num-batched-tokens 16384 \
      --gpu-memory-utilization 0.7 \
      --kv-cache-dtype fp8 \
      --kv-offloading-size 100 \
      --kv-offloading-backend native \
      --block-size 256 \
      --enable-prefix-caching \
      --enable-chunked-prefill \
      --tool-call-parser deepseek_v4 \
      --reasoning-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --default-chat-template-kwargs '{"thinking": true}'
```
