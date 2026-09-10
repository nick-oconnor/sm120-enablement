# Single-Outlet Inference

A workstation serving [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
(sparse-MLA MoE with native vision, 1M-token context) with
[vLLM](https://github.com/nick-oconnor/vllm). Previously: DeepSeek-V4-Flash-0731
and MiniMax-M3-NVFP4 — their configs live in
[`notes/`](notes/).

![system](images/system.jpg)

## Benchmarks

GLM-5.3-Flash, vLLM `0.29.0-sm120-cu130` with the b12x PCIe one-shot all-reduce
(2026-09-02). 16 prompts, concurrency 1, random dataset, zero failed requests.

| Input Tokens | Output Tokens | Decode (tok/s) | Median TTFT | Median ITL |
| --------- | ---------- | -------------- | --------- | -------- |
| 2048      | 256        | 88.3           | 225ms     | 10.48ms  |
| 8192      | 1024       | 88.7           | 814ms     | 10.50ms  |
| 32768     | 4096       | 88.9           | 3112ms    | 10.49ms  |
| 131072    | 8192       | 84.0           | 9977ms    | 10.63ms  |

PSU output (self-reported via the PSU's USB interface): 234W idle, 1.28kW under bench load, 1.76kW peak.

## Hardware

| Component | Spec |
|---|---|
| Motherboard | Gigabyte MH53-G40 |
| CPU | 1x AMD Ryzen Threadripper PRO 9985WX |
| Memory | 8x Kingston KF556R28-32 RDIMM, EXPO 1 |
| GPU | 4x NVIDIA RTX PRO 6000 Blackwell Max-Q, ECC enabled |
| Interconnect | 4x PCIe Gen 5 x16 |
| PSU | 1x Corsair HX1500i, 20A 120V circuit |
| OS / driver | Debian 13, kernel 6.12.101, NVIDIA 610.57.04, CUDA 13.3 |

Memory Bandwidth:

| Threads | Pinning | Copy |
| --- | --- | --- |
| 8 | 1 per CCD | 263GB/s |
| 16 | 2 per CCD (SMT pair) | 293GB/s |
| 32 | 4 per CCD | 209GB/s |
| 128 | default | 137GB/s |

PCIe Topology:

|     | GPU0 | GPU1 | GPU2 | GPU3 |
| --- | ---- | ---- | ---- | ---- |
| GPU0 | X    | NODE | NODE | NODE |
| GPU1 | NODE | X    | NODE | NODE |
| GPU2 | NODE | NODE | X    | NODE |
| GPU3 | NODE | NODE | NODE | X    |

PCIe Speed (between GPU pairs):

| P2P        | Unidirectional | Bidirectional | Latency (GPU) | Latency (CPU) |
| ---------- | -------------- | ------------- | ------------- | ------------- |
| disabled   | 44GB/s         | 58GB/s        | 14us          | 5us           |
| enabled    | 55GB/s         | 105GB/s       | 0.5us         | 1us           |

## Model

- [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
  (`Glm5NextForConditionalGeneration`, sparse-MLA MoE, 1M-token context, native
  vision: 448px tiles / patch 14 / 256 tokens per tile)
- 62-shard checkpoint served from `/models/zai-org/GLM-5.3-Flash` on the local
  models mount; SM120 serving path is the hardware-verified fp8 + FlashInfer
  NoPE sparse-MLA port (see `notes/serving-config.md`)
- Previous models on this rig: DeepSeek-V4-Flash-0731, MiniMax-M3-NVFP4

## vLLM Build

Fork: [github.com/nick-oconnor/vllm](https://github.com/nick-oconnor/vllm),
branch `0.29`, tagged `0.29.0-sm120-cu130` (upstream `0.29`-era base —
GLM-5.3-Flash model support is native upstream since vllm-project #53906; the
branch carries the ocnr SM120 NoPE port and kv-offload fixes).

Build constraints:

- GPUs are SM 12.0 (Blackwell consumer). `TORCH_CUDA_ARCH_LIST="12.0"`.
- Several SM120 kernel paths (DeepGEMM fp8 MoE, TileLang, FlashInfer runtime
  JIT) need the matching CUDA toolchain; CUTLASS comes from the
  `nvidia-cutlass-dsl==4.6.2` PyPI wheel.
- **b12x** comes from the `b12x==1.3.0` PyPI wheel — CuTe DSL PCIe one-shot
  all-reduce kernels (`b12x.comm.pcie`); no native extension build.
- FlashInfer and TileLang JIT kernels at server startup (the boot log shows
  TileLang compiling `mhc_pre_big_fuse_*` on each worker). The `cuda-nvrtc-dev`
  package must be in the runtime image, not just the build image, or the server
  fails to boot.

```bash
git clone --branch 0.29 https://github.com/nick-oconnor/vllm.git
cd vllm
docker build -f docker/Dockerfile -t vllm:0.29.0-sm120-cu130 .
```

Pre-built amd64 image: [ngpitt/vllm:0.29.0-sm120-cu130](https://hub.docker.com/r/ngpitt/vllm/tags?name=0.29.0-sm120-cu130).

## vLLM Execution

```bash
docker run --rm --gpus all --shm-size 120g \
  -v <host-models-path>:/models:ro \
  -v <host-cache-path>:/home/vllm \
  -p 8000:8000 \
# weights come from the local /models mount, not the Hub
  -e HF_HUB_OFFLINE=1 \
# NCCL can't auto-detect P2P from inside the container; declare it manually
  -e NCCL_P2P_LEVEL=NODE \
# cap the Rayon thread pool (used by tokenizers/parquet)
  -e RAYON_NUM_THREADS=4 \
  -e OMP_NUM_THREADS=4 \
# cap JIT parallelism so container PIDs stay sane
  -e MAX_JOBS=32 \
# sync FlashInfer autotune tactic choice across TP ranks during warmup
  -e VLLM_FLASHINFER_AUTOTUNE_PROCESS_GROUP=1 \
# host-side barrier after OffloadingConnector.start_load_kv to prevent the
# TP rank desync on KV load
  -e VLLM_KV_OFFLOAD_COLLECTIVE_BARRIER=1 \
# b12x PCIe one-shot all-reduce replaces NCCL-SHM for decode-size collectives
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  vllm:0.29.0-sm120-cu130 \
    /models/zai-org/GLM-5.3-Flash \
      --served-model-name GLM-5.3-Flash \
# 4-way TP across the four GPUs
      --tensor-parallel-size 4 \
      --enable-expert-parallel \
      --trust-remote-code \
# resolves to the full 1,048,576-token context; auto-fit confirms the
# ~7.91 GiB/GPU fp8 KV cache holds 1,095,931 tokens (1.05x concurrency)
      --max-model-len auto \
      --max-num-seqs 4 \
      --max-num-batched-tokens 8192 \
# keeps the full 1M with the vision stack resident
      --gpu-memory-utilization 0.97 \
      --kv-cache-dtype fp8 \
# 100 GiB host-RAM offload buffer; speeds up long-context requests under
# concurrent load
      --kv-offloading-size 100 \
      --kv-offloading-backend native \
      --enable-prefix-caching \
      --enable-chunked-prefill \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --enable-auto-tool-choice \
      --limit-mm-per-prompt '{"image": 1, "video": 0}' \
# force thinking mode on every turn
      --default-chat-template-kwargs '{"thinking": true}'
```

---

Operational deep-dive — build/deploy config, the full launch-blocker list, and
production-incident post-mortems — lives in [`notes/`](notes/).
