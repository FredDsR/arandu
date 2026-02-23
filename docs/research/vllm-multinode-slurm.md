# vLLM Multi-Node Inference on SLURM: Research Report

> **Date**: 2026-02-22
> **Context**: Evaluating vLLM as a replacement for Ollama to enable multi-node LLM inference on the PCAD SLURM cluster, combining GPU resources across nodes.

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [How vLLM Works](#2-how-vllm-works)
   - [PagedAttention](#21-pagedattention)
   - [KV-Cache Management](#22-kv-cache-management)
   - [Continuous Batching](#23-continuous-batching)
   - [Additional Features](#24-additional-features)
3. [Multi-GPU and Multi-Node Inference](#3-multi-gpu-and-multi-node-inference)
   - [Tensor Parallelism (TP)](#31-tensor-parallelism-tp)
   - [Pipeline Parallelism (PP)](#32-pipeline-parallelism-pp)
   - [Data Parallelism (DP)](#33-data-parallelism-dp)
   - [Expert Parallelism (EP)](#34-expert-parallelism-ep)
   - [Recommended Parallelism Strategies](#35-recommended-parallelism-strategies)
4. [Ray as the Distributed Runtime](#4-ray-as-the-distributed-runtime)
5. [SLURM Integration](#5-slurm-integration)
   - [Modern Approach: ray symmetric-run](#51-modern-approach-ray-symmetric-run)
   - [Traditional Approach](#52-traditional-approach-manual-ray-setup)
   - [Key SLURM Directives](#53-key-slurm-directives)
6. [OpenAI-Compatible API](#6-openai-compatible-api)
7. [Comparison: vLLM vs Ollama](#7-comparison-vllm-vs-ollama)
   - [Architecture Comparison](#71-architecture-comparison)
   - [Performance Benchmarks](#72-performance-benchmarks)
   - [When to Use Which](#73-when-to-use-which)
8. [Integration with Current Architecture](#8-integration-with-current-architecture)
   - [LLMClient Compatibility](#81-llmclient-compatibility)
   - [Current Deployment Pattern](#82-current-deployment-pattern-ollama-sidecar)
   - [Proposed Deployment Pattern](#83-proposed-deployment-pattern-vllm-multi-node)
   - [Environment Variable Mapping](#84-environment-variable-mapping)
9. [Tupi Partition: Concrete Multi-Node Example](#9-tupi-partition-concrete-multi-node-example)
   - [Current Single-Node Setup](#91-current-single-node-setup)
   - [Proposed Multi-Node vLLM Setup](#92-proposed-multi-node-vllm-setup)
   - [Submitting Pipeline Jobs Against vLLM](#93-submitting-pipeline-jobs-against-vllm)
10. [Model Feasibility Matrix](#10-model-feasibility-matrix)
11. [Configuration Reference](#11-configuration-reference)
    - [vLLM Launch Parameters](#111-vllm-launch-parameters)
    - [Environment Variables](#112-environment-variables)
12. [Practical Considerations](#12-practical-considerations)
13. [Summary](#13-summary)
14. [References](#14-references)

---

## 1. Problem Statement

The current pipeline deployment on the **tupi partition** (NVIDIA RTX 4090, 24 GB VRAM) runs each step on **a single node with a single GPU**, using Ollama as the LLM backend:

```bash
# Current tupi.slurm (QA example)
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --gres=gpu:1        # Single RTX 4090 (24 GB)
```

This imposes two hard constraints:

1. **Model size ceiling**: Only models that fit in 24 GB VRAM can be loaded (e.g., `qwen3:14b`, `llama3.1:8b`). Running 70B+ parameter models like `llama3.3:70b` or `deepseek-r1` is impossible on a single tupi node.
2. **No multi-node scaling**: Ollama is architecturally designed for single-machine, single-user serving. It has no mechanism to shard a model across multiple nodes or combine GPU resources from different machines.

**Goal**: Run a single LLM model distributed across 2+ tupi nodes, combining their GPU resources (e.g., 2 nodes = 48 GB, 3 nodes = 72 GB) to serve larger, more capable models for the QA, CEP, and KG pipelines.

---

## 2. How vLLM Works

vLLM is a high-throughput, memory-efficient LLM serving engine. Its core innovations address the two biggest bottlenecks in LLM inference: GPU memory waste and batching inefficiency.

### 2.1 PagedAttention

PagedAttention is vLLM's foundational innovation, described in the paper [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023). It applies OS-style virtual memory management to the KV-cache.

**The problem**: During autoregressive decoding, each token's key and value tensors must be stored for all subsequent attention computations. For a model like LLaMA-13B, the KV-cache for a single sequence can consume up to 1.7 GB. Traditional systems allocate contiguous GPU memory for the maximum possible sequence length, leading to **60-80% memory waste** due to fragmentation and over-reservation.

**The solution — virtual memory for KV-cache**:

- **Blocks instead of pages**: The KV-cache is partitioned into fixed-size blocks (default: 16 tokens per block). Blocks do **not** need to reside in contiguous GPU memory.
- **Block table**: Each request maintains a block table that maps logical block indices to physical block addresses in GPU memory, analogous to a page table in OS virtual memory.
- **Free block pool**: A global `free_block_queue` (implemented as a doubly-linked list) manages available blocks. When a request needs more KV-cache space, blocks are allocated from this pool; when a request completes, its blocks are returned.
- **Minimal waste**: Memory waste occurs only in the last block of each sequence (internal fragmentation), resulting in **under 4% waste** in practice.

**Copy-on-write**: When multiple sequences share a common prefix (e.g., in parallel sampling or beam search), PagedAttention maps their logical blocks to the same physical blocks and tracks reference counts. A copy-on-write mechanism triggers duplication only when a shared block must be modified, reducing memory overhead by up to 55% for parallel sampling workloads.

### 2.2 KV-Cache Management

The KV-cache manager coordinates between the scheduler and the block pool:

- **Block allocation**: The `allocate_slots` function computes required blocks for a request, checks availability in the free pool, and assigns blocks via `req_to_blocks`.
- **Prefix caching**: vLLM can hash completed KV blocks and store them in a lookup table. Subsequent requests sharing the same prefix reuse cached blocks, avoiding redundant computation. This is particularly useful for pipelines that send the same system prompt repeatedly (as our QA/CEP/KG pipelines do).
- **Block size trade-off**: The default 16-token block is small enough to minimize internal fragmentation, large enough for efficient GPU memory operations.

### 2.3 Continuous Batching

Unlike Ollama's fixed parallel request limit (default: 4 concurrent requests), vLLM dynamically interleaves new requests with ongoing ones:

- **Flattened batch**: All sequences are flattened into a single "super sequence" with position indices and attention masks ensuring each sequence attends only to its own tokens. Custom CUDA kernels handle this transparently, eliminating padding waste.
- **Two-queue scheduler**: Maintains `waiting` (prefill) and `running` (decode) queues. Each step:
  1. Decode requests from the `running` queue get token generation slots (prioritized).
  2. Prefill requests from the `waiting` queue are scheduled if token budget and KV-cache blocks permit.
  3. A configurable token budget limits work per step to control latency.
- **Chunked prefill**: Long prompts are split into smaller chunks via a `long_prefill_token_threshold`, preventing a single long prompt from monopolizing GPU compute for an entire step.

**Engine step loop** (three phases per iteration):

1. **Schedule**: Select requests for decode and prefill based on token budget and KV-cache availability.
2. **Forward pass**: Execute the model via eager mode or captured CUDA graphs.
3. **Post-process**: Append generated tokens, detokenize, check stop conditions, free completed requests' KV blocks.

### 2.4 Additional Features

- **Speculative decoding**: A lightweight draft model proposes `k` tokens cheaply. The large model verifies all `k` tokens in a single forward pass, accepting or rejecting left-to-right via probability comparisons.
- **Disaggregated prefill/decode**: Prefill (compute-bound) and decode (memory-bandwidth-bound) can be separated across different worker sets with external KV-cache storage.
- **Quantization**: Native support for FP8, INT8, GPTQ, AWQ, and other quantization schemes.
- **LoRA adapter support**: Runtime loading of LoRA adapters without model reload.
- **Structured outputs**: Constrained generation via grammars (JSON mode, regex, etc.).

---

## 3. Multi-GPU and Multi-Node Inference

This section covers the parallelism strategies that enable running a single model across multiple GPUs and nodes — the core capability we need.

### 3.1 Tensor Parallelism (TP)

Tensor parallelism shards individual model layers across multiple GPUs:

- **Column parallelism**: Weight matrices are split along columns; each GPU computes its portion, and results are concatenated post-computation.
- **Row parallelism**: Matrices are divided along rows; partial results are aggregated via all-reduce operations.
- **Example (LLaMA MLP)**: Up-projection uses column parallelism, SiLU activation operates on sharded outputs, down-projection uses row parallelism with all-reduce aggregation.

**Super-linear scaling**: Moving from TP=1 to TP=2 on LLaMA-70B increased available KV-cache blocks by 13.9x (because each GPU has more free memory per its shard), yielding 3.9x token throughput — exceeding the expected 2x linear improvement.

**Requirement**: High-bandwidth interconnects like NVLink (intra-node) or InfiniBand (inter-node) are essential. Using TCP over Ethernet for TP across nodes introduces significant latency.

```bash
# Single node, 4 GPUs with tensor parallelism
vllm serve meta-llama/Llama-3.3-70B --tensor-parallel-size 4
```

### 3.2 Pipeline Parallelism (PP)

Pipeline parallelism distributes **contiguous groups of layers** across GPUs or nodes:

- Each GPU processes a distinct set of model layers.
- Intermediate activations are transmitted between GPUs/nodes once per pipeline stage.
- **Lower communication volume** compared to tensor parallelism (no all-reduce per layer), making it more tolerant of slower interconnects.
- Does not inherently decrease per-request latency; vLLM mitigates this through micro-batch optimization.

```bash
# Combined TP and PP: 4 GPUs tensor parallel, 2-way pipeline parallel
vllm serve meta-llama/Llama-3.3-70B --tensor-parallel-size 4 --pipeline-parallel-size 2
```

**This is the key strategy for multi-node deployment on networks without InfiniBand**: use TP within a node (where GPUs communicate via NVLink or PCIe) and PP across nodes (where only activations cross the network).

### 3.3 Data Parallelism (DP)

Data parallelism runs independent model replicas, each handling different requests:

- Useful when a model fits on fewer GPUs than are available.
- Three load-balancing strategies: internal, hybrid, and external.
- Configured via `--data-parallel-size`.

### 3.4 Expert Parallelism (EP)

For Mixture-of-Experts (MoE) models like DeepSeek-R1:

- Expert layers use expert parallelism instead of tensor parallelism when `--enable-expert-parallel` is set.
- Communication backends: DeepEP (NVIDIA nvshmem-based, best for multi-node) and PPLX (best for single-node).
- Automatic load balancing via **Expert Parallel Load Balancer (EPLB)**.

### 3.5 Recommended Parallelism Strategies

| Scenario | Strategy |
|----------|----------|
| Model fits on 1 GPU | No parallelism needed |
| Model fits on 1 node, multiple GPUs | `--tensor-parallel-size N` |
| Model exceeds 1 node, fast interconnect (InfiniBand) | TP across all GPUs |
| Model exceeds 1 node, slower interconnect (Ethernet) | TP within nodes + PP across nodes |
| MoE model | Expert parallelism + data parallelism for attention layers |

**For our case** (tupi partition, 1 GPU per node, Ethernet interconnect):

- `--tensor-parallel-size 1` (one GPU per node, no intra-node TP needed)
- `--pipeline-parallel-size N` (N = number of nodes)
- This gives N × 24 GB effective VRAM

---

## 4. Ray as the Distributed Runtime

Multi-node vLLM **requires Ray** as the distributed backend. Ray provides:

- **Resource discovery**: Automatically detects GPUs, CPUs, and memory on each node.
- **Worker management**: Spawns and manages worker processes across nodes.
- **Communication**: Workers communicate via Ray's object store and message queue. Each worker receives work via a broadcast queue, executes forward passes with tensor/pipeline parallelism, and returns results through individual response queues.

**Multi-node distributed serving architecture**:

```
                       HTTP requests
                            │
                      ┌─────▼─────┐
                      │  FastAPI   │  (head node)
                      │ API Server │
                      └─────┬─────┘
                            │
                      ┌─────▼──────┐
                      │ AsyncLLM + │
                      │DPCoordinator│
                      └──────┬─────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │  Engine    │ │  Engine    │ │  Engine    │
        │  (Node 0)  │ │  (Node 1)  │ │  (Node 2)  │
        │  GPU 0     │ │  GPU 0     │ │  GPU 0     │
        └───────────┘ └───────────┘ └───────────┘
```

**Request flow**: HTTP → FastAPI → `AsyncLLM` → `DPAsyncMPClient` → input socket → engine's input thread → main engine loop → output thread → response socket → FastAPI → HTTP response.

**Verification commands** after cluster setup:

```bash
ray status          # Check cluster health
ray list nodes      # List connected nodes
```

---

## 5. SLURM Integration

### 5.1 Modern Approach: `ray symmetric-run`

Available since **Ray 2.49** (November 2025), `ray symmetric-run` dramatically simplifies multi-node deployment on SLURM:

```bash
#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --partition=tupi
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_serve_%j.out
#SBATCH --error=logs/vllm_serve_%j.err

# Head node discovery
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
port=6379
ip_head=$head_node:$port

# Single command launches everything
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
    ray symmetric-run \
    --address "$ip_head" \
    --min-nodes "$SLURM_JOB_NUM_NODES" \
    --num-gpus 1 \
    --num-cpus "${SLURM_CPUS_PER_TASK}" \
    -- \
    vllm serve meta-llama/Llama-3.3-70B-Instruct \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size "$SLURM_JOB_NUM_NODES" \
        --gpu-memory-utilization 0.90 \
        --host 0.0.0.0 \
        --port 8000
```

**How it works**:

- **Head node**: Initializes Ray in `--head` mode, waits for worker registration, executes the user command, manages shutdown.
- **Worker nodes**: Runs Ray cluster initialization via `--address`, waits until job completion, self-terminates automatically.
- **Environment propagation**: Automatically propagates environment variables to all nodes.
- **Lifecycle management**: Cluster is stopped automatically when the job completes.

**Advantages over the traditional approach**:

- No manual `ray start --block` on head/workers.
- No separate terminal for job execution.
- No manual environment variable configuration per node.
- No manual `ray stop` on each node.

### 5.2 Traditional Approach (Manual Ray Setup)

If `ray symmetric-run` is not available (Ray < 2.49):

```bash
#!/bin/bash
#SBATCH --job-name=vllm-multinode
#SBATCH --partition=tupi
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

# Head node discovery
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
HEAD_NODE=${nodes_array[0]}
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
RANDOM_PORT=$(shuf -i 20000-65000 -n 1)

# Start Ray head
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    ray start --block --head --port=${RANDOM_PORT} &
sleep 10

# Start Ray workers on remaining nodes
for ((i = 1; i < ${#nodes_array[@]}; i++)); do
    srun --nodes=1 --ntasks=1 -w "${nodes_array[$i]}" \
        ray start --block --address=${HEAD_IP}:${RANDOM_PORT} &
    sleep 5
done
sleep 20

# Launch vLLM server on head node
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    vllm serve meta-llama/Llama-3.3-70B-Instruct \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 2 \
        --host 0.0.0.0 --port 8000

wait
```

### 5.3 Key SLURM Directives

| Directive | Value | Reason |
|-----------|-------|--------|
| `--tasks-per-node=1` | **Critical** | Ensures one Ray runtime per node (not multiple competing processes) |
| `--exclusive` | Recommended | Exclusive node access avoids resource contention |
| `--gres=gpu:N` | GPUs per node | Must match `--tensor-parallel-size` |
| `--cpus-per-task` | 8-32 | Ray and vLLM use CPUs for tokenization, scheduling |

**Port conflict management** in multi-tenant SLURM environments:

```bash
# Use random ports to avoid collisions with other Ray jobs
--port=$(shuf -i 20000-65000 -n 1)
--node-manager-port=6700
--object-manager-port=6701
```

---

## 6. OpenAI-Compatible API

vLLM exposes an **OpenAI-compatible HTTP API server**, making it a drop-in replacement for any OpenAI SDK client.

### Supported Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (conversational) |
| `POST /v1/completions` | Text completions (traditional) |
| `POST /v1/embeddings` | Vector embeddings |
| `GET /v1/models` | List available models |
| Tokenizer API | Token counting and encoding |

### Launch and Client Usage

```bash
# Launch server
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
    --dtype auto \
    --api-key token-abc123 \
    --host 0.0.0.0 \
    --port 8000
```

```python
# Client usage (identical to OpenAI SDK)
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

response = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### YAML Configuration

```yaml
# config.yaml
model: NousResearch/Meta-Llama-3-8B-Instruct
host: 0.0.0.0
port: 8000
dtype: auto
tensor-parallel-size: 1
pipeline-parallel-size: 2
gpu-memory-utilization: 0.90
enable-prefix-caching: true
```

```bash
vllm serve --config config.yaml
```

Priority order: command line > config file > defaults.

---

## 7. Comparison: vLLM vs Ollama

### 7.1 Architecture Comparison

| Aspect | vLLM | Ollama |
|--------|------|--------|
| **Target use case** | High-throughput production serving | Local development, single-user |
| **Inference backend** | Custom CUDA kernels, PagedAttention | llama.cpp (GGML/GGUF) |
| **Batching** | Continuous batching (dynamic) | Fixed parallel limit (default: 4) |
| **Memory management** | PagedAttention (< 4% waste) | Standard contiguous allocation |
| **Multi-GPU** | TP, PP, DP, EP | Limited/no multi-GPU support |
| **Multi-node** | Full support via Ray | **Not supported** |
| **API** | OpenAI-compatible (`/v1/`) | OpenAI-compatible (`/v1/`) |
| **Setup complexity** | Moderate (CUDA, Python, Ray) | Minimal (`ollama run model`) |
| **Model format** | HuggingFace (FP16/BF16/FP8/GPTQ/AWQ) | GGUF (llama.cpp quantization) |
| **Model management** | HuggingFace Hub download | `ollama pull` (curated registry) |

### 7.2 Performance Benchmarks

From [Red Hat's benchmarking study](https://developers.redhat.com/articles/2025/08/08/ollama-vs-vllm-deep-dive-performance-benchmarking) on identical hardware (single NVIDIA A100-PCIE-40GB, LLaMA 3.1-8B):

| Metric | vLLM | Ollama | Ratio |
|--------|------|--------|-------|
| Peak throughput (tokens/sec) | 793 | 41 | **~19x** |
| P99 latency at peak | 80 ms | 673 ms | **~8.4x better** |
| TTFT scaling under load | Stable | Rises dramatically | — |
| Max concurrent users tested | 256 | Plateaus at 32 | — |
| Throughput scaling | Near-linear | Flat after 32 parallel | — |

### 7.3 When to Use Which

**Use vLLM when**:

- Serving multiple concurrent users/workers in production.
- Need multi-GPU or multi-node inference for large models (70B+, 405B).
- Throughput and latency at scale matter (our batch pipelines with 4-8 workers).
- Running MoE models (DeepSeek-R1) that benefit from expert parallelism.
- Deploying on HPC/SLURM clusters.

**Use Ollama when**:

- Local development and quick prototyping.
- Single-user, single-GPU scenarios.
- Quick model testing with minimal setup.
- CPU-only or consumer hardware (GGUF quantization is highly optimized for CPU).

---

## 8. Integration with Current Architecture

### 8.1 LLMClient Compatibility

Our `LLMClient` (`src/gtranscriber/core/llm_client.py`) already uses the **OpenAI SDK** with a configurable `base_url`:

```python
class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    CUSTOM = "custom"     # <-- vLLM goes here

class LLMClient:
    PROVIDER_URLS: ClassVar[dict[LLMProvider, str | None]] = {
        LLMProvider.OPENAI: None,
        LLMProvider.OLLAMA: "http://localhost:11434/v1",
        LLMProvider.CUSTOM: None,  # Must be provided explicitly
    }
```

vLLM exposes an OpenAI-compatible API at `http://host:8000/v1`. The existing `LLMProvider.CUSTOM` mode works **with zero code changes**:

```python
# Current Ollama usage
client = LLMClient(LLMProvider.OLLAMA, "qwen3:14b", base_url="http://ollama:11434/v1")

# vLLM replacement — zero code changes
client = LLMClient(LLMProvider.CUSTOM, "Qwen/Qwen3-14B", base_url="http://vllm-head:8000/v1")
```

Both use `client.chat.completions.create()` from the OpenAI SDK. The retry logic, thinking trace extraction, and response format handling all work identically.

### 8.2 Current Deployment Pattern (Ollama Sidecar)

```
SLURM Job (1 node, 1 GPU)
    │
    ├── docker compose up ollama-gpu     # Ollama sidecar (holds GPU)
    │   └── ollama pull qwen3:14b        # Download GGUF model
    │
    └── docker compose up gtranscriber-qa  # Pipeline container (CPU)
        └── LLMClient → http://ollama:11434/v1
```

**Key characteristics**:

- Sidecar pattern: Ollama starts/stops with each pipeline job.
- Services communicate via Docker internal network (`ollama:11434`).
- Single node, single GPU — no way to scale beyond 24 GB VRAM.
- Model must be re-pulled if evicted from cache between jobs.

### 8.3 Proposed Deployment Pattern (vLLM Multi-Node)

```
SLURM Job A: vLLM Server (N nodes, N GPUs)
    │
    ├── Node 0 (head): Ray head + vLLM API server → :8000/v1
    │   └── Pipeline parallel stage 0 (layers 0-15)
    │
    ├── Node 1 (worker): Ray worker
    │   └── Pipeline parallel stage 1 (layers 16-31)
    │
    └── Node N (worker): Ray worker
        └── Pipeline parallel stage N (layers ...)

SLURM Job B: Pipeline (1 node, CPU only)
    │
    └── docker compose up gtranscriber-qa
        └── LLMClient → http://<head-node>:8000/v1
```

**Key differences**:

| Aspect | Current (Ollama) | Proposed (vLLM) |
|--------|------------------|-----------------|
| Container pattern | Sidecar (same job) | Separate long-running service |
| GPU allocation | 1 GPU, 1 node | N GPUs across M nodes |
| Model loading | `ollama pull model` (GGUF) | HuggingFace download (native format) |
| API endpoint | `http://ollama:11434/v1` | `http://head-node:8000/v1` |
| Max model size | ~14B (24 GB VRAM) | 70B+ across nodes |
| Concurrent throughput | ~41 tokens/sec | ~793 tokens/sec (single GPU) |
| Multi-node | Not supported | Native via Ray |

### 8.4 Environment Variable Mapping

To switch from Ollama to vLLM, override these environment variables:

```bash
# QA Pipeline
export GTRANSCRIBER_QA_PROVIDER=custom
export GTRANSCRIBER_QA_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
export GTRANSCRIBER_LLM_BASE_URL=http://<vllm-head-node>:8000/v1

# KG Pipeline
export GTRANSCRIBER_KG_PROVIDER=custom
export GTRANSCRIBER_KG_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
export GTRANSCRIBER_KG_BASE_URL=http://<vllm-head-node>:8000/v1

# CEP Pipeline (generator + validator can use the same vLLM instance)
export GTRANSCRIBER_CEP_VALIDATOR_PROVIDER=custom
export GTRANSCRIBER_CEP_VALIDATOR_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
```

The `QAConfig` and `KGConfig` classes in `src/gtranscriber/config.py` already support `provider: str = "custom"` and `base_url: str | None` fields, so no configuration code changes are required.

---

## 9. Tupi Partition: Concrete Multi-Node Example

### 9.1 Current Single-Node Setup

From `scripts/slurm/qa/tupi.slurm`:

```bash
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                    # 1x RTX 4090 (24 GB)
#SBATCH --time=12:00:00

export GTRANSCRIBER_QA_WORKERS=4
export GTRANSCRIBER_QA_MODEL_ID=qwen3:14b   # Must fit in 24 GB
export USE_GPU_OLLAMA=true
```

**Limitation**: The model is constrained to ~14B parameters. With GGUF Q4 quantization, Ollama can sometimes squeeze in larger models, but quality degrades and there's no path to 70B+.

### 9.2 Proposed Multi-Node vLLM Setup

**Step 1**: SLURM script to launch vLLM across 2 tupi nodes.

```bash
#!/bin/bash
#SBATCH --job-name=vllm-serve-tupi
#SBATCH --partition=tupi
#SBATCH --nodes=2                        # 2 tupi nodes
#SBATCH --exclusive
#SBATCH --tasks-per-node=1               # 1 Ray runtime per node
#SBATCH --gres=gpu:1                     # 1x RTX 4090 per node = 48 GB total
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_serve_%j.out
#SBATCH --error=logs/vllm_serve_%j.err

# Configuration
PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"
export HF_HOME="${PROJECT_DIR}/cache/huggingface"
MODEL_ID="${VLLM_MODEL_ID:-meta-llama/Llama-3.3-70B-Instruct}"

echo "=============================================="
echo "vLLM Multi-Node Server"
echo "=============================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Nodes:      ${SLURM_JOB_NUM_NODES}"
echo "Model:      ${MODEL_ID}"
echo "Partition:  ${SLURM_JOB_PARTITION}"
echo "=============================================="

# Head node discovery
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
port=6379
ip_head=$head_node:$port

echo "Head node:  $head_node"
echo "Ray addr:   $ip_head"

# Launch vLLM with ray symmetric-run
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
    ray symmetric-run \
    --address "$ip_head" \
    --min-nodes "$SLURM_JOB_NUM_NODES" \
    --num-gpus 1 \
    --num-cpus "${SLURM_CPUS_PER_TASK}" \
    -- \
    vllm serve "$MODEL_ID" \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size "$SLURM_JOB_NUM_NODES" \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --dtype auto \
        --host 0.0.0.0 \
        --port 8000
```

### 9.3 Submitting Pipeline Jobs Against vLLM

**Option A**: Submit vLLM server first, then pipeline with dependency.

```bash
# 1. Submit vLLM server (starts on 2 tupi nodes)
VLLM_JOB_ID=$(sbatch --parsable scripts/slurm/vllm/tupi-serve.slurm)
echo "vLLM server job: $VLLM_JOB_ID"

# 2. Wait for allocation to get head node name
sleep 10  # Wait for SLURM to allocate
VLLM_HEAD=$(scontrol show job "$VLLM_JOB_ID" \
    | grep -oP 'NodeList=\K[^,\s]+' | head -1)
echo "vLLM head node: $VLLM_HEAD"

# 3. Submit QA pipeline with dependency (starts after vLLM is running)
GTRANSCRIBER_QA_PROVIDER=custom \
GTRANSCRIBER_QA_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct \
GTRANSCRIBER_LLM_BASE_URL="http://${VLLM_HEAD}:8000/v1" \
sbatch --dependency=after:${VLLM_JOB_ID}+1 scripts/slurm/qa/tupi.slurm
```

**Option B**: Wrapper script that launches both.

```bash
#!/bin/bash
# scripts/slurm/vllm/run-pipeline-with-vllm.sh

VLLM_NODES="${VLLM_NODES:-2}"
PIPELINE="${1:?Usage: $0 <qa|cep|kg>}"

# Submit vLLM server
VLLM_JOB=$(sbatch --parsable \
    --nodes="$VLLM_NODES" \
    scripts/slurm/vllm/tupi-serve.slurm)

echo "vLLM server job: $VLLM_JOB"

# Submit pipeline job (dependency ensures vLLM starts first)
# The pipeline script needs to be modified to skip Ollama sidecar
# when GTRANSCRIBER_QA_PROVIDER=custom
sbatch --dependency=after:${VLLM_JOB}+2 \
    --export=ALL,GTRANSCRIBER_QA_PROVIDER=custom \
    "scripts/slurm/${PIPELINE}/tupi.slurm"
```

> **Note**: The existing `qa_common.sh`, `kg_common.sh`, and `cep_common.sh` scripts include Ollama sidecar startup logic. When using vLLM, these scripts would need a conditional to skip the Docker Compose Ollama sidecar when `provider=custom`, or alternatively, new `*_vllm_common.sh` scripts could be created that omit the sidecar logic entirely.

---

## 10. Model Feasibility Matrix

What models become possible with multi-node tupi deployment:

| Model | Parameters | VRAM (FP16) | VRAM (FP8) | 1 Node (24 GB) | 2 Nodes (48 GB) | 3 Nodes (72 GB) |
|-------|-----------|-------------|------------|-----------------|------------------|------------------|
| Qwen3-14B | 14B | ~28 GB | ~14 GB | FP8 only | Yes | Yes |
| LLaMA 3.1-8B | 8B | ~16 GB | ~8 GB | Yes | Yes | Yes |
| LLaMA 3.3-70B | 70B | ~140 GB | ~40 GB | No | **FP8 yes** | FP8 yes |
| Qwen3-72B | 72B | ~144 GB | ~42 GB | No | **FP8 yes** | FP8 yes |
| DeepSeek-R1-70B | 70B | ~140 GB | ~40 GB | No | **FP8 yes** | FP8 yes |
| Qwen3-235B (MoE) | 235B | ~470 GB | ~135 GB | No | No | No (need ~6 nodes) |
| LLaMA 3.1-405B | 405B | ~810 GB | ~220 GB | No | No | No (need ~10 nodes) |

> **Note**: VRAM estimates include model weights + KV-cache overhead. FP8 quantization halves model weight memory with minimal quality loss. With `--gpu-memory-utilization 0.90`, approximately 21.6 GB per RTX 4090 is usable.

---

## 11. Configuration Reference

### 11.1 vLLM Launch Parameters

```bash
vllm serve <model> \
    --tensor-parallel-size 1 \           # GPUs for tensor parallelism (per node)
    --pipeline-parallel-size 2 \          # Stages for pipeline parallelism (nodes)
    --data-parallel-size 1 \              # Number of DP replicas
    --gpu-memory-utilization 0.90 \       # Fraction of GPU memory for KV cache
    --max-model-len 32768 \               # Maximum sequence length
    --dtype auto \                        # Data type (auto, float16, bfloat16)
    --quantization fp8 \                  # Quantization method
    --enable-prefix-caching \             # Cache common prefixes (good for our pipelines)
    --enable-chunked-prefill \            # Chunk long prompts
    --host 0.0.0.0 \                      # Server bind address
    --port 8000 \                         # Server port
    --api-key token-abc123 \              # API authentication key
    --config config.yaml                  # YAML configuration file
```

### 11.2 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_HOST_IP` | IP of current node in distributed setup | `""` |
| `VLLM_PORT` | Communication port | `0` |
| `VLLM_USE_RAY_SPMD_WORKER` | Execute workers as separate Ray processes | `0` |
| `VLLM_RAY_PER_WORKER_GPUS` | GPUs per Ray worker | `1.0` |
| `VLLM_PP_LAYER_PARTITION` | Pipeline stage partitioning strategy | `None` |
| `VLLM_ATTENTION_BACKEND` | Attention backend (FLASH_ATTN, TORCH_SDPA) | Auto |
| `VLLM_API_KEY` | API key for the server | `None` |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | Engine iteration timeout (seconds) | `60` |
| `NCCL_SOCKET_IFNAME` | Network interface for NCCL (e.g., `eth0`) | System default |
| `NCCL_IB_HCA` | InfiniBand adapter (e.g., `mlx5`) | Auto-detect |
| `NCCL_DEBUG` | NCCL debug level (`TRACE` for diagnostics) | `WARN` |
| `HF_HOME` | HuggingFace cache directory | `~/.cache/huggingface` |

**NCCL communication verification**:

```bash
NCCL_DEBUG=TRACE vllm serve ...
# Look for:
#   NET/IB/GDRDMA  → InfiniBand with GPU-Direct RDMA (optimal)
#   NET/Socket     → TCP Socket (works but slower for TP across nodes)
```

---

## 12. Practical Considerations

### 12.1 Network Bandwidth

Multi-node inference performance depends on inter-node bandwidth. **Pipeline parallelism** has lower bandwidth requirements than tensor parallelism — only intermediate activations cross nodes (not per-layer all-reduce). This makes PP more suitable for Ethernet-connected clusters like PCAD.

If PCAD has InfiniBand between tupi nodes, tensor parallelism across nodes becomes viable. Check with:

```bash
ibstat              # Check for InfiniBand adapters
ibstatus            # Check link status
```

### 12.2 Model Format Change

Ollama uses **GGUF** format (llama.cpp quantized). vLLM uses **HuggingFace format** with its own quantization (FP8, GPTQ, AWQ). This means:

- Model IDs change: `qwen3:14b` (Ollama) → `Qwen/Qwen3-14B` (HuggingFace)
- Models are downloaded from HuggingFace Hub, not the Ollama registry
- The existing `HF_HOME` cache (`$PROJECT_DIR/cache/huggingface`) is reused

### 12.3 Service Lifecycle

Unlike the current sidecar pattern (Ollama starts/stops with each pipeline job), vLLM multi-node is better as a **long-running service**:

- Ray cluster setup takes 30-60 seconds; avoid paying this cost per pipeline step.
- Submit vLLM as a long SLURM job (24h), then submit multiple pipeline jobs against it.
- Use SLURM job dependencies (`--dependency=after:JOBID`) to sequence pipeline steps.

### 12.4 Container Runtime

The current Docker Compose sidecar pattern won't work for multi-node vLLM (containers on different SLURM nodes can't share a Docker network). Options:

| Approach | Pros | Cons |
|----------|------|------|
| **Bare metal** (pip/uv install) | Simplest setup | Environment management |
| **Apptainer/Singularity** | HPC-native, `--nv` GPU support | Container build step |
| **Enroot + Pyxis** | NVIDIA's SLURM-native runtime | Requires cluster-level install |

**Recommended**: Install vLLM + Ray in a Python virtual environment (`uv`) on PCAD, or build an Apptainer `.sif` image. The pipeline containers (gtranscriber-qa, etc.) can still use Docker on the head node, connecting to vLLM via HTTP instead of Docker network.

### 12.5 Dependencies

```
vllm >= 0.8.0
ray >= 2.49       # For ray symmetric-run (2.40+ for manual approach)
torch >= 2.4.0
```

### 12.6 Graceful Shutdown

When the vLLM SLURM job is cancelled or reaches its time limit, Ray handles graceful shutdown. However, pipeline jobs connected to the vLLM server will receive connection errors. The existing retry logic in `LLMClient` (3 attempts with exponential backoff) provides some resilience, but pipeline jobs should ideally be cancelled before the vLLM server job.

---

## 13. Summary

| Dimension | Finding |
|-----------|---------|
| **Core capability** | vLLM is the only production-grade framework with native multi-node LLM inference via Ray |
| **Code changes needed** | **Zero** — `LLMProvider.CUSTOM` + `base_url` already supported in `LLMClient` |
| **Config changes needed** | Set `provider=custom` and `base_url=http://head:8000/v1` via environment variables |
| **Deployment changes** | Replace Ollama Docker sidecar with separate vLLM SLURM job using `ray symmetric-run` |
| **Immediate win (2 tupi nodes)** | 48 GB combined VRAM → unlocks 70B-class models (LLaMA 3.3, Qwen3-72B, DeepSeek-R1) in FP8 |
| **Throughput improvement** | ~19x higher throughput than Ollama, even on single GPU |
| **Network recommendation** | Use pipeline parallelism across nodes (tolerant of Ethernet); verify InfiniBand availability |
| **Runtime recommendation** | Bare metal or Apptainer (not Docker) for vLLM on SLURM; pipeline containers can keep Docker |

---

## 14. References

- [Efficient Memory Management for Large Language Model Serving with PagedAttention (Paper)](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention (Blog)](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [PagedAttention — vLLM Design Docs](https://docs.vllm.ai/en/stable/design/paged_attention/)
- [Parallelism and Scaling — vLLM Docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [Distributed Inference and Serving — vLLM Docs](https://docs.vllm.ai/en/v0.9.0/serving/distributed_serving.html)
- [OpenAI-Compatible Server — vLLM Docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
- [vLLM Environment Variables](https://docs.vllm.ai/en/v0.8.1/serving/env_vars.html)
- [Streamlined Multi-Node Serving with Ray Symmetric-Run](https://blog.vllm.ai/2025/11/22/ray-symmetric-run.html)
- [Multi-node & Multi-GPU Inference with vLLM (MeluXina)](https://docs.lxp.lu/howto/llama3-vllm/)
- [Deploying Ray on SLURM](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html)
- [Distributed Inference with vLLM — Red Hat](https://developers.redhat.com/articles/2025/02/06/distributed-inference-with-vllm)
- [Ollama vs. vLLM: Performance Benchmarking — Red Hat](https://developers.redhat.com/articles/2025/08/08/ollama-vs-vllm-deep-dive-performance-benchmarking)
- [vLLM vs Ollama: Key Differences — Northflank](https://northflank.com/blog/vllm-vs-ollama-and-how-to-run-them)
- [The vLLM MoE Playbook — AMD ROCm](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
