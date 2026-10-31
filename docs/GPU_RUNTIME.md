# Morpheus GPU Runtime v0.3

Morpheus now has an optional native NVIDIA CUDA compute substrate intended for both general numerical work and Perseus-class cognition/memory workloads. GPU support is additive: the reference interpreter, ordinary `g++` native builds, and CPU-only programs keep their existing behavior and zero-GPU dependency path.

The architectural rule is simple:

> Morpheus owns the compute contract; CUDA is the first backend.

Morpheus source sees opaque numeric tensor handles. Raw device pointers never cross the language boundary. That gives the runtime ownership, lifetime tracking, error handling, and a stable surface that can later be implemented by HIP, Metal, Vulkan, or CPU SIMD without changing Perseus source.

## Runtime generations

### v1 — foundational GPU tensors

`runtime/gpu/morph_gpu.cu`

- CUDA device discovery and device name
- free/total VRAM reporting
- synchronization
- dense row-major float32 tensor allocation/free
- CSV bootstrap transfer
- binary float32 load/save
- scalar diagnostic get/set
- cuBLAS SGEMM
- add / scale / ReLU

### v2 — AI / transformer primitives

`runtime/gpu/morph_gpu_v2.cu`

v2 keeps the same opaque tensor registry and adds:

- persistent thread-local cuBLAS context rather than creating one per matmul
- strict FP32 and fast TF32 matrix multiplication paths
- elementwise multiply
- transpose
- causal masking
- row softmax
- RMSNorm, including learned per-feature weights
- SiLU
- GELU
- interleaved RoPE
- embedding-row gather
- composed single-head attention (`QKᵀ → scale → optional causal mask → softmax → V`)
- live tensor count and VRAM accounting
- bulk release

### v3 — Perseus / Mnemosyne vector-memory primitives

`runtime/gpu/morph_gpu_v3.cu`

v3 includes v2 into the same CUDA translation unit, so all operations share one handle registry and ownership model. It adds primitives selected from the real Perseus memory workload:

- sum across matrix rows
- row-wise L2 normalization
- exact cosine scoring of one query against all matrix rows
- fused selected-substrate-row mean + L2 normalization
- exact device-side small-k top-k retrieval
- persistent GPU memory worker that keeps the normalized memory matrix resident in VRAM

v3 is the default runtime selected by `gpu_build.js`.

## Public Morpheus API

The stable language surface lives in `stdlib/gpu.morph`.

Representative calls:

```morpheus
import "../stdlib/gpu.morph"

let a = gpu_tensor_from_csv("1,2,3,4,5,6", 2, 3)
let b = gpu_tensor_from_csv("7,8,9,10,11,12", 3, 2)
let c = gpu_matmul(a, b)
print(gpu_tensor_to_csv(c))

gpu_tensor_free(a)
gpu_tensor_free(b)
gpu_tensor_free(c)
```

Transformer-style execution:

```morpheus
let q = gpu_tensor_from_csv("1,0,0,1", 2, 2)
let k = gpu_tensor_from_csv("1,0,0,1", 2, 2)
let v = gpu_tensor_from_csv("10,20,30,40", 2, 2)
let out = gpu_attention(q, k, v, 0.70710678, 1)
```

Perseus/Mnemosyne-style retrieval:

```morpheus
let query = gpu_embedding_mean_l2(substrate, token_ids)
let scores = gpu_cosine_scores(memory_matrix, query)
let top = gpu_topk_indices(scores, 8)
```

## Persistent memory worker

`runtime/gpu/memory_worker.cu` is a long-lived native process intended for hosts such as Bun/Electron where spawning a CUDA program for every request would erase most of the acceleration benefit.

Protocol:

```text
INIT <f32_path> <rows> <dim>
QUERY <k> <comma-separated query floats>
INFO
EXIT
```

The worker:

1. loads the memory embedding matrix once,
2. normalizes it on GPU,
3. keeps it resident in VRAM,
4. receives only small query vectors,
5. performs cosine scoring and top-k on-device,
6. returns only row indices and scores.

This is the path used by the optional Mnemosyne integration in `Aporia-Nous/perseus-current`.

## Build

Requirements for CUDA execution:

- NVIDIA CUDA-capable GPU
- NVIDIA driver
- CUDA Toolkit (`nvcc`, cuBLAS)
- CUDA-supported host C++ compiler

General Morpheus program:

```bash
node gpu_build.js examples/gpu_compute.morph --arch sm_89
```

Transformer primitives:

```bash
node gpu_build.js examples/gpu_ai_primitives.morph --arch sm_89
```

Perseus-style memory workload:

```bash
node gpu_build.js examples/gpu_memory_search.morph --arch sm_89
```

Persistent worker:

```bash
node gpu_worker_build.js --arch sm_89 -o morpheus-gpu-memory-worker.exe
```

For the RTX 4060 Ti, compute capability 8.9 corresponds to `sm_89`.

If `nvcc` is not on PATH, set `CUDA_PATH` or `MORPHEUS_NVCC`.

## Validation layers

### 1. Model-free CI

`gpu_test.js` verifies:

- Morpheus → C++ FFI generation
- no raw GPU pointer exposure at the language boundary
- v1/v2/v3 CUDA source wiring
- AI primitive presence
- vector-memory primitive presence
- default v3 selection
- native build command construction
- compatibility with the existing Morpheus conformance suite

GitHub CI also parses the worker/benchmark tooling and dry-runs all native CUDA build commands.

This proves compiler/API integration. It does **not** prove CUDA hardware execution.

### 2. Hardware gate

On a Windows NVIDIA machine:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validate_gpu_windows.ps1
```

The hardware gate refuses to pass unless it can:

- discover the GPU and CUDA toolkit
- build real CUDA binaries
- verify exact matrix multiplication output
- execute softmax, RMSNorm, SiLU, GELU, RoPE, embedding gather, and causal attention
- execute the Perseus-style vector-memory path
- build the persistent worker
- compare persistent-worker retrieval with a CPU baseline at 1K and 10K memory rows
- verify exact top-k agreement and score tolerance

### 3. Scaling benchmark

For larger measurements:

```bash
node gpu_memory_benchmark.js \
  --worker ./morpheus-gpu-memory-worker.exe \
  --sizes 1000,10000,100000 \
  --dim 64 --k 8 --iterations 20
```

The benchmark reports initialization cost, CPU average latency, GPU average and p95 latency, speedup, and maximum score delta. It aborts rather than reporting a speedup if CPU/GPU top-k results diverge.

## Why not put everything on the GPU?

GPU acceleration should follow arithmetic intensity, not ideology. Mnemosyne's tokenization and vocabulary lookup remain cheap CPU work; a 64-dimensional query vector is tiny. The expensive operation as memory grows is scanning every stored embedding. v3 therefore accelerates the full-table scoring/retrieval path and keeps the matrix resident, while small stores remain on the CPU by default.

The Perseus integration uses a configurable threshold (`MORPHEUS_GPU_MIN_ROWS`, default 256) and falls back to CPU on missing worker, timeout, crash, or CUDA failure.

## Current precision boundary

What is established now:

- Morpheus has a concrete native CUDA substrate in source.
- The compiler/FFI/API path is covered by green model-free CI.
- The runtime includes general tensor, transformer-oriented, and Perseus-specific memory primitives.
- Perseus/Mnemosyne has an optional persistent-worker integration branch with CPU fallback.

What must not be claimed until the hardware gate runs successfully:

- actual CUDA compilation on the target 4060 Ti
- measured GPU latency/speedup
- production stability under sustained CUDA load

## Next deepening targets

The next primitives should be selected from measured Perseus workloads. Likely candidates include:

- symmetric windowed co-occurrence (matching the historical `gpu_cooccur` path)
- batched cosine/query scoring
- segmented reductions
- fused projection + normalization
- quantized GEMM and weight formats
- KV-cache storage/layout
- batched/multi-head attention
- stream/event scheduling and asynchronous transfer
- pinned host memory
- CUDA graph capture for repeated cognition kernels
- backend abstraction above CUDA for future non-NVIDIA execution

The goal is not to recreate llama.cpp for its own sake. The goal is to make Morpheus a sovereign compute runtime that can host the primitives Perseus actually needs.
