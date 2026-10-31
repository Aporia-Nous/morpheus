# Morpheus GPU Runtime v0.4

Morpheus now has an optional native NVIDIA CUDA compute substrate for general numerical work, AI primitives, and Perseus-class cognition/memory workloads. GPU support is additive: the reference interpreter, ordinary `g++` native builds, and CPU-only programs keep their existing behavior.

> Morpheus owns the compute contract; CUDA is the first backend.

Morpheus source sees opaque numeric tensor handles. Raw device pointers never cross the language boundary. That keeps ownership, lifetime tracking, and error handling inside the runtime while leaving room for future HIP, Metal, Vulkan, or CPU-SIMD backends.

## Runtime generations

### v1 — foundational tensors

`runtime/gpu/morph_gpu.cu`

- CUDA device discovery and VRAM reporting
- dense row-major float32 tensors
- allocation/free, synchronization, scalar diagnostics
- CSV bootstrap and binary float32 load/save
- cuBLAS SGEMM
- add, scale, ReLU

### v2 — AI / transformer primitives

`runtime/gpu/morph_gpu_v2.cu`

- persistent thread-local cuBLAS context
- strict FP32 and fast TF32 matmul
- multiply, transpose, causal masking
- row softmax
- RMSNorm and weighted RMSNorm
- SiLU and GELU
- interleaved RoPE
- embedding-row gather
- composed attention (`QKᵀ → scale → optional causal mask → softmax → V`)
- tensor-count / VRAM accounting and bulk release

### v3 — Perseus / Mnemosyne vector memory

`runtime/gpu/morph_gpu_v3.cu`

v3 extends the same tensor registry with primitives chosen from the real Mnemosyne workload:

- row reduction
- row-wise L2 normalization
- exact cosine scoring of a query against all matrix rows
- fused selected-substrate-row mean + L2 embedding
- exact small-k device retrieval
- persistent vector-memory worker that keeps the normalized memory matrix in VRAM

### v4 — native extension ABI

`runtime/gpu/morph_gpu_v4.cu`
`runtime/gpu/morph_gpu_extension.h`

v4 is the default runtime. It preserves all v3 behavior and adds a native-only extension boundary for specialized CUDA kernels.

Extensions can:

- borrow the device pointer/shape behind an existing registered tensor
- allocate a new registered output tensor
- release a registered tensor
- report failures through `gpu_last_error()`

The native ABI is deliberately **not** exposed through `stdlib/gpu.morph`. Morpheus programs still see only safe handles.

The ABI functions are:

```cpp
morph_gpu_tensor_view_f32(...)
morph_gpu_tensor_alloc_f32(...)
morph_gpu_tensor_release_native(...)
morph_gpu_set_extension_error(...)
```

This is the mechanism used for private/system-specific kernels such as Perseus WTR batch dynamics without hard-wiring Perseus logic into the Morpheus core runtime.

## Extension example

`examples/extensions/gpu_affine_extension.cu` implements a custom CUDA affine kernel using the v4 ABI. `examples/gpu_extension_demo.morph` declares it with ordinary Morpheus `extern` syntax:

```morpheus
extern gpu_affine(input: int, scale: float, bias: float) -> int
```

Build it with:

```bash
node gpu_build.js examples/gpu_extension_demo.morph \
  --runtime v4 \
  --arch sm_89 \
  --extension examples/extensions/gpu_affine_extension.cu
```

The hardware gate requires this path to return exactly `3,5,7,9` for the test vector.

## Public Morpheus GPU API

The stable language surface lives in `stdlib/gpu.morph` and includes:

- tensor allocation, persistence, metadata, diagnostics
- strict and fast matrix multiplication
- elementwise arithmetic and activation functions
- normalization, softmax, RoPE, gather, attention
- cosine scoring and exact small-k retrieval
- live tensor/VRAM accounting

Representative memory path:

```morpheus
let query = gpu_embedding_mean_l2(substrate, token_ids)
let scores = gpu_cosine_scores(memory_matrix, query)
let top = gpu_topk_indices(scores, 8)
```

## Persistent Mnemosyne worker

`runtime/gpu/memory_worker.cu` keeps a normalized memory matrix resident on the device.

Protocol:

```text
INIT <f32_path> <rows> <dim>
QUERY <k> <comma-separated query floats>
INFO
EXIT
```

This avoids re-uploading the full memory matrix on every recall. The optional Perseus integration uses a configurable offload threshold and automatically falls back to its existing CPU cosine path if the worker is absent, crashes, times out, or CUDA fails.

## Build

Requirements for real CUDA execution:

- NVIDIA CUDA-capable GPU
- NVIDIA driver
- CUDA Toolkit (`nvcc`, cuBLAS)
- CUDA-supported host C++ compiler

For the RTX 4060 Ti, compute capability 8.9 corresponds to `sm_89`.

Examples:

```bash
node gpu_build.js examples/gpu_compute.morph --arch sm_89
node gpu_build.js examples/gpu_ai_primitives.morph --arch sm_89
node gpu_build.js examples/gpu_memory_search.morph --arch sm_89
node gpu_worker_build.js --arch sm_89 -o morpheus-gpu-memory-worker.exe
```

`gpu_build.js` defaults to v4. Older v1/v2/v3 runtimes remain selectable for regression work.

## Validation layers

### 1. Model-free CI

`gpu_test.js` plus GitHub Actions verify:

- existing Morpheus tests and conformance
- Morpheus → native FFI generation
- v1/v2/v3/v4 source/API wiring
- no raw device-pointer exposure to Morpheus source
- transformer and vector-memory primitives
- v4 native extension ABI
- generic extension build composition
- build/worker/benchmark tooling syntax
- CUDA build-command dry runs

This proves compiler/API integration only. It does **not** prove CUDA execution on hardware.

Current v4 model-free CI is green.

### 2. Windows hardware gate

Run on the target NVIDIA host:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validate_gpu_windows.ps1
```

The gate refuses to pass unless it can:

- discover NVIDIA/CUDA and build real binaries
- verify exact matrix multiplication output
- execute softmax, RMSNorm, SiLU, GELU, RoPE, gather, and causal attention
- execute the Perseus-style vector-memory workload
- compile and execute an external v4 CUDA extension through the native ABI
- build the persistent memory worker
- compare GPU recall with an independent CPU baseline at 1K and 10K rows
- preserve exact top-k and score tolerance

### 3. Memory scaling benchmark

```bash
node gpu_memory_benchmark.js \
  --worker ./morpheus-gpu-memory-worker.exe \
  --sizes 1000,10000,100000 \
  --dim 64 --k 8 --iterations 20
```

The benchmark reports initialization cost, CPU average, GPU average/p95, speedup, and score delta. It aborts rather than report a speedup if CPU/GPU top-k diverges.

## Perseus WTR extension path

The companion private Perseus branch uses the v4 extension ABI for batched WTR cognitive-field dynamics. This is intentionally kept outside Morpheus core because it is a Perseus workload, not a general language primitive.

The current design supports:

- thousands of independent WTR candidate trajectories in parallel
- file-backed CPU/GPU parity validation
- a resident `N×13` state format that can stay in VRAM across repeated advance cycles
- persistent WTR worker commands for repeated `ADVANCE`, ranking, and save

This is useful for numerical parameter/candidate exploration and for testing the accelerated-internal-computation research direction. It must **not** be described as literal time dilation, subjective time, or proof that a given number of external seconds equals a year of experience. The measurable quantity is computational candidate-steps and the quality of learning/selection achieved from them.

## Why not put everything on the GPU?

GPU acceleration should follow measured arithmetic intensity. Cheap tokenization, vocabulary lookup, small state transitions, and low-cardinality logic remain better on the CPU. Full-table memory scans, large candidate batches, matrix operations, and repeated resident numerical dynamics are the current GPU targets.

## Claim boundary

Established:

- concrete CUDA source/runtime/compiler substrate exists
- v4 extension-capable model-free CI is green
- general tensor, transformer, vector-memory, persistent-worker, and native-extension paths are implemented
- Perseus has an optional Mnemosyne integration with CPU fallback and a separate WTR extension path

Not established until the target hardware gate is run successfully:

- actual CUDA compilation/execution on the RTX 4060 Ti
- measured GPU speedup
- sustained-load production stability
- any accelerated-experience claim beyond measured numerical throughput and downstream task performance

## Next deepening targets

Choose these from measured Perseus workloads rather than from a generic LLM checklist:

- symmetric windowed co-occurrence matching the historical `gpu_cooccur` semantics
- batched query scoring and segmented reductions
- pinned host memory and async transfer
- CUDA streams/events
- CUDA graph capture for repeated cognition kernels
- quantized GEMM / weight formats where they materially help Perseus
- KV-cache layout and multi-head attention only if Perseus actually needs a transformer-style generator
- backend abstraction for non-NVIDIA devices

The goal is not to clone llama.cpp. The goal is a sovereign Morpheus compute runtime that hosts the primitives Perseus actually uses.
