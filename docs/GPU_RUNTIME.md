# Morpheus GPU Runtime v0.1

Morpheus now has an optional native NVIDIA CUDA compute backend. It is additive: the reference interpreter, ordinary `g++` native builds, and CPU-only programs keep their existing zero-GPU dependency behavior.

## What exists in v0.1

The GPU boundary is intentionally small and backend-neutral. Morpheus programs see opaque integer tensor handles; raw CUDA pointers never enter the language runtime.

- CUDA device discovery and device name
- free/total VRAM reporting
- explicit synchronization
- dense row-major float32 tensor allocation/free
- CSV bootstrap upload/download for small tensors and debugging
- direct binary float32 tensor load/save for large data paths
- scalar indexed get/set for diagnostics
- cuBLAS SGEMM matrix multiplication
- CUDA vector add
- CUDA scale
- CUDA ReLU
- thread-local error reporting
- handle registry with automatic `cudaFree` when a handle is released

The public Morpheus interface is in `stdlib/gpu.morph`. The CUDA implementation is `runtime/gpu/morph_gpu.cu`. The native build driver is `gpu_build.js`.

## Build

Requirements:

- NVIDIA GPU with a CUDA-capable driver
- NVIDIA CUDA Toolkit (provides `nvcc` and cuBLAS)
- a host C++ compiler supported by your CUDA Toolkit

```bash
node gpu_build.js examples/gpu_compute.morph
```

For an RTX 4060 Ti (Ada, compute capability 8.9):

```bash
node gpu_build.js examples/gpu_compute.morph --arch sm_89
```

If `nvcc` is not on PATH, set either `CUDA_PATH` or `MORPHEUS_NVCC`.

```powershell
$env:MORPHEUS_NVCC = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe"
node gpu_build.js examples/gpu_compute.morph --arch sm_89
```

Use `--dry-run` to validate source-to-native compilation and inspect the `nvcc` command without requiring CUDA on the machine running the check.

## Language API

```morpheus
import "../stdlib/gpu.morph"

let a = gpu_tensor_from_csv("1,2,3,4,5,6", 2, 3)
let b = gpu_tensor_from_csv("7,8,9,10,11,12", 3, 2)
let c = gpu_matmul(a, b)
print(gpu_tensor_to_csv(c))   # 58,64,139,154

gpu_tensor_free(a)
gpu_tensor_free(b)
gpu_tensor_free(c)
```

For larger data, prefer `gpu_tensor_load_f32()` and `gpu_tensor_save_f32()` so tensor payloads do not pass through Morpheus strings.

## Why handles instead of pointers

A handle boundary gives Morpheus ownership and safety controls while keeping the API portable. CUDA is the first backend, not the language contract. A future HIP, Metal, Vulkan, or CPU-SIMD backend can implement the same tensor-handle operations without forcing Perseus or other Morpheus programs to change source code.

## Perseus path

This v0.1 layer gives Morpheus-native programs direct access to GPU matrix/vector compute, but it does **not** by itself move Perseus onto the GPU. Perseus must call this interface (or a higher-level tensor layer built on it) before its cognition is GPU-accelerated.

The next AI-oriented primitives should be implemented above this base: softmax, RMSNorm, RoPE, embedding lookup, quantized matrix kernels, KV-cache storage, and batched attention. Those can then support Perseus workloads and, if desired later, a sovereign model runtime without changing the lower GPU ownership API.
