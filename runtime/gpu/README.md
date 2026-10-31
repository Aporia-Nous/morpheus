# Morpheus GPU Runtime

This directory is reserved for the optional native GPU compute layer for Morpheus.

The implementation is being developed on `feature/gpu-runtime-v0.1`. The design keeps CPU-only Morpheus programs unchanged while allowing native builds to opt into GPU execution. GPU resources use opaque numeric handles at the Morpheus language boundary rather than exposing raw device pointers.

Planned stable surface:
- device discovery
- tensor allocation/free
- host ↔ device transfer
- synchronization
- vector primitives
- matrix multiplication
- backend-independent tensor metadata

The initial backend targets NVIDIA CUDA. The API is intentionally backend-neutral so HIP, Metal, Vulkan, or CPU-SIMD implementations can be added later without changing Morpheus source programs.
