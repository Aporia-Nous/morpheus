#pragma once

// Morpheus native GPU extension ABI.
//
// These functions are for native CUDA/C++ extension modules only. They are NOT
// part of stdlib/gpu.morph and do not expose device pointers to Morpheus source.
// Extensions can borrow a device view for the duration of a call or allocate a
// new registered tensor that is returned to Morpheus as an opaque handle.

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Borrow the device pointer/shape behind an existing opaque tensor handle.
// Ownership remains with the Morpheus runtime. The pointer must not be freed or
// retained after the handle is released. Returns 1 on success, 0 on failure.
int morph_gpu_tensor_view_f32(
    long long handle,
    float** device_ptr,
    long long* rows,
    long long* cols,
    size_t* count);

// Allocate a registered float32 tensor and return both its opaque handle and a
// borrowed device pointer for the extension to fill. Returns 0 on error.
long long morph_gpu_tensor_alloc_f32(
    long long rows,
    long long cols,
    float** device_ptr);

// Release a tensor through the same registry used by the Morpheus language API.
int morph_gpu_tensor_release_native(long long handle);

// Report an extension error into the thread-local Morpheus GPU error channel so
// gpu_last_error() can surface the failure to Morpheus code.
void morph_gpu_set_extension_error(const char* message);

#ifdef __cplusplus
}
#endif
