// Morpheus CUDA runtime v0.4
// v3 compute + vector-memory substrate with a stable native extension ABI.

#include "morph_gpu_v3.cu"
#include "morph_gpu_extension.h"

extern "C" int morph_gpu_tensor_view_f32(
    long long handle,
    float** device_ptr,
    long long* rows,
    long long* cols,
    size_t* count) {
  clear_error();
  if (!device_ptr || !rows || !cols || !count) {
    set_error("morph_gpu_tensor_view_f32 received null output pointer");
    return 0;
  }
  auto tensor = get_tensor(handle);
  if (!tensor) return 0;
  *device_ptr = tensor->data;
  *rows = tensor->rows;
  *cols = tensor->cols;
  *count = tensor->count;
  return 1;
}

extern "C" long long morph_gpu_tensor_alloc_f32(
    long long rows,
    long long cols,
    float** device_ptr) {
  clear_error();
  if (!device_ptr) {
    set_error("morph_gpu_tensor_alloc_f32 received null device_ptr output");
    return 0;
  }
  long long handle = create_tensor(rows, cols, false);
  if (!handle) return 0;
  auto tensor = get_tensor(handle);
  if (!tensor) {
    release_handle(handle);
    return 0;
  }
  *device_ptr = tensor->data;
  return handle;
}

extern "C" int morph_gpu_tensor_release_native(long long handle) {
  clear_error();
  return release_handle(handle) ? 1 : 0;
}

extern "C" void morph_gpu_set_extension_error(const char* message) {
  set_error(message ? message : "native GPU extension error");
}
