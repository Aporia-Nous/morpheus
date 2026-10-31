#include "morph_gpu_extension.h"
#include <cuda_runtime.h>
#include <string>

__global__ void affine_kernel(const float* input, float* output, size_t count, float scale, float bias) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) output[i] = input[i] * scale + bias;
}

extern "C" long long gpu_affine(long long input_handle, double scale, double bias) {
  float* input = nullptr;
  long long rows = 0, cols = 0;
  size_t count = 0;
  if (!morph_gpu_tensor_view_f32(input_handle, &input, &rows, &cols, &count)) return 0;

  float* output = nullptr;
  long long output_handle = morph_gpu_tensor_alloc_f32(rows, cols, &output);
  if (!output_handle) return 0;

  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  affine_kernel<<<blocks, threads>>>(input, output, count, static_cast<float>(scale), static_cast<float>(bias));
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::string message = std::string("gpu_affine kernel: ") + cudaGetErrorString(e);
    morph_gpu_set_extension_error(message.c_str());
    morph_gpu_tensor_release_native(output_handle);
    return 0;
  }
  return output_handle;
}
