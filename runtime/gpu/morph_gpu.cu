// Morpheus optional CUDA runtime v0.1
// Dense row-major float32 tensors behind opaque integer handles.
// Build/link with nvcc and cuBLAS. This file is intentionally independent of
// morph_rt.h so CPU-only Morpheus remains zero-dependency.

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Tensor {
  float* data = nullptr;
  long long rows = 0;
  long long cols = 0;
  size_t count = 0;

  ~Tensor() {
    if (data) cudaFree(data);
  }
};

std::unordered_map<long long, std::shared_ptr<Tensor>>& registry() {
  static std::unordered_map<long long, std::shared_ptr<Tensor>> r;
  return r;
}

std::mutex& registry_mutex() {
  static std::mutex m;
  return m;
}

std::atomic<long long>& next_handle() {
  static std::atomic<long long> h{1};
  return h;
}

thread_local std::string last_error;
thread_local std::string string_result;

void clear_error() { last_error.clear(); }

void set_error(const std::string& message) { last_error = message; }

bool cuda_ok(cudaError_t code, const char* where) {
  if (code == cudaSuccess) return true;
  set_error(std::string(where) + ": " + cudaGetErrorString(code));
  return false;
}

bool cublas_ok(cublasStatus_t code, const char* where) {
  if (code == CUBLAS_STATUS_SUCCESS) return true;
  std::ostringstream ss;
  ss << where << ": cuBLAS status " << static_cast<int>(code);
  set_error(ss.str());
  return false;
}

std::shared_ptr<Tensor> get_tensor(long long handle) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto it = registry().find(handle);
  if (it == registry().end()) {
    set_error("invalid GPU tensor handle: " + std::to_string(handle));
    return nullptr;
  }
  return it->second;
}

long long register_tensor(std::shared_ptr<Tensor> tensor) {
  const long long handle = next_handle().fetch_add(1);
  std::lock_guard<std::mutex> lock(registry_mutex());
  registry()[handle] = std::move(tensor);
  return handle;
}

long long create_tensor(long long rows, long long cols, bool zero_fill = true) {
  clear_error();
  if (rows <= 0 || cols <= 0) {
    set_error("tensor dimensions must be positive");
    return 0;
  }

  const unsigned long long count64 = static_cast<unsigned long long>(rows) * static_cast<unsigned long long>(cols);
  if (count64 > static_cast<unsigned long long>(SIZE_MAX / sizeof(float))) {
    set_error("tensor allocation size overflow");
    return 0;
  }

  auto tensor = std::make_shared<Tensor>();
  tensor->rows = rows;
  tensor->cols = cols;
  tensor->count = static_cast<size_t>(count64);
  const size_t bytes = tensor->count * sizeof(float);

  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&tensor->data), bytes), "cudaMalloc")) return 0;
  if (zero_fill && !cuda_ok(cudaMemset(tensor->data, 0, bytes), "cudaMemset")) return 0;

  return register_tensor(std::move(tensor));
}

std::vector<float> parse_csv(const char* text, size_t expected) {
  std::vector<float> values;
  values.reserve(expected);
  const char* p = text ? text : "";

  while (*p) {
    while (*p && (*p == ',' || *p == ';' || *p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    if (!*p) break;
    char* end = nullptr;
    errno = 0;
    const float value = std::strtof(p, &end);
    if (end == p || errno == ERANGE) {
      values.clear();
      return values;
    }
    values.push_back(value);
    p = end;
  }

  if (values.size() != expected) values.clear();
  return values;
}

__global__ void add_kernel(const float* a, const float* b, float* out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

__global__ void scale_kernel(const float* a, float scalar, float* out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * scalar;
}

__global__ void relu_kernel(const float* a, float* out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] > 0.0f ? a[i] : 0.0f;
}

long long unary_output_like(const std::shared_ptr<Tensor>& src) {
  return create_tensor(src->rows, src->cols, false);
}

} // namespace

extern "C" long long gpu_available() {
  clear_error();
  int count = 0;
  const cudaError_t code = cudaGetDeviceCount(&count);
  if (code != cudaSuccess) {
    cudaGetLastError();
    set_error(std::string("cudaGetDeviceCount: ") + cudaGetErrorString(code));
    return 0;
  }
  return count > 0 ? 1 : 0;
}

extern "C" long long gpu_device_count() {
  clear_error();
  int count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&count), "cudaGetDeviceCount")) return 0;
  return static_cast<long long>(count);
}

extern "C" const char* gpu_device_name(long long index) {
  clear_error();
  cudaDeviceProp props{};
  if (!cuda_ok(cudaGetDeviceProperties(&props, static_cast<int>(index)), "cudaGetDeviceProperties")) {
    string_result.clear();
    return string_result.c_str();
  }
  string_result = props.name;
  return string_result.c_str();
}

extern "C" const char* gpu_last_error() {
  return last_error.c_str();
}

extern "C" double gpu_memory_free_mb() {
  clear_error();
  size_t free_bytes = 0, total_bytes = 0;
  if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo")) return 0.0;
  return static_cast<double>(free_bytes) / (1024.0 * 1024.0);
}

extern "C" double gpu_memory_total_mb() {
  clear_error();
  size_t free_bytes = 0, total_bytes = 0;
  if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo")) return 0.0;
  return static_cast<double>(total_bytes) / (1024.0 * 1024.0);
}

extern "C" long long gpu_sync() {
  clear_error();
  return cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize") ? 1 : 0;
}

extern "C" long long gpu_tensor_create(long long rows, long long cols) {
  return create_tensor(rows, cols, true);
}

extern "C" long long gpu_tensor_from_csv(const char* values, long long rows, long long cols) {
  const long long handle = create_tensor(rows, cols, false);
  if (!handle) return 0;
  auto tensor = get_tensor(handle);
  if (!tensor) return 0;

  const std::vector<float> host = parse_csv(values, tensor->count);
  if (host.empty() && tensor->count != 0) {
    set_error("gpu_tensor_from_csv expected exactly " + std::to_string(tensor->count) + " float values");
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().erase(handle);
    return 0;
  }

  if (!cuda_ok(cudaMemcpy(tensor->data, host.data(), tensor->count * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D")) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().erase(handle);
    return 0;
  }
  return handle;
}

extern "C" long long gpu_tensor_load_f32(const char* path, long long rows, long long cols) {
  const long long handle = create_tensor(rows, cols, false);
  if (!handle) return 0;
  auto tensor = get_tensor(handle);
  if (!tensor) return 0;

  std::ifstream file(path, std::ios::binary);
  if (!file) {
    set_error(std::string("cannot open float32 tensor file: ") + (path ? path : ""));
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().erase(handle);
    return 0;
  }

  std::vector<float> host(tensor->count);
  file.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(host.size() * sizeof(float)));
  if (file.gcount() != static_cast<std::streamsize>(host.size() * sizeof(float))) {
    set_error("float32 tensor file size does not match requested shape");
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().erase(handle);
    return 0;
  }

  char extra = 0;
  if (file.read(&extra, 1)) {
    set_error("float32 tensor file contains more data than requested shape");
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().erase(handle);
    return 0;
  }

  if (!cuda_ok(cudaMemcpy(tensor->data, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D")) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().erase(handle);
    return 0;
  }
  return handle;
}

extern "C" long long gpu_tensor_save_f32(long long handle, const char* path) {
  clear_error();
  auto tensor = get_tensor(handle);
  if (!tensor) return 0;
  std::vector<float> host(tensor->count);
  if (!cuda_ok(cudaMemcpy(host.data(), tensor->data, host.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H")) return 0;

  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) {
    set_error(std::string("cannot write float32 tensor file: ") + (path ? path : ""));
    return 0;
  }
  file.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(host.size() * sizeof(float)));
  if (!file) {
    set_error("failed writing float32 tensor file");
    return 0;
  }
  return 1;
}

extern "C" long long gpu_tensor_free(long long handle) {
  clear_error();
  std::lock_guard<std::mutex> lock(registry_mutex());
  return registry().erase(handle) ? 1 : 0;
}

extern "C" long long gpu_tensor_rows(long long handle) {
  clear_error();
  auto tensor = get_tensor(handle);
  return tensor ? tensor->rows : 0;
}

extern "C" long long gpu_tensor_cols(long long handle) {
  clear_error();
  auto tensor = get_tensor(handle);
  return tensor ? tensor->cols : 0;
}

extern "C" long long gpu_tensor_count(long long handle) {
  clear_error();
  auto tensor = get_tensor(handle);
  return tensor ? static_cast<long long>(tensor->count) : 0;
}

extern "C" double gpu_tensor_get(long long handle, long long index) {
  clear_error();
  auto tensor = get_tensor(handle);
  if (!tensor) return 0.0;
  if (index < 0 || static_cast<size_t>(index) >= tensor->count) {
    set_error("gpu_tensor_get index out of range");
    return 0.0;
  }
  float value = 0.0f;
  if (!cuda_ok(cudaMemcpy(&value, tensor->data + index, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H scalar")) return 0.0;
  return static_cast<double>(value);
}

extern "C" long long gpu_tensor_set(long long handle, long long index, double value) {
  clear_error();
  auto tensor = get_tensor(handle);
  if (!tensor) return 0;
  if (index < 0 || static_cast<size_t>(index) >= tensor->count) {
    set_error("gpu_tensor_set index out of range");
    return 0;
  }
  const float v = static_cast<float>(value);
  return cuda_ok(cudaMemcpy(tensor->data + index, &v, sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D scalar") ? 1 : 0;
}

extern "C" const char* gpu_tensor_to_csv(long long handle) {
  clear_error();
  auto tensor = get_tensor(handle);
  if (!tensor) {
    string_result.clear();
    return string_result.c_str();
  }
  std::vector<float> host(tensor->count);
  if (!cuda_ok(cudaMemcpy(host.data(), tensor->data, host.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H")) {
    string_result.clear();
    return string_result.c_str();
  }

  std::ostringstream out;
  out << std::setprecision(9);
  for (size_t i = 0; i < host.size(); ++i) {
    if (i) out << ',';
    out << host[i];
  }
  string_result = out.str();
  return string_result.c_str();
}

extern "C" long long gpu_matmul(long long a_handle, long long b_handle) {
  clear_error();
  auto a = get_tensor(a_handle);
  auto b = get_tensor(b_handle);
  if (!a || !b) return 0;
  if (a->cols != b->rows) {
    set_error("gpu_matmul shape mismatch: A.cols must equal B.rows");
    return 0;
  }

  const long long out_handle = create_tensor(a->rows, b->cols, false);
  if (!out_handle) return 0;
  auto out = get_tensor(out_handle);
  if (!out) return 0;

  cublasHandle_t handle = nullptr;
  if (!cublas_ok(cublasCreate(&handle), "cublasCreate")) {
    gpu_tensor_free(out_handle);
    return 0;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Morpheus stores matrices row-major. cuBLAS is column-major, so compute
  // C^T = B^T * A^T by swapping operands/dimensions.
  const cublasStatus_t status = cublasSgemm(
    handle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    static_cast<int>(b->cols),
    static_cast<int>(a->rows),
    static_cast<int>(a->cols),
    &alpha,
    b->data,
    static_cast<int>(b->cols),
    a->data,
    static_cast<int>(a->cols),
    &beta,
    out->data,
    static_cast<int>(b->cols)
  );

  const bool ok = cublas_ok(status, "cublasSgemm");
  cublasDestroy(handle);
  if (!ok) {
    gpu_tensor_free(out_handle);
    return 0;
  }
  return out_handle;
}

extern "C" long long gpu_add(long long a_handle, long long b_handle) {
  clear_error();
  auto a = get_tensor(a_handle);
  auto b = get_tensor(b_handle);
  if (!a || !b) return 0;
  if (a->rows != b->rows || a->cols != b->cols) {
    set_error("gpu_add shape mismatch");
    return 0;
  }

  const long long out_handle = create_tensor(a->rows, a->cols, false);
  if (!out_handle) return 0;
  auto out = get_tensor(out_handle);
  const int threads = 256;
  const int blocks = static_cast<int>((a->count + threads - 1) / threads);
  add_kernel<<<blocks, threads>>>(a->data, b->data, out->data, a->count);
  if (!cuda_ok(cudaGetLastError(), "gpu_add kernel launch")) {
    gpu_tensor_free(out_handle);
    return 0;
  }
  return out_handle;
}

extern "C" long long gpu_scale(long long a_handle, double scalar) {
  clear_error();
  auto a = get_tensor(a_handle);
  if (!a) return 0;
  const long long out_handle = unary_output_like(a);
  if (!out_handle) return 0;
  auto out = get_tensor(out_handle);
  const int threads = 256;
  const int blocks = static_cast<int>((a->count + threads - 1) / threads);
  scale_kernel<<<blocks, threads>>>(a->data, static_cast<float>(scalar), out->data, a->count);
  if (!cuda_ok(cudaGetLastError(), "gpu_scale kernel launch")) {
    gpu_tensor_free(out_handle);
    return 0;
  }
  return out_handle;
}

extern "C" long long gpu_relu(long long a_handle) {
  clear_error();
  auto a = get_tensor(a_handle);
  if (!a) return 0;
  const long long out_handle = unary_output_like(a);
  if (!out_handle) return 0;
  auto out = get_tensor(out_handle);
  const int threads = 256;
  const int blocks = static_cast<int>((a->count + threads - 1) / threads);
  relu_kernel<<<blocks, threads>>>(a->data, out->data, a->count);
  if (!cuda_ok(cudaGetLastError(), "gpu_relu kernel launch")) {
    gpu_tensor_free(out_handle);
    return 0;
  }
  return out_handle;
}
