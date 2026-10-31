// Morpheus optional CUDA runtime v0.2
// General-purpose dense float32 tensor substrate + AI-oriented primitives.
// Morpheus sees only opaque integer handles; raw device pointers remain native.

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <atomic>
#include <cerrno>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
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
  ~Tensor() { if (data) cudaFree(data); }
};

std::unordered_map<long long, std::shared_ptr<Tensor>>& registry() {
  static std::unordered_map<long long, std::shared_ptr<Tensor>> r;
  return r;
}
std::mutex& registry_mutex() { static std::mutex m; return m; }
std::atomic<long long>& next_handle() { static std::atomic<long long> h{1}; return h; }

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

struct CublasContext {
  cublasHandle_t handle = nullptr;
  CublasContext() { cublasCreate(&handle); }
  ~CublasContext() { if (handle) cublasDestroy(handle); }
};

cublasHandle_t cublas_handle() {
  thread_local CublasContext ctx;
  if (!ctx.handle) set_error("failed to create thread-local cuBLAS handle");
  return ctx.handle;
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

bool release_handle(long long handle) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  return registry().erase(handle) != 0;
}

long long create_tensor(long long rows, long long cols, bool zero_fill = true) {
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

long long unary_output_like(const std::shared_ptr<Tensor>& src) {
  return create_tensor(src->rows, src->cols, false);
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
    if (end == p || errno == ERANGE) { values.clear(); return values; }
    values.push_back(value);
    p = end;
  }
  if (values.size() != expected) values.clear();
  return values;
}

bool check_same_shape(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b, const char* op) {
  if (a->rows == b->rows && a->cols == b->cols) return true;
  set_error(std::string(op) + " shape mismatch");
  return false;
}

bool check_cublas_dims(long long m, long long n, long long k) {
  const long long maxI = static_cast<long long>(std::numeric_limits<int>::max());
  if (m <= maxI && n <= maxI && k <= maxI) return true;
  set_error("tensor dimensions exceed cuBLAS int32 limits");
  return false;
}

bool row_major_sgemm(const std::shared_ptr<Tensor>& a,
                     const std::shared_ptr<Tensor>& b,
                     const std::shared_ptr<Tensor>& out,
                     bool fast_tf32) {
  if (!check_cublas_dims(a->rows, b->cols, a->cols)) return false;
  cublasHandle_t handle = cublas_handle();
  if (!handle) return false;
  if (fast_tf32) cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
  else cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

  const float alpha = 1.0f, beta = 0.0f;
  const cublasStatus_t status = cublasSgemm(
    handle, CUBLAS_OP_N, CUBLAS_OP_N,
    static_cast<int>(b->cols), static_cast<int>(a->rows), static_cast<int>(a->cols),
    &alpha,
    b->data, static_cast<int>(b->cols),
    a->data, static_cast<int>(a->cols),
    &beta,
    out->data, static_cast<int>(b->cols)
  );
  return cublas_ok(status, fast_tf32 ? "cublasSgemm TF32" : "cublasSgemm");
}

__global__ void add_kernel(const float* a, const float* b, float* out, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
__global__ void mul_kernel(const float* a, const float* b, float* out, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * b[i];
}
__global__ void scale_kernel(const float* a, float scalar, float* out, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * scalar;
}
__global__ void relu_kernel(const float* a, float* out, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] > 0.0f ? a[i] : 0.0f;
}
__global__ void silu_kernel(const float* a, float* out, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) { float x = a[i]; out[i] = x / (1.0f + expf(-x)); }
}
__global__ void gelu_kernel(const float* a, float* out, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = a[i];
    float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    out[i] = 0.5f * x * (1.0f + tanhf(c));
  }
}
__global__ void transpose_kernel(const float* a, float* out, long long rows, long long cols) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (i < n) {
    long long r = static_cast<long long>(i / static_cast<size_t>(cols));
    long long c = static_cast<long long>(i % static_cast<size_t>(cols));
    out[static_cast<size_t>(c) * rows + r] = a[i];
  }
}
__global__ void causal_mask_kernel(const float* a, float* out, long long rows, long long cols, long long diagonal) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (i < n) {
    long long r = static_cast<long long>(i / static_cast<size_t>(cols));
    long long c = static_cast<long long>(i % static_cast<size_t>(cols));
    out[i] = (c > r + diagonal) ? -1.0e30f : a[i];
  }
}

__global__ void softmax_rows_kernel(const float* a, float* out, long long rows, long long cols) {
  extern __shared__ float shared[];
  long long row = blockIdx.x;
  if (row >= rows) return;
  float local_max = -FLT_MAX;
  for (long long c = threadIdx.x; c < cols; c += blockDim.x) local_max = fmaxf(local_max, a[row * cols + c]);
  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + s]);
    __syncthreads();
  }
  float maxv = shared[0];
  float local_sum = 0.0f;
  for (long long c = threadIdx.x; c < cols; c += blockDim.x) local_sum += expf(a[row * cols + c] - maxv);
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  float denom = shared[0] > 0.0f ? shared[0] : 1.0f;
  for (long long c = threadIdx.x; c < cols; c += blockDim.x) out[row * cols + c] = expf(a[row * cols + c] - maxv) / denom;
}

__global__ void rmsnorm_kernel(const float* a, const float* weight, float* out, long long rows, long long cols, float eps) {
  extern __shared__ float shared[];
  long long row = blockIdx.x;
  if (row >= rows) return;
  float sumsq = 0.0f;
  for (long long c = threadIdx.x; c < cols; c += blockDim.x) { float x = a[row * cols + c]; sumsq += x * x; }
  shared[threadIdx.x] = sumsq;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(shared[0] / static_cast<float>(cols) + eps);
  for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
    float w = weight ? weight[c] : 1.0f;
    out[row * cols + c] = a[row * cols + c] * inv * w;
  }
}

__global__ void rope_interleaved_kernel(const float* a, float* out, long long rows, long long cols, long long position_offset, float theta) {
  long long pairs = cols / 2;
  size_t pair_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total_pairs = static_cast<size_t>(rows) * static_cast<size_t>(pairs);
  if (pair_index >= total_pairs) return;
  long long row = static_cast<long long>(pair_index / static_cast<size_t>(pairs));
  long long pair = static_cast<long long>(pair_index % static_cast<size_t>(pairs));
  long long i0 = row * cols + pair * 2;
  float exponent = -2.0f * static_cast<float>(pair) / static_cast<float>(cols);
  float freq = powf(theta, exponent);
  float angle = static_cast<float>(position_offset + row) * freq;
  float cs = cosf(angle), sn = sinf(angle);
  float x0 = a[i0], x1 = a[i0 + 1];
  out[i0] = x0 * cs - x1 * sn;
  out[i0 + 1] = x0 * sn + x1 * cs;
}

__global__ void gather_rows_kernel(const float* table, const float* indices, float* out,
                                   long long table_rows, long long cols, size_t index_count) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t n = index_count * static_cast<size_t>(cols);
  if (i >= n) return;
  size_t out_row = i / static_cast<size_t>(cols);
  long long col = static_cast<long long>(i % static_cast<size_t>(cols));
  long long src_row = static_cast<long long>(llrintf(indices[out_row]));
  out[i] = (src_row >= 0 && src_row < table_rows) ? table[src_row * cols + col] : 0.0f;
}

bool launch_elementwise(cudaError_t launch_status, long long out_handle, const char* where) {
  if (cuda_ok(launch_status, where)) return true;
  release_handle(out_handle);
  return false;
}

} // namespace

extern "C" long long gpu_available() {
  clear_error(); int count = 0; cudaError_t code = cudaGetDeviceCount(&count);
  if (code != cudaSuccess) { cudaGetLastError(); set_error(std::string("cudaGetDeviceCount: ") + cudaGetErrorString(code)); return 0; }
  return count > 0 ? 1 : 0;
}
extern "C" long long gpu_device_count() { clear_error(); int count = 0; return cuda_ok(cudaGetDeviceCount(&count), "cudaGetDeviceCount") ? count : 0; }
extern "C" const char* gpu_device_name(long long index) {
  clear_error(); cudaDeviceProp props{};
  if (!cuda_ok(cudaGetDeviceProperties(&props, static_cast<int>(index)), "cudaGetDeviceProperties")) { string_result.clear(); return string_result.c_str(); }
  string_result = props.name; return string_result.c_str();
}
extern "C" const char* gpu_last_error() { return last_error.c_str(); }
extern "C" double gpu_memory_free_mb() { clear_error(); size_t f=0,t=0; return cuda_ok(cudaMemGetInfo(&f,&t), "cudaMemGetInfo") ? static_cast<double>(f)/(1024.0*1024.0) : 0.0; }
extern "C" double gpu_memory_total_mb() { clear_error(); size_t f=0,t=0; return cuda_ok(cudaMemGetInfo(&f,&t), "cudaMemGetInfo") ? static_cast<double>(t)/(1024.0*1024.0) : 0.0; }
extern "C" long long gpu_sync() { clear_error(); return cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize") ? 1 : 0; }

extern "C" long long gpu_tensor_create(long long rows, long long cols) { clear_error(); return create_tensor(rows, cols, true); }
extern "C" long long gpu_tensor_from_csv(const char* values, long long rows, long long cols) {
  clear_error(); long long h = create_tensor(rows, cols, false); if (!h) return 0;
  auto t = get_tensor(h); if (!t) return 0;
  auto host = parse_csv(values, t->count);
  if (host.empty() && t->count) { set_error("gpu_tensor_from_csv expected exactly " + std::to_string(t->count) + " values"); release_handle(h); return 0; }
  if (!cuda_ok(cudaMemcpy(t->data, host.data(), t->count*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D")) { release_handle(h); return 0; }
  return h;
}
extern "C" long long gpu_tensor_load_f32(const char* path, long long rows, long long cols) {
  clear_error(); long long h=create_tensor(rows,cols,false); if(!h) return 0; auto t=get_tensor(h); if(!t) return 0;
  std::ifstream file(path,std::ios::binary); if(!file){set_error(std::string("cannot open float32 tensor file: ")+(path?path:"")); release_handle(h); return 0;}
  std::vector<float> host(t->count); file.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(host.size()*sizeof(float)));
  if(file.gcount()!=static_cast<std::streamsize>(host.size()*sizeof(float))){set_error("float32 tensor file size does not match requested shape"); release_handle(h); return 0;}
  char extra=0; if(file.read(&extra,1)){set_error("float32 tensor file contains more data than requested shape"); release_handle(h); return 0;}
  if(!cuda_ok(cudaMemcpy(t->data,host.data(),host.size()*sizeof(float),cudaMemcpyHostToDevice),"cudaMemcpy H2D")){release_handle(h);return 0;} return h;
}
extern "C" long long gpu_tensor_save_f32(long long h,const char* path){
  clear_error(); auto t=get_tensor(h); if(!t)return 0; std::vector<float> host(t->count);
  if(!cuda_ok(cudaMemcpy(host.data(),t->data,host.size()*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy D2H"))return 0;
  std::ofstream file(path,std::ios::binary|std::ios::trunc); if(!file){set_error(std::string("cannot write float32 tensor file: ")+(path?path:""));return 0;}
  file.write(reinterpret_cast<const char*>(host.data()),static_cast<std::streamsize>(host.size()*sizeof(float))); if(!file){set_error("failed writing float32 tensor file");return 0;} return 1;
}
extern "C" long long gpu_tensor_clone(long long h){
  clear_error(); auto t=get_tensor(h); if(!t)return 0; long long out_h=create_tensor(t->rows,t->cols,false); if(!out_h)return 0; auto out=get_tensor(out_h);
  if(!cuda_ok(cudaMemcpy(out->data,t->data,t->count*sizeof(float),cudaMemcpyDeviceToDevice),"cudaMemcpy D2D")){release_handle(out_h);return 0;} return out_h;
}
extern "C" long long gpu_tensor_free(long long h){clear_error(); return release_handle(h)?1:0;}
extern "C" long long gpu_release_all(){clear_error(); std::lock_guard<std::mutex> lock(registry_mutex()); long long n=static_cast<long long>(registry().size()); registry().clear(); return n;}
extern "C" long long gpu_tensor_rows(long long h){clear_error(); auto t=get_tensor(h); return t?t->rows:0;}
extern "C" long long gpu_tensor_cols(long long h){clear_error(); auto t=get_tensor(h); return t?t->cols:0;}
extern "C" long long gpu_tensor_count(long long h){clear_error(); auto t=get_tensor(h); return t?static_cast<long long>(t->count):0;}
extern "C" long long gpu_tensor_live_count(){clear_error(); std::lock_guard<std::mutex> lock(registry_mutex()); return static_cast<long long>(registry().size());}
extern "C" double gpu_tensor_total_mb(){clear_error(); std::lock_guard<std::mutex> lock(registry_mutex()); size_t bytes=0; for(auto& kv:registry()) bytes+=kv.second->count*sizeof(float); return static_cast<double>(bytes)/(1024.0*1024.0);}
extern "C" double gpu_tensor_get(long long h,long long index){
  clear_error(); auto t=get_tensor(h); if(!t)return 0.0; if(index<0||static_cast<size_t>(index)>=t->count){set_error("gpu_tensor_get index out of range");return 0.0;}
  float value=0; if(!cuda_ok(cudaMemcpy(&value,t->data+index,sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy D2H scalar"))return 0.0; return value;
}
extern "C" long long gpu_tensor_set(long long h,long long index,double value){
  clear_error(); auto t=get_tensor(h); if(!t)return 0; if(index<0||static_cast<size_t>(index)>=t->count){set_error("gpu_tensor_set index out of range");return 0;}
  float v=static_cast<float>(value); return cuda_ok(cudaMemcpy(t->data+index,&v,sizeof(float),cudaMemcpyHostToDevice),"cudaMemcpy H2D scalar")?1:0;
}
extern "C" const char* gpu_tensor_to_csv(long long h){
  clear_error(); auto t=get_tensor(h); if(!t){string_result.clear();return string_result.c_str();} std::vector<float> host(t->count);
  if(!cuda_ok(cudaMemcpy(host.data(),t->data,host.size()*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy D2H")){string_result.clear();return string_result.c_str();}
  std::ostringstream out; out<<std::setprecision(9); for(size_t i=0;i<host.size();++i){if(i)out<<',';out<<host[i];} string_result=out.str(); return string_result.c_str();
}

extern "C" long long gpu_matmul(long long ah,long long bh){
  clear_error(); auto a=get_tensor(ah),b=get_tensor(bh); if(!a||!b)return 0; if(a->cols!=b->rows){set_error("gpu_matmul shape mismatch");return 0;}
  long long oh=create_tensor(a->rows,b->cols,false); if(!oh)return 0; auto out=get_tensor(oh); if(!row_major_sgemm(a,b,out,false)){release_handle(oh);return 0;} return oh;
}
extern "C" long long gpu_matmul_fast(long long ah,long long bh){
  clear_error(); auto a=get_tensor(ah),b=get_tensor(bh); if(!a||!b)return 0; if(a->cols!=b->rows){set_error("gpu_matmul_fast shape mismatch");return 0;}
  long long oh=create_tensor(a->rows,b->cols,false); if(!oh)return 0; auto out=get_tensor(oh); if(!row_major_sgemm(a,b,out,true)){release_handle(oh);return 0;} return oh;
}
extern "C" long long gpu_add(long long ah,long long bh){
  clear_error(); auto a=get_tensor(ah),b=get_tensor(bh); if(!a||!b||!check_same_shape(a,b,"gpu_add"))return 0; long long oh=create_tensor(a->rows,a->cols,false); if(!oh)return 0; auto out=get_tensor(oh);
  int threads=256,blocks=static_cast<int>((a->count+threads-1)/threads); add_kernel<<<blocks,threads>>>(a->data,b->data,out->data,a->count); return launch_elementwise(cudaGetLastError(),oh,"gpu_add kernel")?oh:0;
}
extern "C" long long gpu_mul(long long ah,long long bh){
  clear_error(); auto a=get_tensor(ah),b=get_tensor(bh); if(!a||!b||!check_same_shape(a,b,"gpu_mul"))return 0; long long oh=create_tensor(a->rows,a->cols,false); if(!oh)return 0; auto out=get_tensor(oh);
  int threads=256,blocks=static_cast<int>((a->count+threads-1)/threads); mul_kernel<<<blocks,threads>>>(a->data,b->data,out->data,a->count); return launch_elementwise(cudaGetLastError(),oh,"gpu_mul kernel")?oh:0;
}
extern "C" long long gpu_scale(long long ah,double scalar){
  clear_error(); auto a=get_tensor(ah); if(!a)return 0; long long oh=unary_output_like(a); if(!oh)return 0; auto out=get_tensor(oh);
  int threads=256,blocks=static_cast<int>((a->count+threads-1)/threads); scale_kernel<<<blocks,threads>>>(a->data,static_cast<float>(scalar),out->data,a->count); return launch_elementwise(cudaGetLastError(),oh,"gpu_scale kernel")?oh:0;
}
extern "C" long long gpu_relu(long long ah){
  clear_error(); auto a=get_tensor(ah); if(!a)return 0; long long oh=unary_output_like(a); if(!oh)return 0; auto out=get_tensor(oh); int threads=256,blocks=static_cast<int>((a->count+threads-1)/threads);
  relu_kernel<<<blocks,threads>>>(a->data,out->data,a->count); return launch_elementwise(cudaGetLastError(),oh,"gpu_relu kernel")?oh:0;
}
extern "C" long long gpu_silu(long long ah){
  clear_error(); auto a=get_tensor(ah); if(!a)return 0; long long oh=unary_output_like(a); if(!oh)return 0; auto out=get_tensor(oh); int threads=256,blocks=static_cast<int>((a->count+threads-1)/threads);
  silu_kernel<<<blocks,threads>>>(a->data,out->data,a->count); return launch_elementwise(cudaGetLastError(),oh,"gpu_silu kernel")?oh:0;
}
extern "C" long long gpu_gelu(long long ah){
  clear_error(); auto a=get_tensor(ah); if(!a)return 0; long long oh=unary_output_like(a); if(!oh)return 0; auto out=get_tensor(oh); int threads=256,blocks=static_cast<int>((a->count+threads-1)/threads);
  gelu_kernel<<<blocks,threads>>>(a->data,out->data,a->count); return launch_elementwise(cudaGetLastError(),oh,"gpu_gelu kernel")?oh:0;
}
extern "C" long long gpu_transpose(long long ah){
  clear_error(); auto a=get_tensor(ah); if(!a)return 0; long long oh=create_tensor(a->cols,a->rows,false); if(!oh)return 0; auto out=get_tensor(oh); int threads=256,blocks=static_cast<int>((a->count+threads-1)/threads);
  transpose_kernel<<<blocks,threads>>>(a->data,out->data,a->rows,a->cols); return launch_elementwise(cudaGetLastError(),oh,"gpu_transpose kernel")?oh:0;
}
extern "C" long long gpu_causal_mask(long long ah,long long diagonal){
  clear_error(); auto a=get_tensor(ah); if(!a)return 0; long long oh=unary_output_like(a); if(!oh)return 0; auto out=get_tensor(oh); int threads=256,blocks=static_cast<int>((a->count+threads-1)/threads);
  causal_mask_kernel<<<blocks,threads>>>(a->data,out->data,a->rows,a->cols,diagonal); return launch_elementwise(cudaGetLastError(),oh,"gpu_causal_mask kernel")?oh:0;
}
extern "C" long long gpu_softmax_rows(long long ah){
  clear_error(); auto a=get_tensor(ah); if(!a)return 0; long long oh=unary_output_like(a); if(!oh)return 0; auto out=get_tensor(oh); int threads=256;
  softmax_rows_kernel<<<static_cast<unsigned int>(a->rows),threads,threads*sizeof(float)>>>(a->data,out->data,a->rows,a->cols); return launch_elementwise(cudaGetLastError(),oh,"gpu_softmax_rows kernel")?oh:0;
}
extern "C" long long gpu_rmsnorm(long long ah,double eps){
  clear_error(); auto a=get_tensor(ah); if(!a)return 0; long long oh=unary_output_like(a); if(!oh)return 0; auto out=get_tensor(oh); int threads=256;
  rmsnorm_kernel<<<static_cast<unsigned int>(a->rows),threads,threads*sizeof(float)>>>(a->data,nullptr,out->data,a->rows,a->cols,static_cast<float>(eps)); return launch_elementwise(cudaGetLastError(),oh,"gpu_rmsnorm kernel")?oh:0;
}
extern "C" long long gpu_rmsnorm_weighted(long long ah,long long wh,double eps){
  clear_error(); auto a=get_tensor(ah),w=get_tensor(wh); if(!a||!w)return 0; if(w->count!=static_cast<size_t>(a->cols)){set_error("gpu_rmsnorm_weighted weight length must equal input cols");return 0;}
  long long oh=unary_output_like(a); if(!oh)return 0; auto out=get_tensor(oh); int threads=256;
  rmsnorm_kernel<<<static_cast<unsigned int>(a->rows),threads,threads*sizeof(float)>>>(a->data,w->data,out->data,a->rows,a->cols,static_cast<float>(eps)); return launch_elementwise(cudaGetLastError(),oh,"gpu_rmsnorm_weighted kernel")?oh:0;
}
extern "C" long long gpu_rope_interleaved(long long ah,long long position_offset,double theta){
  clear_error(); auto a=get_tensor(ah); if(!a)return 0; if((a->cols%2)!=0){set_error("gpu_rope_interleaved requires an even column count");return 0;} if(theta<=0){set_error("gpu_rope_interleaved theta must be positive");return 0;}
  long long oh=unary_output_like(a); if(!oh)return 0; auto out=get_tensor(oh); size_t pairs=a->count/2; int threads=256,blocks=static_cast<int>((pairs+threads-1)/threads);
  rope_interleaved_kernel<<<blocks,threads>>>(a->data,out->data,a->rows,a->cols,position_offset,static_cast<float>(theta)); return launch_elementwise(cudaGetLastError(),oh,"gpu_rope_interleaved kernel")?oh:0;
}
extern "C" long long gpu_gather_rows(long long table_h,long long indices_h){
  clear_error(); auto table=get_tensor(table_h),idx=get_tensor(indices_h); if(!table||!idx)return 0; size_t nidx=idx->count; if(nidx==0){set_error("gpu_gather_rows requires indices");return 0;}
  long long oh=create_tensor(static_cast<long long>(nidx),table->cols,false); if(!oh)return 0; auto out=get_tensor(oh); size_t n=out->count; int threads=256,blocks=static_cast<int>((n+threads-1)/threads);
  gather_rows_kernel<<<blocks,threads>>>(table->data,idx->data,out->data,table->rows,table->cols,nidx); return launch_elementwise(cudaGetLastError(),oh,"gpu_gather_rows kernel")?oh:0;
}
extern "C" long long gpu_attention(long long qh,long long kh,long long vh,double scale,long long causal){
  clear_error(); auto q=get_tensor(qh),k=get_tensor(kh),v=get_tensor(vh); if(!q||!k||!v)return 0;
  if(q->cols!=k->cols){set_error("gpu_attention requires Q.cols == K.cols");return 0;} if(k->rows!=v->rows){set_error("gpu_attention requires K.rows == V.rows");return 0;}
  long long kt=gpu_transpose(kh); if(!kt)return 0;
  long long scores=gpu_matmul_fast(qh,kt); if(!scores){release_handle(kt);return 0;}
  release_handle(kt);
  long long scaled=gpu_scale(scores,scale); if(!scaled){release_handle(scores);return 0;} release_handle(scores);
  long long masked=scaled;
  if(causal){ long long diag=k->rows-q->rows; masked=gpu_causal_mask(scaled,diag); release_handle(scaled); if(!masked)return 0; }
  long long probs=gpu_softmax_rows(masked); release_handle(masked); if(!probs)return 0;
  long long out=gpu_matmul_fast(probs,vh); release_handle(probs); return out;
}
