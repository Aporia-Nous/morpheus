// Morpheus CUDA runtime v0.3
// Extends v0.2 with vector-search / memory primitives chosen from actual
// Perseus Mnemosyne workloads. v0.2 is included into this translation unit so
// the same opaque tensor registry and ownership model are reused.

#include "morph_gpu_v2.cu"

__global__ void sum_rows_kernel(const float* a, float* out, long long rows, long long cols) {
  long long c = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  float sum = 0.0f;
  for (long long r = 0; r < rows; ++r) sum += a[r * cols + c];
  out[c] = sum;
}

__global__ void l2_normalize_rows_kernel(const float* a, float* out, long long rows, long long cols, float eps) {
  extern __shared__ float shared[];
  long long row = blockIdx.x;
  if (row >= rows) return;
  float ss = 0.0f;
  for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
    float x = a[row * cols + c];
    ss += x * x;
  }
  shared[threadIdx.x] = ss;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(shared[0] + eps);
  for (long long c = threadIdx.x; c < cols; c += blockDim.x) out[row * cols + c] = a[row * cols + c] * inv;
}

__global__ void cosine_scores_kernel(const float* matrix, const float* query, float* out,
                                     long long rows, long long cols, float eps) {
  extern __shared__ float shared[];
  float* dotbuf = shared;
  float* mbuf = shared + blockDim.x;
  float* qbuf = shared + 2 * blockDim.x;
  long long row = blockIdx.x;
  if (row >= rows) return;

  float dot = 0.0f, mss = 0.0f, qss = 0.0f;
  for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
    float m = matrix[row * cols + c];
    float q = query[c];
    dot += m * q;
    mss += m * m;
    qss += q * q;
  }
  dotbuf[threadIdx.x] = dot;
  mbuf[threadIdx.x] = mss;
  qbuf[threadIdx.x] = qss;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      dotbuf[threadIdx.x] += dotbuf[threadIdx.x + s];
      mbuf[threadIdx.x] += mbuf[threadIdx.x + s];
      qbuf[threadIdx.x] += qbuf[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float denom = sqrtf(mbuf[0] * qbuf[0]) + eps;
    out[row] = dotbuf[0] / denom;
  }
}

__global__ void embedding_mean_l2_kernel(const float* table, const float* indices, float* out,
                                         long long table_rows, long long cols, size_t nidx, float eps) {
  extern __shared__ float shared[];
  float sum = 0.0f;
  for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
    sum = 0.0f;
    for (size_t j = 0; j < nidx; ++j) {
      long long row = static_cast<long long>(llrintf(indices[j]));
      if (row >= 0 && row < table_rows) sum += table[row * cols + c];
    }
    float mean = nidx ? sum / static_cast<float>(nidx) : 0.0f;
    out[c] = mean;
  }
  __syncthreads();

  float ss = 0.0f;
  for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
    float x = out[c];
    ss += x * x;
  }
  shared[threadIdx.x] = ss;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(shared[0] + eps);
  for (long long c = threadIdx.x; c < cols; c += blockDim.x) out[c] *= inv;
}

// Small-k exact top-k for retrieval workloads. One block repeatedly performs a
// parallel max reduction over all scores, then masks the winner in a temporary
// device copy. This is intentionally optimized for k << N (Mnemosyne uses ~6-8).
__global__ void topk_indices_kernel(float* work, size_t n, float* out_indices, long long k) {
  extern __shared__ unsigned char raw[];
  float* vals = reinterpret_cast<float*>(raw);
  int* ids = reinterpret_cast<int*>(vals + blockDim.x);

  for (long long pick = 0; pick < k; ++pick) {
    float best = -FLT_MAX;
    int best_id = -1;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
      float v = work[i];
      if (v > best || (v == best && static_cast<int>(i) < best_id)) {
        best = v;
        best_id = static_cast<int>(i);
      }
    }
    vals[threadIdx.x] = best;
    ids[threadIdx.x] = best_id;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        float rv = vals[threadIdx.x + s];
        int ri = ids[threadIdx.x + s];
        float lv = vals[threadIdx.x];
        int li = ids[threadIdx.x];
        if (rv > lv || (rv == lv && ri >= 0 && (li < 0 || ri < li))) {
          vals[threadIdx.x] = rv;
          ids[threadIdx.x] = ri;
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      int winner = ids[0];
      out_indices[pick] = static_cast<float>(winner);
      if (winner >= 0) work[winner] = -FLT_MAX;
    }
    __syncthreads();
  }
}

extern "C" long long gpu_sum_rows(long long ah) {
  clear_error();
  auto a = get_tensor(ah); if (!a) return 0;
  long long oh = create_tensor(1, a->cols, false); if (!oh) return 0;
  auto out = get_tensor(oh);
  int threads = 256;
  int blocks = static_cast<int>((a->cols + threads - 1) / threads);
  sum_rows_kernel<<<blocks, threads>>>(a->data, out->data, a->rows, a->cols);
  return launch_elementwise(cudaGetLastError(), oh, "gpu_sum_rows kernel") ? oh : 0;
}

extern "C" long long gpu_l2_normalize_rows(long long ah, double epsilon) {
  clear_error();
  auto a = get_tensor(ah); if (!a) return 0;
  long long oh = unary_output_like(a); if (!oh) return 0;
  auto out = get_tensor(oh);
  int threads = 256;
  l2_normalize_rows_kernel<<<static_cast<unsigned int>(a->rows), threads, threads * sizeof(float)>>>(
    a->data, out->data, a->rows, a->cols, static_cast<float>(epsilon));
  return launch_elementwise(cudaGetLastError(), oh, "gpu_l2_normalize_rows kernel") ? oh : 0;
}

extern "C" long long gpu_cosine_scores(long long matrix_h, long long query_h) {
  clear_error();
  auto matrix = get_tensor(matrix_h), query = get_tensor(query_h);
  if (!matrix || !query) return 0;
  if (query->count != static_cast<size_t>(matrix->cols)) {
    set_error("gpu_cosine_scores query length must equal matrix columns");
    return 0;
  }
  long long oh = create_tensor(matrix->rows, 1, false); if (!oh) return 0;
  auto out = get_tensor(oh);
  int threads = 256;
  size_t shared = static_cast<size_t>(threads) * 3 * sizeof(float);
  cosine_scores_kernel<<<static_cast<unsigned int>(matrix->rows), threads, shared>>>(
    matrix->data, query->data, out->data, matrix->rows, matrix->cols, 1.0e-12f);
  return launch_elementwise(cudaGetLastError(), oh, "gpu_cosine_scores kernel") ? oh : 0;
}

extern "C" long long gpu_embedding_mean_l2(long long table_h, long long indices_h) {
  clear_error();
  auto table = get_tensor(table_h), idx = get_tensor(indices_h);
  if (!table || !idx) return 0;
  if (idx->count == 0) { set_error("gpu_embedding_mean_l2 requires at least one index"); return 0; }
  long long oh = create_tensor(1, table->cols, false); if (!oh) return 0;
  auto out = get_tensor(oh);
  int threads = 256;
  embedding_mean_l2_kernel<<<1, threads, threads * sizeof(float)>>>(
    table->data, idx->data, out->data, table->rows, table->cols, idx->count, 1.0e-12f);
  return launch_elementwise(cudaGetLastError(), oh, "gpu_embedding_mean_l2 kernel") ? oh : 0;
}

extern "C" long long gpu_topk_indices(long long scores_h, long long k) {
  clear_error();
  auto scores = get_tensor(scores_h); if (!scores) return 0;
  if (k <= 0) { set_error("gpu_topk_indices k must be positive"); return 0; }
  if (k > static_cast<long long>(scores->count)) k = static_cast<long long>(scores->count);
  if (scores->count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    set_error("gpu_topk_indices currently supports at most INT_MAX scores");
    return 0;
  }

  float* work = nullptr;
  const size_t bytes = scores->count * sizeof(float);
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&work), bytes), "gpu_topk_indices cudaMalloc")) return 0;
  if (!cuda_ok(cudaMemcpy(work, scores->data, bytes, cudaMemcpyDeviceToDevice), "gpu_topk_indices D2D")) {
    cudaFree(work); return 0;
  }

  long long oh = create_tensor(k, 1, false);
  if (!oh) { cudaFree(work); return 0; }
  auto out = get_tensor(oh);
  int threads = 256;
  size_t shared = static_cast<size_t>(threads) * (sizeof(float) + sizeof(int));
  topk_indices_kernel<<<1, threads, shared>>>(work, scores->count, out->data, k);
  cudaError_t launch = cudaGetLastError();
  cudaFree(work);
  return launch_elementwise(launch, oh, "gpu_topk_indices kernel") ? oh : 0;
}
