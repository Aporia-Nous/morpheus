'use strict';

const fs = require('fs');
const path = require('path');
const { compileToCpp } = require('./morphc.js');
const { buildGpuProgram, buildNvccArgs, runtimeFile } = require('./gpu_build.js');

let pass = 0;
let total = 0;
function check(condition, label) {
  total++;
  if (!condition) throw new Error(`[FAIL] ${label}`);
  pass++;
  process.stdout.write(`[PASS] ${label}\n`);
}

function run() {
  const repo = __dirname;
  const computeExample = path.join(repo, 'examples', 'gpu_compute.morph');
  const aiExample = path.join(repo, 'examples', 'gpu_ai_primitives.morph');
  const memoryExample = path.join(repo, 'examples', 'gpu_memory_search.morph');
  const stdlib = fs.readFileSync(path.join(repo, 'stdlib', 'gpu.morph'), 'utf8');
  const cudaV2 = fs.readFileSync(path.join(repo, 'runtime', 'gpu', 'morph_gpu_v2.cu'), 'utf8');
  const cudaV3 = fs.readFileSync(path.join(repo, 'runtime', 'gpu', 'morph_gpu_v3.cu'), 'utf8');

  const cpp = compileToCpp(computeExample);
  const aiCpp = compileToCpp(aiExample);
  const memoryCpp = compileToCpp(memoryExample);
  check(cpp.includes('extern "C" long long gpu_available();'), 'GPU stdlib compiles into native C ABI declaration');
  check(cpp.includes('extern "C" long long gpu_matmul(long long a, long long b);'), 'gpu_matmul is wired through Morpheus extern compilation');
  check(cpp.includes('extern "C" const char* gpu_tensor_to_csv(long long handle);'), 'string-return GPU FFI is generated correctly');
  check(aiCpp.includes('extern "C" long long gpu_attention(long long q, long long k, long long v, double scale, long long causal);'), 'attention primitive compiles through Morpheus FFI');
  check(memoryCpp.includes('gpu_cosine_scores'), 'Perseus-style cosine scoring compiles through Morpheus FFI');
  check(memoryCpp.includes('gpu_embedding_mean_l2'), 'fused wave embedding compiles through Morpheus FFI');
  check(memoryCpp.includes('gpu_topk_indices'), 'GPU top-k retrieval compiles through Morpheus FFI');

  for (const api of [
    'gpu_tensor_load_f32', 'gpu_tensor_clone', 'gpu_tensor_live_count', 'gpu_release_all',
    'gpu_matmul_fast', 'gpu_mul', 'gpu_softmax_rows', 'gpu_rmsnorm', 'gpu_rmsnorm_weighted',
    'gpu_silu', 'gpu_gelu', 'gpu_rope_interleaved', 'gpu_gather_rows', 'gpu_attention',
    'gpu_sum_rows', 'gpu_l2_normalize_rows', 'gpu_cosine_scores',
    'gpu_embedding_mean_l2', 'gpu_topk_indices'
  ]) check(stdlib.includes(api), `GPU stdlib exports ${api}`);
  check(!stdlib.includes('float*') && !stdlib.includes('void*'), 'Morpheus GPU API does not expose raw device pointers');

  for (const marker of [
    'cudaMalloc', 'cudaMemcpy', 'cublasSgemm', 'CUBLAS_TF32_TENSOR_OP_MATH',
    'softmax_rows_kernel<<<', 'rmsnorm_kernel<<<', 'rope_interleaved_kernel<<<',
    'gather_rows_kernel<<<', 'silu_kernel<<<', 'gelu_kernel<<<'
  ]) check(cudaV2.includes(marker), `v2 CUDA base contains ${marker}`);
  check(cudaV2.includes('thread_local CublasContext'), 'cuBLAS handle is reused per thread instead of recreated per matmul');
  check(cudaV2.includes('gpu_attention'), 'v2 CUDA base contains composed attention primitive');

  check(cudaV3.includes('#include "morph_gpu_v2.cu"'), 'v3 extends the same v2 tensor registry and ownership model');
  for (const marker of [
    'sum_rows_kernel<<<', 'l2_normalize_rows_kernel<<<', 'cosine_scores_kernel<<<',
    'embedding_mean_l2_kernel<<<', 'topk_indices_kernel<<<',
    'gpu_cosine_scores', 'gpu_embedding_mean_l2', 'gpu_topk_indices'
  ]) check(cudaV3.includes(marker), `v3 memory runtime contains ${marker}`);
  check(cudaV3.includes('k << N'), 'top-k implementation explicitly targets retrieval-style small k');

  const args = buildNvccArgs({
    cppPath: 'program.gpu.cpp',
    gpuRuntimePath: 'runtime/gpu/morph_gpu_v3.cu',
    outPath: 'program.gpu.exe',
    arch: 'sm_89'
  });
  check(args.includes('-lcublas'), 'GPU build links cuBLAS');
  check(args.includes('-arch=sm_89'), 'GPU build accepts explicit CUDA architecture');
  check(args.includes('-x') && args.includes('cu'), 'Generated Morpheus C++ is compiled as CUDA translation unit');
  check(args.includes('-lineinfo'), 'GPU build retains kernel line information for profiling/debugging');
  check(runtimeFile('v3') === 'morph_gpu_v3.cu', 'v3 runtime is selectable by build driver');

  const cases = [
    [computeExample, 'general compute'],
    [aiExample, 'transformer-oriented AI'],
    [memoryExample, 'Perseus memory search']
  ];
  for (const [file, label] of cases) {
    const dry = buildGpuProgram({ file, arch: 'sm_89', dryRun: true, keepCpp: true, runtime: 'v3' });
    check(dry.dryRun === true, `${label} build supports CUDA-free dry-run validation`);
    check(fs.existsSync(dry.cppPath), `${label} example compiles from Morpheus to native C++`);
    check(dry.args.some((x) => String(x).endsWith('morph_gpu_v3.cu')), `${label} build links v3 runtime`);
    try { fs.unlinkSync(dry.cppPath); } catch {}
  }

  const defaultDry = buildGpuProgram({ file: memoryExample, arch: 'sm_89', dryRun: true, keepCpp: true });
  check(defaultDry.runtime === 'v3', 'Perseus-oriented v3 is the default GPU runtime');
  try { fs.unlinkSync(defaultDry.cppPath); } catch {}

  process.stdout.write(`\nMORPHEUS GPU RUNTIME VALIDATION: ${pass}/${total} passed\n`);
}

try { run(); }
catch (error) { process.stderr.write(`${error.stack || error}\n`); process.exit(1); }
