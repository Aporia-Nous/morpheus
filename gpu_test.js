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
  const stdlib = fs.readFileSync(path.join(repo, 'stdlib', 'gpu.morph'), 'utf8');
  const cuda = fs.readFileSync(path.join(repo, 'runtime', 'gpu', 'morph_gpu_v2.cu'), 'utf8');

  const cpp = compileToCpp(computeExample);
  const aiCpp = compileToCpp(aiExample);
  check(cpp.includes('extern "C" long long gpu_available();'), 'GPU stdlib compiles into native C ABI declaration');
  check(cpp.includes('extern "C" long long gpu_matmul(long long a, long long b);'), 'gpu_matmul is wired through Morpheus extern compilation');
  check(cpp.includes('extern "C" const char* gpu_tensor_to_csv(long long handle);'), 'string-return GPU FFI is generated correctly');
  check(aiCpp.includes('extern "C" long long gpu_attention(long long q, long long k, long long v, double scale, long long causal);'), 'multi-argument attention primitive compiles through Morpheus FFI');
  check(aiCpp.includes('gpu_rmsnorm_weighted'), 'AI example compiles weighted RMSNorm call');

  for (const api of [
    'gpu_tensor_load_f32', 'gpu_tensor_clone', 'gpu_tensor_live_count', 'gpu_release_all',
    'gpu_matmul_fast', 'gpu_mul', 'gpu_softmax_rows', 'gpu_rmsnorm', 'gpu_rmsnorm_weighted',
    'gpu_silu', 'gpu_gelu', 'gpu_rope_interleaved', 'gpu_gather_rows', 'gpu_attention'
  ]) check(stdlib.includes(api), `GPU stdlib exports ${api}`);
  check(!stdlib.includes('float*') && !stdlib.includes('void*'), 'Morpheus GPU API does not expose raw device pointers');

  for (const marker of [
    'cudaMalloc', 'cudaMemcpy', 'cublasSgemm', 'CUBLAS_TF32_TENSOR_OP_MATH',
    'softmax_rows_kernel<<<', 'rmsnorm_kernel<<<', 'rope_interleaved_kernel<<<',
    'gather_rows_kernel<<<', 'silu_kernel<<<', 'gelu_kernel<<<'
  ]) check(cuda.includes(marker), `enriched CUDA runtime contains ${marker}`);
  check(cuda.includes('thread_local CublasContext'), 'cuBLAS handle is reused per thread instead of recreated per matmul');
  check(cuda.includes('gpu_attention'), 'CUDA runtime contains composed attention primitive');
  check(cuda.includes('gpu_tensor_total_mb'), 'CUDA runtime exposes live tensor memory accounting');

  const args = buildNvccArgs({
    cppPath: 'program.gpu.cpp',
    gpuRuntimePath: 'runtime/gpu/morph_gpu_v2.cu',
    outPath: 'program.gpu.exe',
    arch: 'sm_89'
  });
  check(args.includes('-lcublas'), 'GPU build links cuBLAS');
  check(args.includes('-arch=sm_89'), 'GPU build accepts explicit CUDA architecture');
  check(args.includes('-x') && args.includes('cu'), 'Generated Morpheus C++ is compiled as CUDA translation unit');
  check(args.includes('-lineinfo'), 'GPU build retains kernel line information for profiling/debugging');
  check(runtimeFile('v2') === 'morph_gpu_v2.cu', 'v0.2 runtime is selectable by build driver');

  const computeDry = buildGpuProgram({ file: computeExample, arch: 'sm_89', dryRun: true, keepCpp: true, runtime: 'v2' });
  check(computeDry.dryRun === true, 'GPU build driver supports CUDA-free dry-run validation');
  check(fs.existsSync(computeDry.cppPath), 'GPU build driver generated native C++ from compute example');
  check(computeDry.args.some((x) => String(x).endsWith('morph_gpu_v2.cu')), 'GPU build defaults/selects enriched v0.2 runtime');
  try { fs.unlinkSync(computeDry.cppPath); } catch {}

  const aiDry = buildGpuProgram({ file: aiExample, arch: 'sm_89', dryRun: true, keepCpp: true, runtime: 'v2' });
  check(fs.existsSync(aiDry.cppPath), 'transformer-oriented Morpheus example compiles to native C++');
  check(aiDry.args.includes('-lcublas'), 'AI primitive build retains cuBLAS linkage');
  try { fs.unlinkSync(aiDry.cppPath); } catch {}

  process.stdout.write(`\nMORPHEUS GPU RUNTIME VALIDATION: ${pass}/${total} passed\n`);
}

try { run(); }
catch (error) { process.stderr.write(`${error.stack || error}\n`); process.exit(1); }
