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
  const extensionExample = path.join(repo, 'examples', 'gpu_extension_demo.morph');
  const affineExtension = path.join(repo, 'examples', 'extensions', 'gpu_affine_extension.cu');
  const includeDir = path.join(repo, 'runtime', 'gpu');
  const stdlib = fs.readFileSync(path.join(repo, 'stdlib', 'gpu.morph'), 'utf8');
  const cudaV2 = fs.readFileSync(path.join(repo, 'runtime', 'gpu', 'morph_gpu_v2.cu'), 'utf8');
  const cudaV3 = fs.readFileSync(path.join(repo, 'runtime', 'gpu', 'morph_gpu_v3.cu'), 'utf8');
  const cudaV4 = fs.readFileSync(path.join(repo, 'runtime', 'gpu', 'morph_gpu_v4.cu'), 'utf8');
  const extensionHeader = fs.readFileSync(path.join(repo, 'runtime', 'gpu', 'morph_gpu_extension.h'), 'utf8');
  const affineSource = fs.readFileSync(affineExtension, 'utf8');

  const cpp = compileToCpp(computeExample);
  const aiCpp = compileToCpp(aiExample);
  const memoryCpp = compileToCpp(memoryExample);
  const extensionCpp = compileToCpp(extensionExample);
  check(cpp.includes('extern "C" long long gpu_available();'), 'GPU stdlib compiles into native C ABI declaration');
  check(cpp.includes('extern "C" long long gpu_matmul(long long a, long long b);'), 'gpu_matmul is wired through Morpheus extern compilation');
  check(cpp.includes('extern "C" const char* gpu_tensor_to_csv(long long handle);'), 'string-return GPU FFI is generated correctly');
  check(aiCpp.includes('extern "C" long long gpu_attention(long long q, long long k, long long v, double scale, long long causal);'), 'attention primitive compiles through Morpheus FFI');
  check(memoryCpp.includes('gpu_cosine_scores'), 'Perseus-style cosine scoring compiles through Morpheus FFI');
  check(memoryCpp.includes('gpu_embedding_mean_l2'), 'fused wave embedding compiles through Morpheus FFI');
  check(memoryCpp.includes('gpu_topk_indices'), 'GPU top-k retrieval compiles through Morpheus FFI');
  check(extensionCpp.includes('extern "C" long long gpu_affine(long long input, double scale, double bias);'), 'third-party GPU extension function compiles through ordinary Morpheus extern syntax');

  for (const api of [
    'gpu_tensor_load_f32', 'gpu_tensor_clone', 'gpu_tensor_live_count', 'gpu_release_all',
    'gpu_matmul_fast', 'gpu_mul', 'gpu_softmax_rows', 'gpu_rmsnorm', 'gpu_rmsnorm_weighted',
    'gpu_silu', 'gpu_gelu', 'gpu_rope_interleaved', 'gpu_gather_rows', 'gpu_attention',
    'gpu_sum_rows', 'gpu_l2_normalize_rows', 'gpu_cosine_scores',
    'gpu_embedding_mean_l2', 'gpu_topk_indices'
  ]) check(stdlib.includes(api), `GPU stdlib exports ${api}`);
  check(!stdlib.includes('float*') && !stdlib.includes('void*'), 'Morpheus language GPU API does not expose raw device pointers');

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

  check(cudaV4.includes('#include "morph_gpu_v3.cu"'), 'v4 preserves the v3 runtime in one ownership domain');
  for (const symbol of ['morph_gpu_tensor_view_f32', 'morph_gpu_tensor_alloc_f32', 'morph_gpu_tensor_release_native', 'morph_gpu_set_extension_error']) {
    check(extensionHeader.includes(symbol), `native extension ABI declares ${symbol}`);
    check(cudaV4.includes(symbol), `v4 runtime implements ${symbol}`);
  }
  check(!stdlib.includes('morph_gpu_tensor_view_f32'), 'raw native extension ABI is not exposed to Morpheus source');
  check(affineSource.includes('morph_gpu_tensor_view_f32') && affineSource.includes('morph_gpu_tensor_alloc_f32'), 'generic affine extension consumes the native tensor ABI');

  const args = buildNvccArgs({
    cppPath: 'program.gpu.cpp',
    gpuRuntimePath: 'runtime/gpu/morph_gpu_v4.cu',
    extensionPaths: ['extension.cu'],
    outPath: 'program.gpu.exe',
    arch: 'sm_89',
    includeDir
  });
  check(args.includes('-lcublas'), 'GPU build links cuBLAS');
  check(args.includes('-arch=sm_89'), 'GPU build accepts explicit CUDA architecture');
  check(args.includes('-x') && args.includes('cu'), 'Generated Morpheus C++ is compiled as CUDA translation unit');
  check(args.includes('-lineinfo'), 'GPU build retains kernel line information for profiling/debugging');
  check(args.includes('extension.cu'), 'GPU build accepts native CUDA extensions');
  check(args.includes(includeDir), 'GPU build exposes the extension ABI header include directory');
  check(runtimeFile('v4') === 'morph_gpu_v4.cu', 'v4 runtime is selectable by build driver');

  const cases = [
    [computeExample, 'general compute'],
    [aiExample, 'transformer-oriented AI'],
    [memoryExample, 'Perseus memory search']
  ];
  for (const [file, label] of cases) {
    const dry = buildGpuProgram({ file, arch: 'sm_89', dryRun: true, keepCpp: true, runtime: 'v4', extensions: [] });
    check(dry.dryRun === true, `${label} build supports CUDA-free dry-run validation`);
    check(fs.existsSync(dry.cppPath), `${label} example compiles from Morpheus to native C++`);
    check(dry.args.some((x) => String(x).endsWith('morph_gpu_v4.cu')), `${label} build links v4 runtime`);
    try { fs.unlinkSync(dry.cppPath); } catch {}
  }

  const extDry = buildGpuProgram({
    file: extensionExample,
    arch: 'sm_89',
    dryRun: true,
    keepCpp: true,
    runtime: 'v4',
    extensions: [affineExtension]
  });
  check(extDry.extensionPaths.length === 1, 'extension demo records one native CUDA extension');
  check(extDry.args.includes(path.resolve(affineExtension)), 'extension demo links the requested extension source');
  try { fs.unlinkSync(extDry.cppPath); } catch {}

  const defaultDry = buildGpuProgram({ file: memoryExample, arch: 'sm_89', dryRun: true, keepCpp: true });
  check(defaultDry.runtime === 'v4', 'extension-capable v4 is the default GPU runtime');
  try { fs.unlinkSync(defaultDry.cppPath); } catch {}

  process.stdout.write(`\nMORPHEUS GPU RUNTIME VALIDATION: ${pass}/${total} passed\n`);
}

try { run(); }
catch (error) { process.stderr.write(`${error.stack || error}\n`); process.exit(1); }
