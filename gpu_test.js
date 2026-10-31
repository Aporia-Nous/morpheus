'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { compileToCpp } = require('./morphc.js');
const { buildGpuProgram, buildNvccArgs } = require('./gpu_build.js');

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
  const example = path.join(repo, 'examples', 'gpu_compute.morph');
  const stdlib = fs.readFileSync(path.join(repo, 'stdlib', 'gpu.morph'), 'utf8');
  const cuda = fs.readFileSync(path.join(repo, 'runtime', 'gpu', 'morph_gpu.cu'), 'utf8');

  const cpp = compileToCpp(example);
  check(cpp.includes('extern "C" long long gpu_available();'), 'GPU stdlib compiles into native C ABI declaration');
  check(cpp.includes('extern "C" long long gpu_matmul(long long a, long long b);'), 'gpu_matmul is wired through Morpheus extern compilation');
  check(cpp.includes('extern "C" const char* gpu_tensor_to_csv(long long handle);'), 'string-return GPU FFI is generated correctly');

  check(stdlib.includes('gpu_tensor_load_f32'), 'GPU stdlib includes direct binary float32 loading');
  check(stdlib.includes('gpu_tensor_free'), 'GPU stdlib exposes explicit tensor lifetime control');
  check(!stdlib.includes('float*') && !stdlib.includes('void*'), 'Morpheus GPU API does not expose raw device pointers');

  check(cuda.includes('cudaMalloc'), 'CUDA runtime allocates device memory');
  check(cuda.includes('cudaMemcpy'), 'CUDA runtime implements host/device transfer');
  check(cuda.includes('cublasSgemm'), 'CUDA runtime uses cuBLAS SGEMM for matrix multiplication');
  check(cuda.includes('add_kernel<<<'), 'CUDA runtime contains a native vector-add kernel');
  check(cuda.includes('relu_kernel<<<'), 'CUDA runtime contains a native ReLU kernel');

  const args = buildNvccArgs({
    cppPath: 'program.gpu.cpp',
    gpuRuntimePath: 'runtime/gpu/morph_gpu.cu',
    outPath: 'program.gpu.exe',
    arch: 'sm_89'
  });
  check(args.includes('-lcublas'), 'GPU build links cuBLAS');
  check(args.includes('-arch=sm_89'), 'GPU build accepts explicit CUDA architecture');
  check(args.includes('-x') && args.includes('cu'), 'Generated Morpheus C++ is compiled as CUDA translation unit');

  // Dry-run the real build driver. This compiles Morpheus source to generated C++
  // without requiring CUDA/nvcc on CI machines.
  const result = buildGpuProgram({ file: example, arch: 'sm_89', dryRun: true, keepCpp: true });
  check(result.dryRun === true, 'GPU build driver supports CUDA-free dry-run validation');
  check(fs.existsSync(result.cppPath), 'GPU build driver generated native C++ from .morph source');
  check(result.args.some((x) => String(x).endsWith('morph_gpu.cu')), 'GPU build driver links the Morpheus CUDA runtime source');
  try { fs.unlinkSync(result.cppPath); } catch {}

  process.stdout.write(`\nMORPHEUS GPU RUNTIME VALIDATION: ${pass}/${total} passed\n`);
}

try {
  run();
} catch (error) {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
}
