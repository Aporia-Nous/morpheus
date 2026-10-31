#!/usr/bin/env node
/* Morpheus GPU build driver.
   Compiles ordinary .morph source through morphc, then links the generated
   native C++ with the selected NVIDIA CUDA runtime + cuBLAS.

   v1 = foundational tensor compute
   v2 = transformer-oriented primitives
   v3 = v2 + Perseus/Mnemosyne vector-search and embedding primitives (default)
*/
'use strict';

const fs = require('fs');
const path = require('path');
const cp = require('child_process');
const { compileToCpp } = require('./morphc.js');

function resolveNvcc(env = process.env) {
  if (env.MORPHEUS_NVCC) return env.MORPHEUS_NVCC;
  if (env.CUDA_PATH) {
    const candidate = path.join(env.CUDA_PATH, 'bin', process.platform === 'win32' ? 'nvcc.exe' : 'nvcc');
    if (fs.existsSync(candidate)) return candidate;
  }
  return process.platform === 'win32' ? 'nvcc.exe' : 'nvcc';
}

function parseArgs(argv) {
  const args = { file: null, out: null, arch: null, dryRun: false, keepCpp: false, runtime: 'v3' };
  for (let i = 0; i < argv.length; ++i) {
    const token = argv[i];
    if (!args.file && !token.startsWith('-')) args.file = token;
    else if (token === '-o' && argv[i + 1]) args.out = argv[++i];
    else if (token === '--arch' && argv[i + 1]) args.arch = argv[++i];
    else if (token === '--runtime' && argv[i + 1]) args.runtime = argv[++i];
    else if (token === '--dry-run') args.dryRun = true;
    else if (token === '--keep-cpp') args.keepCpp = true;
    else throw new Error(`Unknown GPU build option: ${token}`);
  }
  return args;
}

function runtimeFile(version) {
  if (version === 'v1') return 'morph_gpu.cu';
  if (version === 'v2') return 'morph_gpu_v2.cu';
  if (version === 'v3') return 'morph_gpu_v3.cu';
  throw new Error(`Unknown GPU runtime version: ${version}`);
}

function buildNvccArgs({ cppPath, gpuRuntimePath, outPath, arch }) {
  const args = [
    '-std=c++17',
    '-O3',
    '-lineinfo',
    '-x', 'cu',
    cppPath,
    gpuRuntimePath,
    '-lcublas',
    '-o', outPath
  ];
  if (arch) args.unshift(`-arch=${arch}`);
  return args;
}

function buildGpuProgram(options) {
  const abs = path.resolve(options.file);
  if (!fs.existsSync(abs)) throw new Error(`Morpheus source not found: ${abs}`);
  if (!/\.morph$/i.test(abs)) throw new Error('GPU build input must be a .morph file.');

  const cppPath = abs.replace(/\.morph$/i, '.gpu.cpp');
  const defaultExe = abs.replace(/\.morph$/i, process.platform === 'win32' ? '.gpu.exe' : '.gpu');
  const outPath = path.resolve(options.out || defaultExe);
  const version = options.runtime || 'v3';
  const gpuRuntimePath = path.resolve(__dirname, 'runtime', 'gpu', runtimeFile(version));
  if (!fs.existsSync(gpuRuntimePath)) throw new Error(`GPU runtime source not found: ${gpuRuntimePath}`);
  const nvcc = resolveNvcc();

  fs.writeFileSync(cppPath, compileToCpp(abs));
  const nvccArgs = buildNvccArgs({ cppPath, gpuRuntimePath, outPath, arch: options.arch });

  if (options.dryRun) return { nvcc, args: nvccArgs, cppPath, outPath, gpuRuntimePath, runtime: version, dryRun: true };

  const env = { ...process.env };
  const result = cp.spawnSync(nvcc, nvccArgs, { stdio: 'inherit', env });
  if (result.error) {
    if (!options.keepCpp) { try { fs.unlinkSync(cppPath); } catch {} }
    throw new Error(`Could not run nvcc (${nvcc}): ${result.error.message}. Install the NVIDIA CUDA Toolkit or set MORPHEUS_NVCC.`);
  }
  if (result.status !== 0) {
    if (!options.keepCpp) { try { fs.unlinkSync(cppPath); } catch {} }
    throw new Error(`CUDA build failed with exit code ${result.status}.`);
  }

  if (!options.keepCpp) { try { fs.unlinkSync(cppPath); } catch {} }
  return { nvcc, args: nvccArgs, cppPath, outPath, gpuRuntimePath, runtime: version, dryRun: false };
}

function main() {
  let options;
  try { options = parseArgs(process.argv.slice(2)); }
  catch (error) { process.stderr.write(`${error.message}\n`); process.exit(2); }

  if (!options.file) {
    process.stdout.write(
      'Usage: node gpu_build.js <file.morph> [-o output] [--arch sm_89] [--runtime v3] [--keep-cpp] [--dry-run]\n' +
      'RTX 4060 Ti example: node gpu_build.js examples/gpu_memory_search.morph --arch sm_89\n'
    );
    process.exit(2);
  }

  try {
    const result = buildGpuProgram(options);
    if (result.dryRun) process.stdout.write(`RUNTIME: ${result.runtime}\nNVCC: ${result.nvcc}\nARGS: ${result.args.map((x) => JSON.stringify(x)).join(' ')}\n`);
    else process.stdout.write(`Built CUDA Morpheus program (${result.runtime}): ${result.outPath}\n`);
  } catch (error) {
    process.stderr.write(`gpu build error: ${error.message || error}\n`);
    process.exit(1);
  }
}

if (require.main === module) main();
module.exports = { resolveNvcc, parseArgs, runtimeFile, buildNvccArgs, buildGpuProgram };
