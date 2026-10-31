#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const cp = require('child_process');
const { resolveNvcc } = require('./gpu_build.js');

function parseArgs(argv) {
  const out = { output: null, arch: null, dryRun: false };
  for (let i = 0; i < argv.length; ++i) {
    const a = argv[i];
    if (a === '-o' && argv[i + 1]) out.output = argv[++i];
    else if (a === '--arch' && argv[i + 1]) out.arch = argv[++i];
    else if (a === '--dry-run') out.dryRun = true;
    else throw new Error(`Unknown worker build option: ${a}`);
  }
  return out;
}

function workerBuildArgs({ source, output, arch }) {
  const args = ['-std=c++17', '-O3', '-lineinfo', source, '-lcublas', '-o', output];
  if (arch) args.unshift(`-arch=${arch}`);
  return args;
}

function buildWorker(options = {}) {
  const source = path.resolve(__dirname, 'runtime', 'gpu', 'memory_worker.cu');
  if (!fs.existsSync(source)) throw new Error(`GPU memory worker source not found: ${source}`);
  const output = path.resolve(options.output || path.join(__dirname, process.platform === 'win32' ? 'morpheus-gpu-memory-worker.exe' : 'morpheus-gpu-memory-worker'));
  const nvcc = resolveNvcc();
  const args = workerBuildArgs({ source, output, arch: options.arch });
  if (options.dryRun) return { nvcc, args, source, output, dryRun: true };

  const result = cp.spawnSync(nvcc, args, { stdio: 'inherit', env: { ...process.env } });
  if (result.error) throw new Error(`Could not run nvcc (${nvcc}): ${result.error.message}`);
  if (result.status !== 0) throw new Error(`GPU memory worker build failed with exit code ${result.status}`);
  if (!fs.existsSync(output)) throw new Error(`nvcc reported success but worker was not produced: ${output}`);
  return { nvcc, args, source, output, dryRun: false };
}

function main() {
  let opts;
  try { opts = parseArgs(process.argv.slice(2)); }
  catch (e) { process.stderr.write(`${e.message}\n`); process.exit(2); }
  try {
    const result = buildWorker(opts);
    if (result.dryRun) process.stdout.write(`NVCC: ${result.nvcc}\nARGS: ${result.args.map((x) => JSON.stringify(x)).join(' ')}\n`);
    else process.stdout.write(`Built persistent Morpheus GPU memory worker: ${result.output}\n`);
  } catch (e) {
    process.stderr.write(`GPU worker build error: ${e.message || e}\n`);
    process.exit(1);
  }
}

if (require.main === module) main();
module.exports = { parseArgs, workerBuildArgs, buildWorker };
