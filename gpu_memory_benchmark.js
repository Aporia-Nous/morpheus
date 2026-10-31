#!/usr/bin/env node
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');
const { performance } = require('perf_hooks');

function parseArgs(argv) {
  const out = { worker: process.env.MORPHEUS_GPU_MEMORY_WORKER || null, sizes: [1000, 10000, 100000], dim: 64, k: 8, iterations: 20, warmup: 3 };
  for (let i = 0; i < argv.length; ++i) {
    if (argv[i] === '--worker' && argv[i + 1]) out.worker = argv[++i];
    else if (argv[i] === '--sizes' && argv[i + 1]) out.sizes = argv[++i].split(',').map(Number).filter((n) => Number.isInteger(n) && n > 0);
    else if (argv[i] === '--dim' && argv[i + 1]) out.dim = Number(argv[++i]);
    else if (argv[i] === '--k' && argv[i + 1]) out.k = Number(argv[++i]);
    else if (argv[i] === '--iterations' && argv[i + 1]) out.iterations = Number(argv[++i]);
    else if (argv[i] === '--warmup' && argv[i + 1]) out.warmup = Number(argv[++i]);
    else throw new Error(`Unknown argument: ${argv[i]}`);
  }
  if (!out.worker) throw new Error('Provide --worker <path> or MORPHEUS_GPU_MEMORY_WORKER.');
  return out;
}

function rng(seed = 0x51f15e) {
  let x = seed >>> 0;
  return () => {
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    return (x >>> 0) / 4294967296;
  };
}

function normalizeRows(matrix, rows, dim) {
  for (let r = 0; r < rows; ++r) {
    const off = r * dim;
    let ss = 0;
    for (let c = 0; c < dim; ++c) { const v = matrix[off + c]; ss += v * v; }
    const inv = 1 / (Math.sqrt(ss) + 1e-12);
    for (let c = 0; c < dim; ++c) matrix[off + c] *= inv;
  }
}

function makeMatrix(rows, dim, random) {
  const m = new Float32Array(rows * dim);
  for (let i = 0; i < m.length; ++i) m[i] = random() * 2 - 1;
  normalizeRows(m, rows, dim);
  return m;
}

function makeQuery(dim, random) {
  const q = new Float32Array(dim);
  let ss = 0;
  for (let i = 0; i < dim; ++i) { q[i] = random() * 2 - 1; ss += q[i] * q[i]; }
  const inv = 1 / (Math.sqrt(ss) + 1e-12);
  for (let i = 0; i < dim; ++i) q[i] *= inv;
  return q;
}

function cpuTopK(matrix, rows, dim, q, k) {
  const best = [];
  for (let r = 0; r < rows; ++r) {
    const off = r * dim;
    let score = 0;
    for (let c = 0; c < dim; ++c) score += matrix[off + c] * q[c];
    if (best.length < k) {
      best.push({ row: r, score });
      best.sort((a, b) => b.score - a.score || a.row - b.row);
    } else if (score > best[k - 1].score) {
      best[k - 1] = { row: r, score };
      for (let i = k - 1; i > 0 && (best[i].score > best[i - 1].score || (best[i].score === best[i - 1].score && best[i].row < best[i - 1].row)); --i) {
        const t = best[i - 1]; best[i - 1] = best[i]; best[i] = t;
      }
    }
  }
  return best;
}

class WorkerClient {
  constructor(exe) { this.exe = exe; this.proc = null; this.buf = ''; this.pending = []; this.ready = null; }
  async start() {
    this.proc = spawn(this.exe, [], { stdio: ['pipe', 'pipe', 'inherit'], windowsHide: true });
    this.proc.stdout.setEncoding('utf8');
    this.ready = new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('worker READY timeout')), 5000);
      this._readyResolve = (v) => { clearTimeout(timer); resolve(v); };
    });
    this.proc.stdout.on('data', (chunk) => {
      this.buf += chunk;
      for (;;) {
        const n = this.buf.indexOf('\n'); if (n < 0) break;
        const line = this.buf.slice(0, n).replace(/\r$/, ''); this.buf = this.buf.slice(n + 1);
        if (line.startsWith('READY ')) { if (this._readyResolve) this._readyResolve(line.slice(6)); this._readyResolve = null; continue; }
        const p = this.pending.shift(); if (!p) continue;
        clearTimeout(p.timer);
        if (line.startsWith('ERR ')) p.reject(new Error(line.slice(4))); else p.resolve(line);
      }
    });
    this.proc.on('error', (e) => { for (const p of this.pending.splice(0)) p.reject(e); });
    this.proc.on('exit', (code) => { if (code && this.pending.length) for (const p of this.pending.splice(0)) p.reject(new Error(`worker exited ${code}`)); });
    return this.ready;
  }
  request(line, timeout = 15000) {
    return new Promise((resolve, reject) => {
      const item = { resolve, reject, timer: null };
      item.timer = setTimeout(() => reject(new Error(`worker request timeout: ${line.slice(0, 24)}`)), timeout);
      this.pending.push(item);
      this.proc.stdin.write(line + '\n');
    });
  }
  async close() { try { await this.request('EXIT', 1000); } catch {} try { this.proc.kill(); } catch {} }
}

function parseResult(line) {
  if (!line.startsWith('RESULT ')) throw new Error(`Unexpected worker response: ${line}`);
  return line.slice(7).split(',').filter(Boolean).map((p) => {
    const [row, score] = p.split(':');
    return { row: Number(row), score: Number(score) };
  });
}

function percentile(values, p) {
  const a = [...values].sort((x, y) => x - y);
  const i = Math.min(a.length - 1, Math.max(0, Math.ceil(p * a.length) - 1));
  return a[i];
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const random = rng();
  const client = new WorkerClient(path.resolve(opts.worker));
  const device = await client.start();
  console.log(`GPU worker: ${device}`);
  console.log(`dim=${opts.dim} k=${opts.k} iterations=${opts.iterations}`);

  const results = [];
  try {
    for (const rows of opts.sizes) {
      const matrix = makeMatrix(rows, opts.dim, random);
      const q = makeQuery(opts.dim, random);
      const file = path.join(os.tmpdir(), `morpheus-memory-bench-${process.pid}-${rows}.f32`);
      fs.writeFileSync(file, Buffer.from(matrix.buffer, matrix.byteOffset, matrix.byteLength));

      const portable = file.replace(/\\/g, '/');
      const init0 = performance.now();
      const initReply = await client.request(`INIT "${portable}" ${rows} ${opts.dim}`, 60000);
      const initMs = performance.now() - init0;
      if (!initReply.startsWith('OK INIT ')) throw new Error(initReply);

      // CPU warmup + benchmark.
      let expected = cpuTopK(matrix, rows, opts.dim, q, opts.k);
      const cpuSamples = [];
      for (let i = 0; i < opts.iterations; ++i) {
        const t0 = performance.now();
        expected = cpuTopK(matrix, rows, opts.dim, q, opts.k);
        cpuSamples.push(performance.now() - t0);
      }

      const csv = Array.from(q, (x) => Number(x).toPrecision(9)).join(',');
      for (let i = 0; i < opts.warmup; ++i) await client.request(`QUERY ${opts.k} ${csv}`);
      const gpuSamples = [];
      let actual = null;
      for (let i = 0; i < opts.iterations; ++i) {
        const t0 = performance.now();
        actual = parseResult(await client.request(`QUERY ${opts.k} ${csv}`));
        gpuSamples.push(performance.now() - t0);
      }

      const expectedIds = expected.map((x) => x.row);
      const actualIds = actual.map((x) => x.row);
      if (expectedIds.join(',') !== actualIds.join(',')) {
        throw new Error(`top-k mismatch at ${rows} rows: CPU=${expectedIds} GPU=${actualIds}`);
      }
      let maxScoreDelta = 0;
      for (let i = 0; i < expected.length; ++i) maxScoreDelta = Math.max(maxScoreDelta, Math.abs(expected[i].score - actual[i].score));
      if (maxScoreDelta > 2e-5) throw new Error(`score mismatch at ${rows} rows: max delta ${maxScoreDelta}`);

      const cpuAvg = cpuSamples.reduce((a, b) => a + b, 0) / cpuSamples.length;
      const gpuAvg = gpuSamples.reduce((a, b) => a + b, 0) / gpuSamples.length;
      const rowResult = {
        rows,
        dim: opts.dim,
        matrixMB: matrix.byteLength / 1048576,
        initMs,
        cpuAvgMs: cpuAvg,
        gpuAvgMs: gpuAvg,
        gpuP95Ms: percentile(gpuSamples, 0.95),
        speedup: cpuAvg / gpuAvg,
        maxScoreDelta,
        topKExact: true,
      };
      results.push(rowResult);
      console.log(`${rows.toLocaleString()} rows | init ${initMs.toFixed(2)} ms | CPU ${cpuAvg.toFixed(3)} ms | GPU ${gpuAvg.toFixed(3)} ms | p95 ${rowResult.gpuP95Ms.toFixed(3)} ms | ${rowResult.speedup.toFixed(2)}x | exact top-k`);
      try { fs.rmSync(file, { force: true }); } catch {}
    }
  } finally {
    await client.close();
  }

  console.log('\nJSON');
  console.log(JSON.stringify({ device, options: opts, results }, null, 2));
}

main().catch((e) => { console.error(e.stack || e); process.exit(1); });
