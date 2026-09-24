/**
 * Microbenchmarks for backend-mlx; `python_ref.py` runs the same graphs in
 * Python MLX. Run under the GPU lock:
 *   node --conditions=source bench/ops.bench.ts   |   bun --conditions=source bench/ops.bench.ts
 * Timings: median of N warmed iterations, graph build + eval (+ no readback).
 */
import { host } from "@johnhenry/tensor-backend";
import { createMlxBackend, type MlxTensor } from "../src/index.ts";

const b = createMlxBackend();
console.log(`runtime ${b.info.runtime} | lib ${b.info.libPath} | ${b.info.mlxcAbi}`);

function median(xs: number[]): number {
  const s = [...xs].sort((a, c) => a - c);
  return s[s.length >> 1]!;
}
function time(name: string, iters: number, fn: () => void, warm = 10): number {
  for (let i = 0; i < warm; i++) fn();
  const ts: number[] = [];
  for (let i = 0; i < iters; i++) {
    const t0 = performance.now();
    fn();
    ts.push(performance.now() - t0);
  }
  const m = median(ts);
  console.log(`${name.padEnd(48)} ${m.toFixed(3)} ms`);
  return m;
}
const rnd = (n: number, s = 0.02) => Array.from({ length: n }, () => (Math.random() * 2 - 1) * s);
const up = (shape: number[], s = 0.02, dtype: "f16" | "f32" = "f16") => {
  const n = shape.reduce((a, c) => a * c, 1);
  return b.fromHost(host(dtype, shape, rnd(n, s)));
};

// ---- FFI overhead ------------------------------------------------------------
const a = await up([4], 1, "f32"), c = await up([4], 1, "f32");
{
  const N = 10000;
  const us = time("10k tiny adds: build+dispose (per op, us)", 5, () => {
    for (let i = 0; i < N; i++) b.dispose(b.add(a, c));
  }) * 1000 / N;
  console.log(`  -> ${us.toFixed(3)} us/op`);
  const us2 = time("10k chained adds: build+eval (per op, us)", 5, () => {
    b.scope(() => {
      let x = a;
      for (let i = 0; i < N; i++) x = b.add(x, c);
      b.flush(x);
      return null;
    });
  }) * 1000 / N;
  console.log(`  -> ${us2.toFixed(3)} us/op`);
  time("1 tiny add + read (round trip)", 200, () => {
    const y = b.add(a, c);
    b.readSync(y);
    b.dispose(y);
  });
}

// ---- matmul / sdpa at encoder shapes ------------------------------------------------
{
  const x = await up([16, 128, 1024], 1), w = await up([1024, 1024]);
  const ms = time("linear f16 [16,128,1024]x[1024,1024]^T", 50, () => {
    const y = b.linear(x, w);
    b.flush(y);
    b.dispose(y);
  });
  console.log(`  -> ${(2 * 16 * 128 * 1024 * 1024 / ms / 1e9).toFixed(2)} TFLOP/s`);
  const B = 16, H = 16, L = 128, D = 64;
  const q = await up([B, H, L, D], 1), k = await up([B, H, L, D], 1), v = await up([B, H, L, D], 1);
  const mask = await b.fromHost(host("bool", [B, 1, 1, L], Array.from({ length: B * L }, (_, i) => (i % L < 100 ? 1 : 0))));
  time("sdpa f16 [16,16,128,64] bool mask", 50, () => {
    const y = b.sdpa(q, k, v, mask, D ** -0.5);
    b.flush(y);
    b.dispose(y);
  });
}

// ---- one ModernBERT-large encoder layer ----------------------------------------------
{
  const Dm = 1024, H = 16, Dh = 64, I = 2624;
  const W = { attnNorm: await up([Dm], 1), Wqkv: await up([3 * Dm, Dm]), Wo: await up([Dm, Dm]), mlpNorm: await up([Dm], 1), Wi: await up([2 * I, Dm]), Wo2: await up([Dm, I]) };
  const layer = (x: MlxTensor, mask: MlxTensor): MlxTensor =>
    b.scope(() => {
      const [B, L] = x.shape as [number, number, number];
      const h = b.layerNorm(x, W.attnNorm, null, 1e-5);
      const qkv = b.transpose(b.reshape(b.linear(h, W.Wqkv), [B, L, 3, H, Dh]), [2, 0, 3, 1, 4]);
      const [q, k, v] = b.split(qkv, 3, 0).map((t) => b.reshape(t, [B, H, L, Dh])) as [MlxTensor, MlxTensor, MlxTensor];
      const att = b.sdpa(b.rope(q, 160000), b.rope(k, 160000), v, mask, Dh ** -0.5);
      const x1 = b.add(x, b.linear(b.reshape(b.transpose(att, [0, 2, 1, 3]), [B, L, Dm]), W.Wo));
      const m = b.linear(b.geglu(b.linear(b.layerNorm(x1, W.mlpNorm, null, 1e-5), W.Wi)), W.Wo2);
      return b.add(x1, m);
    });
  for (const [B, L] of [[1, 32], [16, 128]] as const) {
    const x = await up([B, L, Dm], 1);
    const mask = await b.fromHost(host("bool", [B, 1, 1, L], new Array(B * L).fill(1)));
    time(`encoder layer eager B=${B} L=${L}`, 30, () => {
      const y = layer(x, mask);
      b.flush(y);
      b.dispose(y);
    });
    const compiled = b.compile(layer);
    time(`encoder layer compiled B=${B} L=${L}`, 30, () => {
      const y = compiled(x, mask);
      b.flush(y);
      b.dispose(y);
    });
    // graph-build cost alone (no eval)
    time(`  graph build only (JS+FFI) B=${B} L=${L}`, 30, () => {
      b.dispose(layer(x, mask));
    });
  }
}
