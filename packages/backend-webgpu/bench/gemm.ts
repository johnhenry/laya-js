/**
 * GEMM throughput: linear x[M,K] · w[N,K]ᵀ (ModernBERT-large MLP-in shape:
 * 16·128 tokens × 1024 → 3072) and batched matmul [16,128,1024]·[16,1024,128].
 * Usage: node --conditions=source bench/gemm.ts [--scan]
 */
import { createWebGpuBackend, GEMM_DEFAULT, type GemmConfig, type WebGpuTensor } from "../src/index.ts";

const b = await createWebGpuBackend();
const { limits: _l, ...info } = b.adapterInfo;
console.log("adapter:", JSON.stringify({ ...info, features: info.features.filter((f) => f.startsWith("shader") || f.startsWith("subgroup")) }));
const rnd = (n: number, s = 1) => Float32Array.from({ length: n }, () => (Math.random() * 2 - 1) * s);

async function time(label: string, flops: number, fn: () => WebGpuTensor) {
  for (let i = 0; i < 3; i++) b.dispose(fn());
  await b.sync();
  const iters = 20;
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) b.dispose(fn());
  await b.sync();
  const ms = (performance.now() - t0) / iters;
  console.log(`${label}: ${ms.toFixed(3)} ms  ${(flops / ms / 1e6).toFixed(0)} GFLOP/s`);
}

const M = 16 * 128, K = 1024, N = 3072;
const x32 = b.fromHost({ dtype: "f32", shape: [M, K], data: rnd(M * K) });
const w32 = b.fromHost({ dtype: "f32", shape: [N, K], data: rnd(N * K, 0.03) });
const q32 = b.fromHost({ dtype: "f32", shape: [16, 128, 1024], data: rnd(16 * 128 * 1024) });
const k32 = b.fromHost({ dtype: "f32", shape: [16, 1024, 128], data: rnd(16 * 1024 * 128) });
const dts = b.supports("f16") ? (["f32", "f16"] as const) : (["f32"] as const);
const T = (t: WebGpuTensor, dt: "f32" | "f16") => (dt === "f32" ? t : b.cast(t, dt));
const configs: [string, GemmConfig][] = [["default", GEMM_DEFAULT]];
if (process.argv.includes("--scan")) {
  for (const [TM, TN, WX, WY] of [[4, 8, 4, 32], [8, 4, 8, 16], [8, 4, 8, 8], [4, 4, 8, 8]] as const)
    configs.push([`direct ${TM}x${TN} wg ${WX}x${WY}`, { ...GEMM_DEFAULT, direct: { TM, TN, WX, WY } }]);
  for (const t of [{ BM: 64, BN: 64, BK: 32, TM: 4, TN: 4 }, { BM: 128, BN: 128, BK: 16, TM: 8, TN: 8 }])
    configs.push([`tiled ${t.BM}x${t.BN}x${t.BK}/${t.TM}x${t.TN}`, { tiled: t, direct: null, skinny: [] }]);
}
for (const [name, cfg] of configs) {
  b.gemmConfig = cfg;
  for (const dt of dts) {
    const x = T(x32, dt), w = T(w32, dt);
    await time(`${name} linear [${M},${K}]x[${N},${K}]T ${dt}`, 2 * M * N * K, () => b.linear(x, w));
    if (name === "default") {
      const xs = b.slice(x, [0, 0], [33, K]);
      await time(`${name} linear [33,${K}]x[${N},${K}]T ${dt} (skinny; weight ${((N * K * (dt === "f16" ? 2 : 4)) / 1e6).toFixed(1)} MB)`, 2 * 33 * N * K, () => b.linear(xs, w));
      const q = T(q32, dt), k = T(k32, dt);
      await time(`${name} matmul [16,128,1024]x[16,1024,128] ${dt}`, 2 * 16 * 128 * 128 * 1024, () => b.matmul(q, k));
    }
  }
}
b.destroy();
