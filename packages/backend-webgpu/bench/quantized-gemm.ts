/**
 * GFLOP/s of quantizedLinear (q8 symmetric / q4 affine, groups of 64,
 * dequantized in the tile load, f32 accumulation) vs the fp16 Linear, per
 * Laya Linear shape and M = B·L, with the default kernel choice for each M
 * (skinny for M ≤ 64, subgroup-matrix above when available, else direct).
 * f16 activations. Interleaved per (shape, M) in short bursts, best of 3
 * rounds, COOL seconds idle between M values (the M2 is fanless).
 *
 *   [MODEL=english|multilingual|both] [MS=1,16,64,93,512,1488] [COOL=2] \
 *     node --conditions=source bench/quantized-gemm.ts
 */
import { disposeQuantized, packQuantized, quantizedLinear, uploadQuantized, type HostQuantized, type QuantBits } from "@johnhenry/tensor-backend";
import { createWebGpuBackend, type WebGpuTensor } from "../src/index.ts";

const env = process.env;
const b = await createWebGpuBackend();
const shapesBy: Record<string, [string, number, number][]> = {
  english: [["qkv", 3072, 1024], ["o", 1024, 1024], ["wi", 5248, 1024], ["mlp-o", 1024, 2624]],
  multilingual: [["qkv", 2304, 768], ["o", 768, 768], ["wi", 2304, 768], ["mlp-o", 768, 1152]],
};
const models = env.MODEL === "both" ? ["english", "multilingual"] : [env.MODEL ?? "english"];
const Ms = (env.MS ?? "1,16,64,93,512,1488").split(",").map(Number);
const cool = Number(env.COOL ?? 2) * 1000;
const rnd = (n: number, s = 1) => Float32Array.from({ length: n }, () => (Math.random() * 2 - 1) * s);

function hostQ(N: number, K: number, bits: QuantBits): HostQuantized {
  const G = K / 64;
  const q = Int32Array.from({ length: N * K }, () => (bits === 8 ? Math.floor(Math.random() * 255) - 127 : Math.floor(Math.random() * 16)));
  const s = Float16Array.from({ length: N * G }, () => 1e-3 + Math.random() * 1e-3);
  return {
    shape: [N, K], bits, groupSize: 64, mode: bits === 8 ? "symmetric" : "affine", data: packQuantized(q, bits),
    scales: { dtype: "f16", shape: [N, G], data: s },
    biases: bits === 4 ? { dtype: "f16", shape: [N, G], data: s.map((v) => -8 * v) } : null,
  };
}

async function time(fn: () => WebGpuTensor, iters: number): Promise<number> {
  b.dispose(fn());
  await b.sync();
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) b.dispose(fn());
  await b.sync();
  return (performance.now() - t0) / iters;
}

console.log(`adapter: ${b.adapterInfo.description ?? b.adapterInfo.vendor} | f16 ${b.hasF16} | subgroup matrix ${b.hasSubgroupMatrix}`);
const rows: string[] = [];
for (const model of models) {
  for (const [name, N, K] of shapesBy[model]!) {
    const w16 = b.cast(await b.fromHost({ dtype: "f32", shape: [N, K], data: rnd(N * K, 0.05) }), "f16");
    const q8 = await uploadQuantized(b, hostQ(N, K, 8), "f16");
    const q4 = await uploadQuantized(b, hostQ(N, K, 4), "f16");
    for (const M of Ms) {
      await new Promise((r) => setTimeout(r, cool));
      const x = b.cast(await b.fromHost({ dtype: "f32", shape: [M, K], data: rnd(M * K) }), "f16");
      const flop = 2 * M * N * K;
      const iters = Math.max(3, Math.min(50, Math.round(2e9 / flop)));
      const best = [Infinity, Infinity, Infinity];
      for (let r = 0; r < 3; r++) {
        best[0] = Math.min(best[0]!, await time(() => b.linear(x, w16), iters));
        best[1] = Math.min(best[1]!, await time(() => quantizedLinear(b, x, q8), iters));
        best[2] = Math.min(best[2]!, await time(() => quantizedLinear(b, x, q4), iters));
      }
      const g = best.map((ms) => flop / ms / 1e6);
      rows.push(`| ${model} ${name} [${N}×${K}] | ${M} | ${g[0]!.toFixed(0)} | ${g[1]!.toFixed(0)} (${(g[1]! / g[0]!).toFixed(2)}×) | ${g[2]!.toFixed(0)} (${(g[2]! / g[0]!).toFixed(2)}×) |`);
      console.error(rows[rows.length - 1]);
      b.dispose(x);
    }
    b.dispose(w16);
    disposeQuantized(b, q8);
    disposeQuantized(b, q4);
  }
}
console.log("\n| Linear | M | fp16 GFLOP/s | q8 GFLOP/s | q4 GFLOP/s |\n|---|---:|---:|---:|---:|\n" + rows.join("\n"));
b.destroy();
