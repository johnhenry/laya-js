/**
 * Per-op timing at ModernBERT-large shapes (B=1, L tokens, H=1024, 16 heads).
 *   node --conditions=source bench/ops.ts [L=33] [f16|f32]
 */
import { createWebGpuBackend, type WebGpuTensor } from "../src/index.ts";

const L = Number(process.argv[2] ?? 33), dt = (process.argv[3] ?? "f16") as "f16" | "f32";
const b = await createWebGpuBackend();
if (process.env.SKINNY) b.gemmConfig = { ...b.gemmConfig, skinny: [{ maxM: 1e9, ...JSON.parse(process.env.SKINNY) }] };
if (process.env.NOSKINNY) b.gemmConfig = { ...b.gemmConfig, skinny: [] };
const rnd = (n: number, s = 1) => Float32Array.from({ length: n }, () => (Math.random() * 2 - 1) * s);
const T = async (shape: number[], s = 1) => b.cast(await b.fromHost({ dtype: "f32", shape, data: rnd(shape.reduce((a, c) => a * c, 1), s) }), dt);
const H = 1024, nh = 16, hd = 64;
const x = await T([1, L, H]);
const w = { qkv: await T([3 * H, H], 0.03), o: await T([H, H], 0.03), wi: await T([5248, H], 0.03), wo: await T([H, 2624], 0.03), ln: await T([H]) };
const bias = await T([3 * H]);
const qkv = b.linear(x, w.qkv);
const q = await T([1, nh, L, hd]);
const mask = await b.fromHost({ dtype: "bool", shape: [1, 1, L, L], data: new Uint8Array(L * L).fill(1) });
const m = await T([1, L, 5248]);
const g = await T([1, L, 2624]);

async function time(name: string, fn: () => WebGpuTensor, bytes = 0) {
  for (let i = 0; i < 5; i++) b.dispose(fn());
  await b.sync();
  const n = 200, t0 = performance.now();
  for (let i = 0; i < n; i++) b.dispose(fn());
  await b.sync();
  const us = ((performance.now() - t0) / n) * 1000;
  console.log(`${name.padEnd(28)} ${us.toFixed(1).padStart(8)} µs${bytes ? `  ${(bytes / us / 1e3).toFixed(1)} GB/s` : ""}`);
  return us;
}
const eb = dt === "f16" ? 2 : 4;
let total = 0;
total += await time("layerNorm", () => b.layerNorm(x, w.ln, null, 1e-5));
total += await time("linear qkv 1024→3072", () => b.linear(x, w.qkv, bias), 3 * H * H * eb);
total += await time("transpose qkv", () => b.transpose(b.reshape(qkv, [1, L, 3, nh, hd]), [2, 0, 3, 1, 4]));
total += 2 * (await time("rope (x2)", () => b.rope(q, 160000)));
total += await time("sdpa", () => b.sdpa(q, q, q, mask, 0.125));
total += await time("transpose att", () => b.transpose(q, [0, 2, 1, 3]));
total += await time("linear o 1024→1024", () => b.linear(x, w.o), H * H * eb);
total += 2 * (await time("add (x2)", () => b.add(x, x)));
total += await time("layerNorm", () => b.layerNorm(x, w.ln, null, 1e-5));
total += await time("linear wi 1024→5248", () => b.linear(x, w.wi), 5248 * H * eb);
total += await time("geglu", () => b.geglu!(m));
total += await time("linear wo 2624→1024", () => b.linear(g, w.wo), 2624 * H * eb);
console.log(`sum per layer ≈ ${total.toFixed(0)} µs → 28 layers ≈ ${((total * 28) / 1000).toFixed(1)} ms (${dt}, L=${L})`);
b.destroy();
