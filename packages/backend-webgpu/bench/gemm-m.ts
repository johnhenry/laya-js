/**
 * Linear latency vs M for the ModernBERT-large shapes (f16), per GEMM config.
 * Configs are interleaved per M (short bursts) so GPU thermal state affects
 * them alike. Usage:
 *   [MS=16,33,64,93,128,...] node --conditions=source bench/gemm-m.ts
 */
import { createWebGpuBackend, GEMM_DEFAULT, type GemmConfig, type WebGpuTensor } from "../src/index.ts";

const b = await createWebGpuBackend({ ...(process.env.SGMAT === "1" ? { subgroupMatrix: true } : {}) } as never);
const Ms = (process.env.MS ?? "16,33,64,93,128,192,279,384,512,1024,1488,2048").split(",").map(Number);
const shapes: [number, number][] = [[3072, 1024], [1024, 1024], [5248, 1024], [1024, 2624]];
const rnd = (n: number, s = 1) => Float32Array.from({ length: n }, () => (Math.random() * 2 - 1) * s);
const f16 = async (shape: number[], s = 1) => {
  const t = await b.fromHost({ dtype: "f32", shape, data: rnd(shape.reduce((a, c) => a * c, 1), s) });
  const h = b.cast(t, "f16");
  b.dispose(t);
  return h;
};
const W = await Promise.all(shapes.map(([N, K]) => f16([N, K], 0.03)));
const maxM = Math.max(...Ms);
const X = await Promise.all([1024, 2624].map((K) => f16([maxM, K])));

const configs: [string, GemmConfig][] = [["default", GEMM_DEFAULT]];
const extra = (process.env.CONFIGS ?? "direct,tiled").split(",");
if (extra.includes("direct")) configs.push(["direct", { ...GEMM_DEFAULT, sg: null, skinny: [] }]);
if (extra.includes("tiled")) configs.push(["tiled", { ...GEMM_DEFAULT, sg: null, skinny: [], direct: null }]);
if (extra.includes("skinny")) configs.push(["skinny-any", { ...GEMM_DEFAULT, sg: null, skinny: [{ maxM: 1e9, WX: 8, TN: 4, WY: 8, KS: 4, KP4: 0 }] }]);
if (extra.includes("nosg")) configs.push(["no-sg", { ...GEMM_DEFAULT, sg: null }]);
for (const v of extra.filter((e) => e.startsWith("sg:"))) {
  // sg:BMxBNxBK/WMxWN
  const [t, w] = v.slice(3).split("/");
  const [BM, BN, BK] = t!.split("x").map(Number), [WM, WN] = w!.split("x").map(Number);
  configs.push([v, { ...GEMM_DEFAULT, skinny: [], sg: [{ minM: 0, BM: BM!, BN: BN!, BK: BK!, WM: WM!, WN: WN! }] }]);
}

async function time(fn: () => WebGpuTensor, iters = 10): Promise<number> {
  b.dispose(fn());
  await b.sync();
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) b.dispose(fn());
  await b.sync();
  return (performance.now() - t0) / iters;
}

console.log(`| M | ${configs.map(([n]) => n).join(" | ")} |  (sum of 4 Linears, ms; GFLOP/s of best)`);
for (const M of Ms) {
  // 3 interleaved rounds, best of each config (reduces DVFS/thermal bias)
  const row = configs.map(() => Infinity);
  for (let round = 0; round < 3; round++) {
    for (let c = 0; c < configs.length; c++) {
      b.gemmConfig = configs[c]![1];
      let tot = 0;
      const per: string[] = [];
      for (let s = 0; s < shapes.length; s++) {
        const [N, K] = shapes[s]!;
        const x = b.slice(X[K === 1024 ? 0 : 1]!, [0, 0], [M, K]);
        const ms = await time(() => b.linear(x, W[s]!));
        tot += ms;
        per.push(`${N}x${K}: ${ms.toFixed(3)} ms ${((2 * M * N * K) / ms / 1e9).toFixed(2)} TF/s`);
        b.dispose(x);
      }
      if (process.env.PERSHAPE && round === 2) console.log(`  M=${M} ${configs[c]![0]}: ${per.join(", ")}`);
      row[c] = Math.min(row[c]!, tot);
    }
  }
  const flops = shapes.reduce((a, [N, K]) => a + 2 * M * N * K, 0);
  const best = Math.min(...row);
  console.log(`| ${M} | ${row.map((r) => r.toFixed(3)).join(" | ")} | ${(flops / best / 1e6).toFixed(0)} |`);
  await new Promise((r) => setTimeout(r, 2000));
}
b.destroy();
