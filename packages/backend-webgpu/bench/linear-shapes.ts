/**
 * GFLOP/s per Linear shape of the real Laya checkpoints (f16 storage, f32
 * accumulation), for M = B·L. WebGPU GEMM configs (and optionally MLX's
 * `x @ w.T`) are interleaved per (shape, M) in short bursts, best of 3
 * rounds, with a short idle between M values (the M2 is fanless).
 *
 *   [MODEL=english|multilingual|both] [MS=16,33,48,64,93,128,256,279,512,1488,2048,4096]
 *   [CONFIGS=default,base,...] [MLX=1] [COOL=2] node --conditions=source bench/linear-shapes.ts
 *
 * CONFIGS: "default" (GEMM_DEFAULT), "base" (the 0.2.0 configuration),
 * "main" (the unmodified 0.2.0 backend, from a copy of main's src/ in
 * ../.base/src — see bench/grid.ts),
 * "nosplit" (no split-K), "sg:BMxBNxBK/WMxWN[/S][/db][/narrow][/block][/pN]" (a single
 * subgroup-matrix config for every M, optional split-K S, db = double-buffered), "skinny" (skinny kernel for every M).
 */
import { createWebGpuBackend, GEMM_DEFAULT, GEMM_V020, type GemmConfig, type WebGpuBackend, type WebGpuTensor } from "../src/index.ts";

const env = process.env;
const b = await createWebGpuBackend();
const shapesBy: Record<string, [string, number, number][]> = {
  english: [["qkv", 3072, 1024], ["o", 1024, 1024], ["wi", 5248, 1024], ["mlp-o", 1024, 2624]],
  multilingual: [["qkv", 2304, 768], ["o", 768, 768], ["wi", 2304, 768], ["mlp-o", 768, 1152]],
};
const models = env.MODEL === "both" ? ["english", "multilingual"] : [env.MODEL ?? "english"];
const Ms = (env.MS ?? "16,33,48,64,93,128,256,279,512,1488,2048,4096").split(",").map(Number);
const cool = Number(env.COOL ?? 2) * 1000;
const rnd = (n: number, s = 1) => Float32Array.from({ length: n }, () => (Math.random() * 2 - 1) * s);

const configs: [string, GemmConfig][] = [];
let main: WebGpuBackend | null = null;
for (const c of (env.CONFIGS ?? "main,default").split(",")) {
  if (c === "main") {
    const m = await import(new URL("../.base/src/index.ts", import.meta.url).href);
    main = await m.createWebGpuBackend();
    configs.push([c, main!.gemmConfig]);
  } else if (c === "default") configs.push([c, GEMM_DEFAULT]);
  else if (c === "base") configs.push([c, GEMM_V020]);
  else if (c === "nosplit") configs.push([c, { ...GEMM_DEFAULT, sg: GEMM_DEFAULT.sg?.map((s) => ({ ...s, splitK: undefined })) }]);
  else if (c === "skinny") configs.push([c, { ...GEMM_DEFAULT, sg: null, skinny: [{ maxM: 1e9, WX: 8, TN: 4, WY: 8, KS: 4, KP4: 0 }] }]);
  else if (c.startsWith("sg:")) {
    // sg:BMxBNxBK/WMxWN[/S][/db]
    const [t, w, ...rest] = c.slice(3).split("/");
    const [BM, BN, BK] = t!.split("x").map(Number), [WM, WN] = w!.split("x").map(Number);
    const S = rest.find((r) => /^\d+$/.test(r));
    configs.push([c, { ...GEMM_DEFAULT, skinny: [], sg: [{ minM: 0, BM: BM!, BN: BN!, BK: BK!, WM: WM!, WN: WN!, db: rest.includes("db"), wide: !rest.includes("narrow"), ...(rest.includes("block") ? { epi: "block" as const } : {}), ...(rest.find((r) => r.startsWith("p")) ? { pad: Number(rest.find((r) => r.startsWith("p"))!.slice(1)) } : {}), ...(S ? { splitK: [{ S: Number(S) }] } : {}) }] }]);
  } else throw new Error(`unknown config ${c}`);
}

let mlx: { b: any } | null = null;
if (env.MLX === "1") {
  const m = await import("@johnhenry/backend-mlx");
  mlx = { b: m.createMlxBackend() };
}

async function timeGpu(b: WebGpuBackend, fn: () => WebGpuTensor, iters: number): Promise<number> {
  b.dispose(fn());
  await b.sync();
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) b.dispose(fn());
  await b.sync();
  return (performance.now() - t0) / iters;
}
function timeMlx(fn: () => unknown, iters: number): number {
  const mb = mlx!.b;
  const w = fn();
  mb.flush(w);
  mb.dispose(w);
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) {
    const t = fn();
    mb.flush(t);
    mb.dispose(t);
  }
  return (performance.now() - t0) / iters;
}

const names = [...configs.map(([n]) => n), ...(mlx ? ["mlx"] : [])];
const json: Record<string, Record<string, Record<string, number>>> = {};
for (const model of models) {
  const shapes = shapesBy[model]!;
  const maxM = Math.max(...Ms);
  const Ks = [...new Set(shapes.map((s) => s[2]))];
  const up = async (b: WebGpuBackend, shape: number[], s: number) => {
    const t = await b.fromHost({ dtype: "f32", shape, data: rnd(shape.reduce((a, c) => a * c, 1), s) });
    const h = b.cast(t, "f16");
    b.dispose(t);
    return h;
  };
  const W = await Promise.all(shapes.map(([, N, K]) => up(b, [N, K], 0.03)));
  const X = new Map(await Promise.all(Ks.map(async (K) => [K, await up(b, [maxM, K], 1)] as const)));
  const mainW = main ? await Promise.all(shapes.map(([, N, K]) => up(main!, [N, K], 0.03))) : [];
  const mainX = main ? new Map(await Promise.all(Ks.map(async (K) => [K, await up(main!, [maxM, K], 1)] as const))) : new Map();
  const mW = mlx ? await Promise.all(shapes.map(async ([, N, K]) => mlx!.b.cast(await mlx!.b.fromHost({ dtype: "f32", shape: [N, K], data: rnd(N * K, 0.03) }), "f16"))) : [];
  const mX = mlx ? new Map(await Promise.all(Ks.map(async (K) => [K, mlx!.b.cast(await mlx!.b.fromHost({ dtype: "f32", shape: [maxM, K], data: rnd(maxM * K) }), "f16")] as const))) : new Map();
  console.log(`\n### ${model} (GFLOP/s, f16, best of 3 interleaved rounds)\n`);
  console.log(`| M | shape | ${names.join(" | ")} |`);
  console.log(`|---:|---|${names.map(() => "---:").join("|")}|`);
  for (const M of Ms) {
    for (let s = 0; s < shapes.length; s++) {
      const [label, N, K] = shapes[s]!;
      const flops = 2 * M * N * K;
      const iters = Math.max(3, Math.min(50, Math.round(2e10 / flops)));
      const best = names.map(() => Infinity);
      const x = b.slice(X.get(K)!, [0, 0], [M, K]);
      const bx = main ? main.slice(mainX.get(K)!, [0, 0], [M, K]) : null;
      const mx = mlx ? mlx.b.slice(mX.get(K), [0, 0], [M, K]) : null;
      for (let round = 0; round < 3; round++) {
        for (let c = 0; c < configs.length; c++) {
          const ms = configs[c]![0] === "main"
            ? await timeGpu(main!, () => main!.linear(bx!, mainW[s]!), iters)
            : ((b.gemmConfig = configs[c]![1]), await timeGpu(b, () => b.linear(x, W[s]!), iters));
          best[c] = Math.min(best[c]!, ms);
        }
        if (mlx) best[configs.length] = Math.min(best[configs.length]!, timeMlx(() => mlx!.b.linear(mx, mW[s]), iters));
      }
      b.dispose(x);
      if (bx) main!.dispose(bx);
      if (mx) mlx!.b.dispose(mx);
      const gf = best.map((ms) => flops / ms / 1e6);
      names.forEach((n, i) => (((json[model] ??= {})[n] ??= {})[`${label}:${M}`] = gf[i]!));
      console.log(`| ${M} | ${label} ${N}×${K} | ${gf.map((g) => g.toFixed(0)).join(" | ")} |`);
    }
    await new Promise((r) => setTimeout(r, cool));
  }
  for (const t of [...W, ...X.values()]) b.dispose(t);
  for (const t of [...mainW, ...mainX.values()]) main!.dispose(t);
}
main?.destroy();
console.log(JSON.stringify(json));
b.destroy();
