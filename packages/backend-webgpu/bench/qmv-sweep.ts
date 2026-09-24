/**
 * Sweep of quantized-Linear kernel configs (tuning aid): for each Laya
 * Linear shape and M, times fp16 (default kernel choice) against
 * quantizedLinear with each candidate `qmv` config (and the 0.4.0 choice,
 * `quant: null`), after checking each candidate against the 0.4.0 result.
 *
 *   [MODEL=english|multilingual|both] [MS=1,3,8,16] [BITS=8,4] [CFGS='TK,NR,R,C,MT,sub;...'] [COOL=1] \
 *     node --conditions=source bench/qmv-sweep.ts
 */
import { disposeQuantized, packQuantized, quantizedLinear, uploadQuantized, type HostQuantized, type QuantBits } from "@johnhenry/tensor-backend";
import { createWebGpuBackend, GEMM_DEFAULT, type QmvGemmConfig, type WebGpuTensor } from "../src/index.ts";

const env = process.env;
const b = await createWebGpuBackend();
const shapesBy: Record<string, [string, number, number][]> = {
  english: [["qkv", 3072, 1024], ["o", 1024, 1024], ["wi", 5248, 1024], ["mlp-o", 1024, 2624]],
  multilingual: [["qkv", 2304, 768], ["o", 768, 768], ["wi", 2304, 768], ["mlp-o", 768, 1152]],
};
const models = env.MODEL === "both" ? ["english", "multilingual"] : [env.MODEL ?? "english"];
const Ms = (env.MS ?? "1,3,8,16").split(",").map(Number);
const bitsL = (env.BITS ?? "8,4").split(",").map(Number) as QuantBits[];
const cool = Number(env.COOL ?? 1) * 1000;
const cfgs: QmvGemmConfig[] = (env.CFGS ?? "32,4,4,8,8,1;32,4,4,16,8,1;32,2,4,8,8,1;32,4,2,8,8,1;32,4,4,8,8,0").split(";").map((c) => {
  const [TK, NR, R, C, MT, sub] = c.split(",").map(Number) as number[];
  return { TK: TK!, NR: NR!, R: R!, C: C!, MT: MT!, sub: !!sub };
});
const rnd = (n: number, s = 1) => Float32Array.from({ length: n }, () => (Math.random() * 2 - 1) * s);
console.log(`subgroupSize ${b.subgroupSize}`);

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
const read = async (t: WebGpuTensor) => (await b.read(b.cast(t, "f32"))).data as Float32Array;
const base = { ...GEMM_DEFAULT, quant: null };
const only = (c: QmvGemmConfig) => ({ ...GEMM_DEFAULT, quant: { qmv: [{ ...c, maxM: 1e9 }], sg: null } });
const header = `| Linear | M | q | fp16 | old | ${cfgs.map((c) => `${c.TK}x${c.NR}x${c.R}/${c.C}/${c.MT}${c.sub ? "s" : ""}`).join(" | ")} |`;
console.log(header);
for (const model of models) {
  for (const [name, N, K] of shapesBy[model]!) {
    const w16 = b.cast(await b.fromHost({ dtype: "f32", shape: [N, K], data: rnd(N * K, 0.05) }), "f16");
    for (const bits of bitsL) {
      const qt = await uploadQuantized(b, hostQ(N, K, bits), "f16");
      for (const M of Ms) {
        await new Promise((r) => setTimeout(r, cool));
        const x = b.cast(await b.fromHost({ dtype: "f32", shape: [M, K], data: rnd(M * K) }), "f16");
        b.gemmConfig = base;
        const ref = await read(quantizedLinear(b, x, qt));
        const errs: number[] = [];
        for (const c of cfgs) {
          b.gemmConfig = only(c);
          const got = await read(quantizedLinear(b, x, qt));
          let e = 0, mag = 0;
          for (let i = 0; i < got.length; i++) (e = Math.max(e, Math.abs(got[i]! - ref[i]!))), (mag = Math.max(mag, Math.abs(ref[i]!)));
          errs.push(e / mag);
        }
        const flop = 2 * M * N * K;
        const iters = Math.max(10, Math.min(400, Math.round(Number(env.WORK ?? 2e10) / flop)));
        const best = new Array(cfgs.length + 2).fill(Infinity);
        for (let r = 0; r < Number(env.ROUNDS ?? 5); r++) {
          b.gemmConfig = GEMM_DEFAULT;
          best[0] = Math.min(best[0], await time(() => b.linear(x, w16), iters));
          b.gemmConfig = base;
          best[1] = Math.min(best[1], await time(() => quantizedLinear(b, x, qt), iters));
          for (let c = 0; c < cfgs.length; c++) {
            b.gemmConfig = only(cfgs[c]!);
            best[c + 2] = Math.min(best[c + 2], await time(() => quantizedLinear(b, x, qt), iters));
          }
        }
        const g = best.map((ms) => flop / ms / 1e6);
        console.log(`| ${model} ${name} | ${M} | q${bits} | ${g[0].toFixed(0)} | ${g[1].toFixed(0)} | ${g.slice(2).map((v, i) => `${v.toFixed(0)}${errs[i]! > 5e-3 ? `!${errs[i]!.toExponential(1)}` : ""}`).join(" | ")} |`);
        b.dispose(x);
      }
      disposeQuantized(b, qt);
    }
    b.dispose(w16);
  }
}
b.destroy();
