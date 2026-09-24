/**
 * Quantized Linear on every GEMM path (skinny, subgroup-matrix incl. split-K,
 * direct, tiled) and the dequantizing gather, q8/q4 × symmetric/affine, f16
 * and f32 activations, against an f64 JS reference of x · dequant(W)ᵀ + b.
 */
import assert from "node:assert/strict";
import { disposeQuantized, packQuantized, quantizedEmbedding, quantizedLinear, toF32, uploadQuantized, type HostQuantized, type QuantBits, type QuantMode } from "@johnhenry/tensor-backend";
import { createWebGpuBackend, isWebGpuAvailable, GEMM_DEFAULT, type GemmConfig, type WebGpuBackend } from "../src/index.ts";
import { harness, isBun } from "./harness.ts";

// @ts-ignore -- bun types are not installed
const t = harness(isBun ? await import("bun:test") : null);
const available = await isWebGpuAvailable();

let seed = 99;
const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);

function randomQuantized(N: number, K: number, bits: QuantBits, mode: QuantMode, g: number): { h: HostQuantized; w: Float64Array } {
  const G = Math.ceil(K / g);
  const lo = mode === "symmetric" ? -(1 << (bits - 1)) : 0, hi = mode === "symmetric" ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;
  const q = Int32Array.from({ length: N * K }, () => lo + Math.floor(rnd() * (hi - lo + 1)));
  const s = Float16Array.from({ length: N * G }, () => (0.2 + rnd()) * (bits === 4 ? 0.01 : 6e-4));
  const b = mode === "affine" ? Float16Array.from({ length: N * G }, () => (rnd() - 0.5) * 0.05) : null;
  const w = new Float64Array(N * K);
  for (let r = 0; r < N; r++)
    for (let k = 0; k < K; k++) {
      const gi = r * G + Math.floor(k / g);
      w[r * K + k] = q[r * K + k]! * s[gi]! + (b ? b[gi]! : 0);
    }
  return {
    w,
    h: { shape: [N, K], bits, groupSize: g, mode, data: packQuantized(q, bits), scales: { dtype: "f16", shape: [N, G], data: s }, biases: b ? { dtype: "f16", shape: [N, G], data: b } : null },
  };
}

const PATHS: Record<string, (sg: boolean) => GemmConfig | null> = {
  skinny: () => ({ ...GEMM_DEFAULT, sg: null, skinny: [{ maxM: 1e9, WX: 16, TN: 4, WY: 4, KS: 4, KP4: 0 }] }),
  "skinny (K panels)": () => ({ ...GEMM_DEFAULT, sg: null, skinny: [{ maxM: 1e9, WX: 8, TN: 4, WY: 8, KS: 2, KP4: 8 }] }),
  "subgroup matrix": (sg) => (sg ? { ...GEMM_DEFAULT, skinny: [], sg: [{ minM: 0, BM: 32, BN: 64, BK: 8, WM: 1, WN: 2, pad: 0 }] } : null),
  "subgroup matrix, split-K": (sg) => (sg ? { ...GEMM_DEFAULT, skinny: [], sg: [{ minM: 0, BM: 32, BN: 64, BK: 8, WM: 1, WN: 2, pad: 0, splitK: [{ S: 2 }] }] } : null),
  direct: () => ({ ...GEMM_DEFAULT, skinny: [], sg: null }),
  tiled: () => ({ ...GEMM_DEFAULT, skinny: [], sg: null, direct: null }),
};

if (!available) {
  t.skip("webgpu quantized", "no WebGPU adapter in this runtime");
} else {
  const created: WebGpuBackend[] = [];
  const probe = await createWebGpuBackend();
  created.push(probe);
  for (const [path, cfgOf] of Object.entries(PATHS)) {
    const gemm = cfgOf(probe.hasSubgroupMatrix);
    if (!gemm) {
      t.skip(`webgpu quantizedLinear: ${path}`, "no subgroup matrices on this adapter");
      continue;
    }
    t.describe(`webgpu quantizedLinear: ${path}`, () => {
      for (const [bits, mode] of [[8, "symmetric"], [4, "affine"], [8, "affine"], [4, "symmetric"]] as const) {
        for (const dtype of ["f16", "f32"] as const) {
          t.it(`q${bits} ${mode}, ${dtype} x, M ∈ {1, 7, 70, 133}, bias and partial last group`, async () => {
            const b = await createWebGpuBackend({ gemm });
            created.push(b);
            if (dtype === "f16" && !b.hasF16) return;
            const N = 72, K = 224; // 224 = 3·64 + 32: partial last group
            const rq = randomQuantized(N, K, bits, mode, 64);
            const q = await uploadQuantized(b, rq.h, dtype);
            assert.equal(q.native, true);
            const bias = Float32Array.from({ length: N }, () => rnd() - 0.5);
            const bt = b.cast(await b.fromHost({ dtype: "f32", shape: [N], data: bias }), dtype);
            for (const M of [1, 7, 70, 133]) {
              const xv = Float32Array.from({ length: M * K }, () => Math.round((rnd() * 2 - 1) * 64) / 64);
              const x = b.cast(await b.fromHost({ dtype: "f32", shape: [M, K], data: xv }), dtype);
              const y = quantizedLinear(b, x, q, bt);
              assert.equal(y.dtype, dtype);
              assert.deepEqual(y.shape, [M, N]);
              const got = toF32(await b.read(y));
              for (let m = 0; m < M; m++)
                for (let n = 0; n < N; n++) {
                  let acc = bias[n]!, mag = Math.abs(bias[n]!);
                  for (let k = 0; k < K; k++) (acc += xv[m * K + k]! * rq.w[n * K + k]!), (mag += Math.abs(xv[m * K + k]! * rq.w[n * K + k]!));
                  const err = Math.abs(got[m * N + n]! - acc);
                  const tol = dtype === "f16" ? 2e-3 * mag + 1e-3 : 2e-6 * mag + 1e-6;
                  if (!(err <= tol)) assert.fail(`M=${M} [${m}, ${n}] got ${got[m * N + n]} want ${acc} (err ${err}, tol ${tol})`);
                }
              b.dispose(x);
              b.dispose(y);
            }
            b.dispose(bt);
            disposeQuantized(b, q);
          });
        }
      }
    });
  }

  t.describe("webgpu quantizedEmbedding", () => {
    for (const [bits, mode] of [[8, "symmetric"], [4, "affine"]] as const) {
      t.it(`q${bits} ${mode}: gathers dequantized rows (f32 exact to fl32(q·s + b))`, async () => {
        const b = probe;
        const rq = randomQuantized(40, 96, bits, mode, 64);
        const q = await uploadQuantized(b, rq.h, "f32");
        const ids = Int32Array.from([3, 0, 39, 3, 17]);
        const it = await b.fromHost({ dtype: "i32", shape: [5, 1], data: ids });
        const e = quantizedEmbedding(b, q, it);
        assert.deepEqual(e.shape, [5, 1, 96]);
        const got = toF32(await b.read(e));
        for (let r = 0; r < 5; r++)
          for (let d = 0; d < 96; d++) assert.equal(got[r * 96 + d], Math.fround(rq.w[ids[r]! * 96 + d]!), `row ${r} col ${d}`);
        disposeQuantized(b, q);
      });
    }
  });

  t.it("declines group sizes that are not a multiple of 4", async () => {
    const rq = randomQuantized(8, 64, 8, "symmetric", 6);
    assert.equal(await probe.fromHostQuantized(rq.h, "f32"), null);
    const q = await uploadQuantized(probe, rq.h, "f32");
    assert.equal(q.native, false, "falls back to the default composition");
    disposeQuantized(probe, q);
  });

  t.after(() => {
    for (const b of created) b.destroy();
  });
}
