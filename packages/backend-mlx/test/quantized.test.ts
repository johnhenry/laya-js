import * as nodeTest from "node:test";
// @ts-ignore -- bun types are not installed
const bunTest: unknown = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const { describe, it } = (bunTest ?? nodeTest) as Pick<typeof nodeTest, "describe" | "it">;
import assert from "node:assert/strict";
import {
  QUANTIZED_OPS,
  disposeQuantized,
  packQuantized,
  quantizedEmbedding,
  quantizedLinear,
  toF32,
  uploadQuantized,
  type HostQuantized,
  type QuantBits,
  type QuantMode,
} from "@johnhenry/tensor-backend";
import { withoutOptionalOps } from "@johnhenry/tensor-backend/conformance";
import { createMlxBackend, type MlxBackend } from "../src/index.ts";
import { skipReason } from "./env.ts";

/** Deterministic pseudo-random quantized matrix with f16 scales (and biases). */
function randomQuantized(N: number, K: number, bits: QuantBits, mode: QuantMode, g: number, seed = 1): { h: HostQuantized; q: Int32Array } {
  let s = seed >>> 0;
  const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
  const G = Math.ceil(K / g);
  const q = new Int32Array(N * K);
  const lo = mode === "symmetric" ? -(1 << (bits - 1)) : 0, hi = mode === "symmetric" ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;
  for (let i = 0; i < q.length; i++) q[i] = lo + Math.floor(rnd() * (hi - lo + 1));
  const scales = Float16Array.from({ length: N * G }, () => (0.2 + rnd()) * 1e-3 * (bits === 4 ? 16 : 1));
  const biases = mode === "affine" ? Float16Array.from({ length: N * G }, () => (rnd() - 0.5) * 0.02) : null;
  return {
    q,
    h: {
      shape: [N, K], bits, groupSize: g, mode, data: packQuantized(q, bits),
      scales: { dtype: "f16", shape: [N, G], data: scales },
      biases: biases ? { dtype: "f16", shape: [N, G], data: biases } : null,
    },
  };
}

/** fl32(q · scale + bias): what a single-rounding f32 dequantization gives. */
function dequantRef({ h, q }: { h: HostQuantized; q: Int32Array }): Float32Array {
  const [N, K] = h.shape, G = Math.ceil(K / h.groupSize);
  const s = toF32(h.scales), b = h.biases ? toF32(h.biases) : null;
  const out = new Float32Array(N * K);
  for (let r = 0; r < N; r++)
    for (let j = 0; j < K; j++) {
      const gi = r * G + Math.floor(j / h.groupSize);
      out[r * K + j] = q[r * K + j]! * s[gi]! + (b ? b[gi]! : 0);
    }
  return out;
}

if (skipReason) {
  describe("backend-mlx quantized", () => it.skip(`skipped: ${skipReason}`, () => {}));
} else {
  describe("backend-mlx quantized", () => {
    let b: MlxBackend;
    const get = () => (b ??= createMlxBackend());

    it("keeps MLX-supported configurations native and declines the rest", async () => {
      const be = get();
      for (const [g, K, native] of [[64, 128, true], [32, 64, true], [128, 256, true], [16, 64, false], [64, 96, false]] as const) {
        const { h } = randomQuantized(8, K, 4, "affine", g);
        const q = await be.fromHostQuantized(h, "f16");
        assert.equal(q !== null, native, `g${g} K${K}`);
        if (q) {
          assert.deepEqual(q.w.shape, [8, K / 8], "u32 words, 8 nibbles each");
          disposeQuantized(be, q);
        }
        const u = await uploadQuantized(be, h, "f16");
        assert.equal(u.native, native);
        disposeQuantized(be, u);
      }
    });

    for (const [bits, mode] of [[8, "symmetric"], [4, "affine"], [8, "affine"], [4, "symmetric"]] as const) {
      it(`q${bits} ${mode}: the repacked weights dequantize bit-for-bit to fl32(q·scale + bias)`, async () => {
        const be = get();
        const rq = randomQuantized(96, 256, bits, mode, 64, bits * 7 + mode.length);
        const q = await uploadQuantized(be, rq.h, "f32");
        assert.equal(q.native, true);
        const ids = await be.fromHost({ dtype: "i32", shape: [96], data: Int32Array.from({ length: 96 }, (_, i) => i) });
        const got = toF32(await be.read(quantizedEmbedding(be, q, ids)));
        const want = dequantRef(rq);
        let diff = 0;
        for (let i = 0; i < want.length; i++) if (got[i] !== want[i]) diff++;
        assert.equal(diff, 0, `${diff} of ${want.length} values differ`);
        // and the composed path (values as floats, dequantized by elementwise ops) agrees bit-for-bit too
        const composed = withoutOptionalOps(be, QUANTIZED_OPS);
        const qc = await uploadQuantized(composed, rq.h, "f32");
        assert.equal(qc.native, false);
        const gotC = toF32(await be.read(quantizedEmbedding(composed, qc, ids)));
        assert.deepEqual(gotC, want);
        disposeQuantized(be, q);
        disposeQuantized(be, qc);
      });

      it(`q${bits} ${mode}: quantizedLinear (quantized_matmul) matches the dequantized f32 GEMM`, async () => {
        const be = get();
        const rq = randomQuantized(160, 512, bits, mode, 64, bits * 13 + mode.length);
        const M = 9, K = 512, N = 160;
        const xv = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 0.37) * 0.5);
        const x = await be.fromHost({ dtype: "f32", shape: [M, K], data: xv });
        const q = await uploadQuantized(be, rq.h, "f32");
        const y = toF32(await be.read(quantizedLinear(be, x, q)));
        const w = dequantRef(rq);
        // Error relative to Σ|x·w| (the dot product's natural error scale): MLX
        // computes scale·Σx·q + bias·Σx per group, and symmetric weights run as
        // affine with bias = −2^(bits−1)·scale, which cancels (≈1e-4 of |y| at worst).
        let worst = 0;
        for (let m = 0; m < M; m++)
          for (let n = 0; n < N; n++) {
            let acc = 0, mag = 0;
            for (let k = 0; k < K; k++) (acc += xv[m * K + k]! * w[n * K + k]!), (mag += Math.abs(xv[m * K + k]! * w[n * K + k]!));
            worst = Math.max(worst, Math.abs(acc - y[m * N + n]!) / mag);
          }
        assert.ok(worst < 2e-6, `max error / Σ|x·w| = ${worst}`);
        // f16 activations: the result keeps x's dtype
        const x16 = be.cast(x, "f16");
        const q16 = await uploadQuantized(be, rq.h, "f16");
        const y16 = quantizedLinear(be, x16, q16);
        assert.equal(y16.dtype, "f16");
        const y32x16 = quantizedLinear(be, x16, q);
        assert.equal(y32x16.dtype, "f16");
        disposeQuantized(be, q);
        disposeQuantized(be, q16);
      });
    }

    it("works under compile (quantized weights captured as constants)", async () => {
      const be = get();
      const rq = randomQuantized(64, 128, 4, "affine", 64, 5);
      const q = await uploadQuantized(be, rq.h, "f16");
      const x = be.cast(await be.fromHost({ dtype: "f32", shape: [3, 128], data: Float32Array.from({ length: 384 }, (_, i) => (i % 7) / 7) }), "f16");
      const f = be.compile((t) => quantizedLinear(be, t, q));
      const a = toF32(await be.read(f(x)));
      const e = toF32(await be.read(quantizedLinear(be, x, q)));
      assert.deepEqual(a, e);
      disposeQuantized(be, q);
    });
  });
}
