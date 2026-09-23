/**
 * Paths the small conformance fixtures don't reach: large/aligned GEMM
 * (direct + tiled kernels), head-dim-64 flash attention with sliding masks,
 * long rows, 2-D launch grids, sort fallback, views/offsets, bf16/f16 storage.
 * Oracles are straightforward f64 JS reference implementations.
 */
import assert from "node:assert/strict";
import { toF32 } from "@johnhenry/tensor-backend";
import { createWebGpuBackend, isWebGpuAvailable, type WebGpuBackend, type WebGpuTensor } from "../src/index.ts";
import { harness, isBun } from "./harness.ts";

// @ts-ignore -- bun types are not installed
const t = harness(isBun ? await import("bun:test") : null);

const available = await isWebGpuAvailable();
let seed = 1234;
const rnd = (n: number, s = 1) =>
  Float32Array.from({ length: n }, () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return ((seed / 2 ** 32) * 2 - 1) * s;
  });

function close(got: ArrayLike<number>, want: ArrayLike<number>, atol: number, rtol: number, label: string) {
  assert.equal(got.length, want.length, `${label}: length`);
  for (let i = 0; i < got.length; i++) {
    const e = want[i]!, v = got[i]!;
    if (!(Math.abs(v - e) <= atol + rtol * Math.abs(e))) assert.fail(`${label}: [${i}] got ${v} want ${e}`);
  }
}

function refLinear(x: Float32Array, w: Float32Array, bias: Float32Array | null, M: number, N: number, K: number) {
  const out = new Float32Array(M * N);
  for (let m = 0; m < M; m++)
    for (let n = 0; n < N; n++) {
      let s = bias ? bias[n]! : 0;
      for (let k = 0; k < K; k++) s += x[m * K + k]! * w[n * K + k]!;
      out[m * N + n] = s;
    }
  return out;
}

function refSdpa(q: Float32Array, k: Float32Array, v: Float32Array, mask: Uint8Array | null, B: number, H: number, L: number, D: number, scale: number) {
  const out = new Float32Array(B * H * L * D);
  for (let b = 0; b < B; b++)
    for (let h = 0; h < H; h++) {
      const base = (b * H + h) * L * D;
      for (let i = 0; i < L; i++) {
        const s = new Float64Array(L);
        let mx = -Infinity;
        for (let j = 0; j < L; j++) {
          let acc = 0;
          for (let d = 0; d < D; d++) acc += q[base + i * D + d]! * k[base + j * D + d]!;
          s[j] = mask && !mask[(b * L + i) * L + j] ? -Infinity : acc * scale;
          mx = Math.max(mx, s[j]!);
        }
        let z = 0;
        for (let j = 0; j < L; j++) z += s[j] = Math.exp(s[j]! - mx);
        for (let d = 0; d < D; d++) {
          let acc = 0;
          for (let j = 0; j < L; j++) acc += s[j]! * v[base + j * D + d]!;
          out[base + i * D + d] = acc / z;
        }
      }
    }
  return out;
}

// High-precision erf for the GELU accuracy check.
function erfRef(x: number): number {
  const ax = Math.abs(x);
  let r: number;
  if (ax < 3) {
    let t = ax, s = ax;
    for (let n = 1; n < 200; n++) {
      t *= (-ax * ax) / n;
      const term = t / (2 * n + 1);
      s += term;
      if (Math.abs(term) < 1e-18) break;
    }
    r = (2 / Math.sqrt(Math.PI)) * s;
  } else {
    let f = 0;
    for (let k = 60; k >= 1; k--) f = k / 2 / (ax + f);
    r = 1 - Math.exp(-ax * ax) / Math.sqrt(Math.PI) / (ax + f);
  }
  return x < 0 ? -r : r;
}

if (!available) t.skip("webgpu kernels (large / edge paths)", "no WebGPU adapter");
else t.describe("webgpu kernels (large / edge paths)", () => {
  let b: WebGpuBackend;
  const get = async () => (b ??= await createWebGpuBackend());
  const up = (bk: WebGpuBackend, shape: number[], data: Float32Array) => bk.fromHost({ dtype: "f32", shape, data });
  const rd = async (bk: WebGpuBackend, t: WebGpuTensor) => toF32(await bk.read(t));
  t.after(() => b?.destroy());

  t.it("linear: direct, tiled and skinny kernels, bias, f16", async () => {
    const bk = await get();
    // direct (M>64), tiled (K % 4 != 0), skinny buckets (M ≤ 40, M ≤ 64)
    for (const [M, N, K] of [[130, 200, 256], [37, 45, 70], [1, 3072, 1024], [33, 100, 256], [50, 96, 128]] as const) {
      const x = rnd(M * K), w = rnd(N * K, 0.05), bias = rnd(N);
      const want = refLinear(x, w, bias, M, N, K);
      const X = up(bk, [M, K], x), W = up(bk, [N, K], w), Bi = up(bk, [N], bias);
      close(await rd(bk, bk.linear(X, W, Bi)), want, 1e-4, 1e-4, `linear f32 ${M}x${N}x${K}`);
      const y16 = bk.linear(bk.cast(X, "f16"), bk.cast(W, "f16"), bk.cast(Bi, "f16"));
      assert.equal(y16.dtype, "f16");
      close(await rd(bk, y16), want, 2e-2, 2e-2, `linear f16 ${M}x${N}x${K}`);
    }
  });

  t.it("linear on a sliced (offset) view", async () => {
    const bk = await get();
    const M = 8, N = 64, K = 64;
    const x = rnd(2 * M * K), w = rnd(N * K, 0.1);
    const X = up(bk, [2, M, K], x);
    const second = bk.slice(X, [1, 0, 0], [2, M, K]); // free view with offset M*K
    const got = await rd(bk, bk.linear(second, up(bk, [N, K], w)));
    close(got, refLinear(x.subarray(M * K), w, null, M, N, K), 1e-4, 1e-4, "offset view");
  });

  t.it("batched matmul with broadcasting (tiled kernel, vec and scalar B loads)", async () => {
    const bk = await get();
    for (const [M, K, N] of [[70, 64, 96], [33, 17, 29]] as const) {
      const a = rnd(3 * M * K), bb = rnd(K * N);
      const got = await rd(bk, bk.matmul(up(bk, [3, M, K], a), up(bk, [1, K, N], bb)));
      const want = new Float32Array(3 * M * N);
      const bt = new Float32Array(N * K);
      for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) bt[n * K + k] = bb[k * N + n]!;
      for (let i = 0; i < 3; i++) want.set(refLinear(a.subarray(i * M * K, (i + 1) * M * K), bt, null, M, N, K), i * M * N);
      close(got, want, 1e-4, 1e-4, `matmul ${M}x${K}x${N}`);
      // Batch-broadcast on the left operand ([1,M,K] @ [3,K,N]).
      const b3 = rnd(3 * K * N);
      const got2 = await rd(bk, bk.matmul(up(bk, [1, M, K], a.subarray(0, M * K)), up(bk, [3, K, N], b3)));
      for (let i = 0; i < 3; i++) {
        const bti = new Float32Array(N * K);
        for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) bti[n * K + k] = b3[i * K * N + k * N + n]!;
        close(got2.subarray(i * M * N, (i + 1) * M * N), refLinear(a.subarray(0, M * K), bti, null, M, N, K), 1e-4, 1e-4, `matmul bcast ${i}`);
      }
    }
  });

  t.it("sdpa: head dim 64, L=100, sliding + padding mask, f32 and f16", async () => {
    const bk = await get();
    const B = 2, H = 2, L = 100, D = 64, W = 16;
    const q = rnd(B * H * L * D), k = rnd(B * H * L * D), v = rnd(B * H * L * D);
    const mask = new Uint8Array(B * L * L);
    for (let bb = 0; bb < B; bb++)
      for (let i = 0; i < L; i++)
        for (let j = 0; j < L; j++) mask[(bb * L + i) * L + j] = Math.abs(i - j) <= W && j < L - bb * 10 ? 1 : 0;
    const want = refSdpa(q, k, v, mask, B, H, L, D, 0.125);
    const shape = [B, H, L, D];
    const M = bk.fromHost({ dtype: "bool", shape: [B, 1, L, L], data: mask });
    const Q = up(bk, shape, q), K = up(bk, shape, k), V = up(bk, shape, v);
    close(await rd(bk, bk.sdpa(Q, K, V, M, 0.125)), want, 1e-4, 1e-4, "sdpa f32");
    const y16 = bk.sdpa(bk.cast(Q, "f16"), bk.cast(K, "f16"), bk.cast(V, "f16"), M, 0.125);
    close(await rd(bk, y16), want, 2e-2, 2e-2, "sdpa f16");
  });

  t.it("layerNorm / softmax on long rows", async () => {
    const bk = await get();
    const R = 3, D = 5000;
    const x = rnd(R * D, 4);
    const X = up(bk, [R, D], x);
    const sm = await rd(bk, bk.softmax(X, -1));
    const ln = await rd(bk, bk.layerNorm(X, null, null, 1e-5));
    for (let r = 0; r < R; r++) {
      const row = Array.from(x.subarray(r * D, (r + 1) * D));
      const mx = Math.max(...row);
      const z = row.reduce((s, v) => s + Math.exp(v - mx), 0);
      close(sm.subarray(r * D, (r + 1) * D), row.map((v) => Math.exp(v - mx) / z), 1e-7, 1e-4, "softmax");
      const mean = row.reduce((s, v) => s + v, 0) / D;
      const vr = row.reduce((s, v) => s + (v - mean) ** 2, 0) / D;
      close(ln.subarray(r * D, (r + 1) * D), row.map((v) => (v - mean) / Math.sqrt(vr + 1e-5)), 1e-4, 1e-4, "layerNorm");
    }
  });

  t.it("sort: bitonic (n=1000) and fallback (n=5000); non-last axis", async () => {
    const bk = await get();
    for (const n of [1000, 5000]) {
      const x = rnd(2 * n);
      const got = await rd(bk, bk.sort(up(bk, [2, n], x), -1));
      const want = new Float32Array(2 * n);
      for (let r = 0; r < 2; r++) want.set(Float32Array.from(x.subarray(r * n, (r + 1) * n)).sort(), r * n);
      close(got, want, 0, 0, `sort n=${n}`);
    }
    const x = rnd(5 * 3);
    const got = await rd(bk, bk.sort(up(bk, [5, 3], x), 0));
    for (let c = 0; c < 3; c++) {
      const col = Float32Array.from([0, 1, 2, 3, 4].map((r) => x[r * 3 + c]!)).sort();
      close([0, 1, 2, 3, 4].map((r) => got[r * 3 + c]!), col, 0, 0, "sort axis 0");
    }
  });

  t.it("2-D launch grid (> 65535 workgroups) for elementwise ops", async () => {
    const bk = await get();
    const n = 65536 * 256 + 1000;
    const x = rnd(n);
    const got = await rd(bk, bk.add(up(bk, [n], x), up(bk, [1], new Float32Array([1]))));
    for (const i of [0, 1, 65535 * 256, n - 1]) assert.ok(Math.abs(got[i]! - (x[i]! + 1)) < 1e-6, `at ${i}`);
  });

  t.it("gelu matches exact erf GELU to ~1e-6 in f32", async () => {
    const bk = await get();
    const x = Float32Array.from({ length: 16001 }, (_, i) => -8 + i * 0.001);
    const got = await rd(bk, bk.gelu(up(bk, [x.length], x)));
    let worst = 0;
    for (let i = 0; i < x.length; i++) worst = Math.max(worst, Math.abs(got[i]! - 0.5 * x[i]! * (1 + erfRef(x[i]! / Math.SQRT2))));
    assert.ok(worst < 2e-6, `max abs err ${worst}`);
  });

  t.it("dtype storage: bf16 round-trip, f16 odd-offset read, i32/bool casts, reshape is a view", async () => {
    const bk = await get();
    const bits = Uint16Array.from([0x3f80, 0xc000, 0x3e20, 0x7f80]);
    const t = bk.fromHost({ dtype: "bf16", shape: [4], data: bits });
    assert.deepEqual([...((await bk.read(t)).data as Uint16Array)], [...bits]);
    const h = bk.cast(up(bk, [5], Float32Array.from([1, 2, 3, 4, 5])), "f16");
    const odd = bk.slice(h, [1], [4]);
    close(await rd(bk, odd), [2, 3, 4], 0, 0, "f16 odd offset");
    const i = bk.fromHost({ dtype: "i32", shape: [3], data: Int32Array.from([-2, 0, 7]) });
    assert.deepEqual([...(await bk.read(bk.cast(i, "bool"))).data], [1, 0, 1]);
    assert.deepEqual([...(await bk.read(bk.add(i, i))).data], [-4, 0, 14]);
    const before = bk.rt.stats.dispatches;
    const r = bk.reshape(up(bk, [2, 3], rnd(6)), [3, -1]);
    assert.deepEqual(r.shape, [3, 2]);
    assert.equal(bk.rt.stats.dispatches, before);
  });
});
