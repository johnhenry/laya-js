/**
 * Paths the small conformance fixtures don't reach: large/aligned GEMM
 * (direct + tiled kernels), head-dim-64 flash attention with sliding masks,
 * long rows, 2-D launch grids, sort fallback, views/offsets, bf16/f16 storage.
 * Oracles are straightforward f64 JS reference implementations.
 */
import assert from "node:assert/strict";
import { toF32 } from "@johnhenry/tensor-backend";
import { createWebGpuBackend, isWebGpuAvailable, requestAdapter, type GemmConfig, type WebGpuBackend, type WebGpuTensor } from "../src/index.ts";
import { sdpaBytes, sdpaConfig, sdpaFastBytes } from "../src/kernels.ts";
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
      const X = await up(bk, [M, K], x), W = await up(bk, [N, K], w), Bi = await up(bk, [N], bias);
      close(await rd(bk, bk.linear(X, W, Bi)), want, 1e-4, 1e-4, `linear f32 ${M}x${N}x${K}`);
      const y16 = bk.linear(bk.cast(X, "f16"), bk.cast(W, "f16"), bk.cast(Bi, "f16"));
      assert.equal(y16.dtype, "f16");
      close(await rd(bk, y16), want, 2e-2, 2e-2, `linear f16 ${M}x${N}x${K}`);
    }
  });

  t.it("linear: subgroup-matrix kernels (when available) vs direct, partial tiles, K % 8 != 0, wide loads, split-K", async () => {
    const bk = await get();
    const saved = bk.gemmConfig;
    try {
      for (const [M, N, K] of [[93, 200, 256], [300, 130, 268], [65, 64, 64]] as const) {
        const x = rnd((M + 1) * K), w = rnd(N * K, 0.05), bias = rnd(N);
        const want = refLinear(x.subarray(K), w, bias, M, N, K);
        const X = bk.slice(await up(bk, [M + 1, K], x), [1, 0], [M + 1, K]); // offset view
        const W = await up(bk, [N, K], w), Bi = await up(bk, [N], bias);
        const cfgs: [string, GemmConfig][] = [
          ["default", saved],
          ["direct", { ...saved, sg: null, skinny: [] }],
          ["sg 32x64", { ...saved, skinny: [], sg: [{ minM: 0, BM: 32, BN: 64, BK: 8, WM: 2, WN: 2 }] }],
          ["sg 96x64x16", { ...saved, skinny: [], sg: [{ minM: 0, BM: 96, BN: 64, BK: 16, WM: 2, WN: 2 }] }],
          ["sg 64x64 pad 0", { ...saved, skinny: [], sg: [{ minM: 0, BM: 64, BN: 64, BK: 8, WM: 1, WN: 2, pad: 0 }] }],
          ["sg 32x64 narrow, double-buffered, block epilogue", { ...saved, skinny: [], sg: [{ minM: 0, BM: 32, BN: 64, BK: 8, WM: 1, WN: 2, wide: false, db: true, epi: "block" }] }],
          ["sg 32x64 split-K 3", { ...saved, skinny: [], sg: [{ minM: 0, BM: 32, BN: 64, BK: 8, WM: 1, WN: 2, splitK: [{ S: 3 }] }] }],
          ["sg 64x64x16 split-K 2", { ...saved, skinny: [], sg: [{ minM: 0, BM: 64, BN: 64, BK: 16, WM: 2, WN: 2, splitK: [{ S: 2 }] }] }],
        ];
        for (const [name, cfg] of cfgs) {
          bk.gemmConfig = cfg;
          close(await rd(bk, bk.linear(X, W, Bi)), want, 1e-4, 1e-4, `linear ${name} f32 ${M}x${N}x${K}`);
          close(await rd(bk, bk.linear(bk.cast(X, "f16"), bk.cast(W, "f16"), bk.cast(Bi, "f16"))), want, 2e-2, 2e-2, `linear ${name} f16 ${M}x${N}x${K}`);
        }
      }
    } finally {
      bk.gemmConfig = saved;
    }
  });

  t.it("buffer reuse: a host upload into a buffer freed while pending work still reads it; repeated chains hit the bind-group cache", async () => {
    const bk = await get();
    const [M, N, K] = [70, 96, 64];
    const x1 = rnd(M * K), x2 = rnd(M * K), w = rnd(N * K, 0.1);
    const W = await up(bk, [N, K], w);
    const X1 = await up(bk, [M, K], x1);
    const Y1 = bk.linear(X1, W); // enqueued, not submitted
    bk.dispose(X1); // back to the pool while Y1's dispatch is pending
    const X2 = await up(bk, [M, K], x2); // reuses X1's buffer: must not clobber Y1's input
    const Y2 = bk.linear(X2, W);
    close(await rd(bk, Y1), refLinear(x1, w, null, M, N, K), 1e-4, 1e-4, "Y1 after reuse");
    close(await rd(bk, Y2), refLinear(x2, w, null, M, N, K), 1e-4, 1e-4, "Y2");
    const chain = () => bk.scope(() => bk.gelu(bk.add(bk.linear(X2, W), bk.linear(X2, W))));
    for (let i = 0; i < 3; i++) bk.dispose(chain());
    const before = bk.rt.stats.bindGroups;
    for (let i = 0; i < 5; i++) bk.dispose(chain());
    await bk.sync();
    assert.equal(bk.rt.stats.bindGroups, before, "steady-state chain creates no bind groups");
  });

  t.it("tuneGemm records a measured choice per shape and uses it", async () => {
    const bk = await get();
    const shapes = [{ M: 70, N: 96, K: 64 }, { M: 20, N: 64, K: 32 }];
    const picks = await bk.tuneGemm(shapes, { dtype: "f32", rounds: 1 });
    for (const [key, v] of Object.entries(picks)) {
      assert.equal(bk.gemmTuning.get(key), v);
      assert.ok(v === "skinny" || v === "direct" || typeof v === "number", key);
    }
    const x = rnd(70 * 64), w = rnd(96 * 64, 0.1);
    close(await rd(bk, bk.linear(await up(bk, [70, 64], x), await up(bk, [96, 64], w))), refLinear(x, w, null, 70, 96, 64), 1e-4, 1e-4, "tuned linear");
    bk.gemmTuning.clear();
  });

  t.it("strided copies: collapsed 5-D transpose (vector and scalar inner loops), concat", async () => {
    const bk = await get();
    for (const hd of [8, 6]) {
      const [B, L, nh] = [2, 5, 3];
      const x = rnd(B * L * 3 * nh * hd);
      const X = await up(bk, [B, L, 3, nh, hd], x);
      const perm = [2, 0, 3, 1, 4];
      const shape = perm.map((p) => [B, L, 3, nh, hd][p]!);
      const want = new Float32Array(x.length);
      let o = 0;
      for (let a = 0; a < 3; a++) for (let b0 = 0; b0 < B; b0++) for (let h = 0; h < nh; h++) for (let l = 0; l < L; l++) for (let d = 0; d < hd; d++)
        want[o++] = x[(((b0 * L + l) * 3 + a) * nh + h) * hd + d]!;
      const y = bk.transpose(X, perm);
      assert.deepEqual([...y.shape], shape);
      close(await rd(bk, y), want, 0, 0, `transpose f32 hd=${hd}`);
      close(await rd(bk, bk.transpose(bk.cast(X, "f16"), perm)), want, 2e-3, 2e-3, `transpose f16 hd=${hd}`);
    }
    const a = rnd(6), c = rnd(9);
    close(await rd(bk, bk.concat([await up(bk, [2, 3], a), await up(bk, [3, 3], c)], 0)), Float32Array.from([...a, ...c]), 0, 0, "concat axis 0");
  });

  t.it("linear on a sliced (offset) view", async () => {
    const bk = await get();
    const M = 8, N = 64, K = 64;
    const x = rnd(2 * M * K), w = rnd(N * K, 0.1);
    const X = await up(bk, [2, M, K], x);
    const second = bk.slice(X, [1, 0, 0], [2, M, K]); // free view with offset M*K
    const got = await rd(bk, bk.linear(second, await up(bk, [N, K], w)));
    close(got, refLinear(x.subarray(M * K), w, null, M, N, K), 1e-4, 1e-4, "offset view");
  });

  t.it("batched matmul with broadcasting (tiled kernel, vec and scalar B loads)", async () => {
    const bk = await get();
    for (const [M, K, N] of [[70, 64, 96], [33, 17, 29]] as const) {
      const a = rnd(3 * M * K), bb = rnd(K * N);
      const got = await rd(bk, bk.matmul(await up(bk, [3, M, K], a), await up(bk, [1, K, N], bb)));
      const want = new Float32Array(3 * M * N);
      const bt = new Float32Array(N * K);
      for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) bt[n * K + k] = bb[k * N + n]!;
      for (let i = 0; i < 3; i++) want.set(refLinear(a.subarray(i * M * K, (i + 1) * M * K), bt, null, M, N, K), i * M * N);
      close(got, want, 1e-4, 1e-4, `matmul ${M}x${K}x${N}`);
      // Batch-broadcast on the left operand ([1,M,K] @ [3,K,N]).
      const b3 = rnd(3 * K * N);
      const got2 = await rd(bk, bk.matmul(await up(bk, [1, M, K], a.subarray(0, M * K)), await up(bk, [3, K, N], b3)));
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
    const M = await bk.fromHost({ dtype: "bool", shape: [B, 1, L, L], data: mask });
    const Q = await up(bk, shape, q), K = await up(bk, shape, k), V = await up(bk, shape, v);
    close(await rd(bk, bk.sdpa(Q, K, V, M, 0.125)), want, 1e-4, 1e-4, "sdpa f32");
    const y16 = bk.sdpa(bk.cast(Q, "f16"), bk.cast(K, "f16"), bk.cast(V, "f16"), M, 0.125);
    close(await rd(bk, y16), want, 2e-2, 2e-2, "sdpa f16");
  });

  t.it("layerNorm / softmax on long rows", async () => {
    const bk = await get();
    const R = 3, D = 5000;
    const x = rnd(R * D, 4);
    const X = await up(bk, [R, D], x);
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
      const got = await rd(bk, bk.sort(await up(bk, [2, n], x), -1));
      const want = new Float32Array(2 * n);
      for (let r = 0; r < 2; r++) want.set(Float32Array.from(x.subarray(r * n, (r + 1) * n)).sort(), r * n);
      close(got, want, 0, 0, `sort n=${n}`);
    }
    const x = rnd(5 * 3);
    const got = await rd(bk, bk.sort(await up(bk, [5, 3], x), 0));
    for (let c = 0; c < 3; c++) {
      const col = Float32Array.from([0, 1, 2, 3, 4].map((r) => x[r * 3 + c]!)).sort();
      close([0, 1, 2, 3, 4].map((r) => got[r * 3 + c]!), col, 0, 0, "sort axis 0");
    }
  });

  t.it("2-D launch grid (> 65535 workgroups) for elementwise ops", async () => {
    const bk = await get();
    const n = 65536 * 256 + 1000;
    const x = rnd(n);
    const got = await rd(bk, bk.add(await up(bk, [n], x), await up(bk, [1], new Float32Array([1]))));
    for (const i of [0, 1, 65535 * 256, n - 1]) assert.ok(Math.abs(got[i]! - (x[i]! + 1)) < 1e-6, `at ${i}`);
  });

  t.it("gelu matches exact erf GELU to ~1e-6 in f32", async () => {
    const bk = await get();
    const x = Float32Array.from({ length: 16001 }, (_, i) => -8 + i * 0.001);
    const got = await rd(bk, bk.gelu(await up(bk, [x.length], x)));
    let worst = 0;
    for (let i = 0; i < x.length; i++) worst = Math.max(worst, Math.abs(got[i]! - 0.5 * x[i]! * (1 + erfRef(x[i]! / Math.SQRT2))));
    assert.ok(worst < 2e-6, `max abs err ${worst}`);
  });

  t.it("erf (canonical math-plus algorithm) is within 2.5e-7 of the f64 erf over [-8, 8]", async () => {
    const bk = await get();
    const x = Float32Array.from({ length: 16001 }, (_, i) => -8 + i * 0.001);
    const got = await rd(bk, bk.erf(await up(bk, [x.length], x)));
    let worst = 0, at = 0;
    for (let i = 0; i < x.length; i++) {
      const d = Math.abs(got[i]! - erfRef(x[i]!));
      if (d > worst) { worst = d; at = x[i]!; }
    }
    assert.ok(worst < 2.5e-7, `max abs err ${worst} at ${at}`);
  });

  t.it("dtype storage: bf16 round-trip, f16 odd-offset read, i32/bool casts, reshape is a view", async () => {
    const bk = await get();
    const bits = Uint16Array.from([0x3f80, 0xc000, 0x3e20, 0x7f80]);
    const t = await bk.fromHost({ dtype: "bf16", shape: [4], data: bits });
    assert.deepEqual([...((await bk.read(t)).data as Uint16Array)], [...bits]);
    const h = bk.cast(await up(bk, [5], Float32Array.from([1, 2, 3, 4, 5])), "f16");
    const odd = bk.slice(h, [1], [4]);
    close(await rd(bk, odd), [2, 3, 4], 0, 0, "f16 odd offset");
    const i = await bk.fromHost({ dtype: "i32", shape: [3], data: Int32Array.from([-2, 0, 7]) });
    assert.deepEqual([...(await bk.read(bk.cast(i, "bool"))).data], [1, 0, 1]);
    assert.deepEqual([...(await bk.read(bk.add(i, i))).data], [-4, 0, 14]);
    const before = bk.rt.stats.dispatches;
    const r = bk.reshape(await up(bk, [2, 3], rnd(6)), [3, -1]);
    assert.deepEqual(r.shape, [3, 2]);
    assert.equal(bk.rt.stats.dispatches, before);
  });

  t.it("fromHost uploads TypedArray views with a non-zero byteOffset (Bun/Dawn regression)", async () => {
    const bk = await get();
    const backing = Float32Array.from([9, 9, 1, 2, 3, 4, 9]);
    const view = new Float32Array(backing.buffer, 8, 4);
    const x = await bk.fromHost({ dtype: "f32", shape: [4], data: view });
    close(await rd(bk, x), [1, 2, 3, 4], 0, 0, "offset view");
    const h = new Float16Array([7, 7, 7, 0.5, 1.5, 2.5]).subarray(3);
    const y = await bk.fromHost({ dtype: "f16", shape: [3], data: h });
    close(await rd(bk, bk.cast(y, "f32")), [0.5, 1.5, 2.5], 0, 0, "f16 offset view");
  });
  t.it("sdpa respects maxComputeWorkgroupStorageSize: on a default-limit (16 KiB) device, fast falls back and generic tiles shrink; results match", async () => {
    // Pure: the configs fit the limit.
    for (const D of [32, 48, 64, 96, 128, 256]) {
      const c = sdpaConfig(D, 16384);
      assert.ok(sdpaBytes(D, c.BQ, c.BKV) <= 16384, `D=${D} generic fits 16 KiB`);
    }
    assert.ok(sdpaFastBytes(64, true) > 16384 && sdpaFastBytes(32, true) <= 16384);
    assert.deepEqual(sdpaConfig(64), { BQ: 32, BKV: 32, WG: 128 }, "unlimited: the tuned tiles");
    // Real: a device requested with the spec-default limits.
    const adapter = (await requestAdapter())!;
    const device = await adapter.requestDevice();
    assert.equal(device.limits.maxComputeWorkgroupStorageSize, 16384);
    const bk = await createWebGpuBackend({ device, adapter });
    try {
      const B = 1, H = 2, L = 40;
      for (const D of [32, 48, 64, 128, 256]) {
        const q = rnd(B * H * L * D), k = rnd(B * H * L * D), v = rnd(B * H * L * D);
        const mask = new Uint8Array(B * L * L).map((_, i) => ((i % L) <= Math.floor(i / L) ? 1 : 0)); // causal
        const scale = 1 / Math.sqrt(D);
        const shape = [B, H, L, D];
        const M = await bk.fromHost({ dtype: "bool", shape: [B, 1, L, L], data: mask });
        const Q = await up(bk, shape, q), K = await up(bk, shape, k), V = await up(bk, shape, v);
        device.pushErrorScope("validation");
        const plain = await rd(bk, bk.sdpa(Q, K, V, null, scale));
        const masked = await rd(bk, bk.sdpa(Q, K, V, M, scale));
        const err = await device.popErrorScope();
        assert.equal(err, null, `D=${D}: ${err?.message}`);
        close(plain, refSdpa(q, k, v, null, B, H, L, D, scale), 1e-4, 1e-4, `sdpa D=${D} 16 KiB`);
        close(masked, refSdpa(q, k, v, mask, B, H, L, D, scale), 1e-4, 1e-4, `sdpa D=${D} 16 KiB masked`);
      }
    } finally {
      bk.destroy();
      device.destroy();
    }
  });

  t.it("elementwise: custom n-ary kernel over x0..xN with broadcasting, offsets, i32 inputs, helpers and outDtype", async () => {
    const bk = await get();
    const a = rnd(3 * 4), b4 = rnd(4), c = rnd(3), e = rnd(12);
    const A = await up(bk, [3, 4], a);
    const Bv = await up(bk, [4], b4); // broadcast along rows
    const C = await up(bk, [3, 1], c); // broadcast along columns
    const E = bk.reshape(bk.slice(await up(bk, [2, 3, 4], Float32Array.from([...rnd(12), ...e])), [1, 0, 0], [2, 3, 4]), [3, 4]); // view at offset 12
    const I = await bk.fromHost({ dtype: "i32", shape: [3, 4], data: Int32Array.from({ length: 12 }, (_, i) => i - 5) });
    const helpers = "fn sq(x: f32) -> f32 { return x * x; }";
    const before = bk.rt.stats.dispatches;
    const y = bk.elementwise("sq(x0) + x1 * x2 - x3 + x4", [A, Bv, C, E, I], { helpers });
    assert.equal(bk.rt.stats.dispatches - before, 1, "one dispatch");
    assert.deepEqual(y.shape, [3, 4]);
    const want = Float32Array.from({ length: 12 }, (_, i) => a[i]! ** 2 + b4[i % 4]! * c[Math.floor(i / 4)]! - e[i]! + (i - 5));
    close(await rd(bk, y), want, 1e-5, 1e-5, "elementwise");
    const pos = bk.elementwise("select(0.0, 1.0, x0 > 0.0)", [A], { outDtype: "bool" }); // bool out: nonzero = true
    assert.equal(pos.dtype, "bool");
    assert.deepEqual([...(await bk.read(pos)).data], [...a].map((v) => (v > 0 ? 1 : 0)));
    const pipelines = bk.rt.stats.pipelines;
    bk.elementwise("sq(x0) + x1 * x2 - x3 + x4", [A, Bv, C, E, I], { helpers });
    assert.equal(bk.rt.stats.pipelines, pipelines, "the same expression reuses its pipeline");
    assert.throws(() => bk.elementwise("x0", []), /at least one input/);
  });

  t.it("empty + rt.kernel/rt.dispatch run a custom kernel into a scope-tracked output; wrapBuffer views a caller-owned buffer and never pools it", async () => {
    const bk = await get();
    const x = await up(bk, [5], Float32Array.from([1, 2, 3, 4, 5]));
    let tmp: WebGpuTensor | undefined;
    const out = bk.scope(() => {
      tmp = bk.empty([5], "f32");
      const o = bk.empty([5], "f32");
      const k = bk.rt.kernel(() => ({
        key: "test:triple",
        bindings: [{ name: "inp", elem: "f32", access: "read" }, { name: "outp", elem: "f32", access: "read_write" }],
        params: [["n", "u32"]],
        body: "@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < P.n) { outp[g.x] = 3.0 * inp[g.x]; } }",
        f16: false,
      }), "test:triple");
      bk.rt.dispatch(k, [x.storage.buffer, o.storage.buffer], { n: 5 }, [1]);
      return o;
    });
    assert.equal(tmp!.disposed, true, "an unreturned empty() tensor is freed by the scope");
    close(await rd(bk, out), [3, 6, 9, 12, 15], 0, 0, "custom kernel");

    const buf = bk.device.createBuffer({ size: 32, usage: 0x80 | 0x04 | 0x08 /* STORAGE | COPY_SRC | COPY_DST */ });
    bk.device.queue.writeBuffer(buf, 0, Float32Array.from([0, 0, 1, 2, 3, 4, 0, 0]));
    const w = bk.wrapBuffer(buf, [4], "f32", 2);
    close(await rd(bk, bk.scale(w, 2)), [2, 4, 6, 8], 0, 0, "wrapped view");
    const pooled = bk.rt.stats.pooledBytes;
    bk.dispose(w);
    assert.equal(bk.rt.stats.pooledBytes, pooled, "a wrapped buffer is not returned to the pool");
    const again = bk.wrapBuffer(buf, [4], "f32", 2); // still alive: not destroyed by dispose
    close(await rd(bk, again), [1, 2, 3, 4], 0, 0, "buffer survives dispose");
    assert.throws(() => bk.wrapBuffer(buf, [8], "f32", 2), /needs 40 bytes/);
    buf.destroy();
  });

  t.it("createWebGpuBackend({ device, adapter }) detects subgroup matrices like a backend that requested its own device; sleepThresholdMs is applied", async () => {
    const own = await get();
    const adapter = (await requestAdapter(undefined, true))!;
    const device = await adapter.requestDevice({ requiredFeatures: [...adapter.features].filter((f) => f === "chromium-experimental-subgroup-matrix" || f === "shader-f16") as GPUFeatureName[] });
    const withAdapter = await createWebGpuBackend({ device, adapter, sleepThresholdMs: 15 });
    assert.equal(withAdapter.hasSubgroupMatrix, own.hasSubgroupMatrix);
    assert.equal(withAdapter.rt.sleepThresholdMs, 15);
    assert.equal(own.rt.sleepThresholdMs, 3, "default");
    withAdapter.destroy();
    device.destroy();
  });
});
