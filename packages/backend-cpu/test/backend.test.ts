import * as nodeTest from "node:test";
// Bun 1.2's node:test shim only registers tests from the first file of a run,
// so use bun:test natively under Bun (same describe/it/test API).
// @ts-ignore -- bun types are not installed
const bunTest: unknown = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const { describe, it: test } = (bunTest ?? nodeTest) as Pick<typeof nodeTest, "describe" | "it">;
import assert from "node:assert/strict";
import { host, toF32 } from "@johnhenry/tensor-backend";
import { CpuTensor, createCpuBackend, erf, erfc, geluScalar } from "../src/index.ts";

describe("backend-cpu", () => {
test("supports: f32 reference only", () => {
  const b = createCpuBackend();
  assert.equal(b.supports("f32"), true);
  assert.equal(b.supports("i32"), true);
  assert.equal(b.supports("bool"), true);
  assert.equal(b.supports("f16"), false);
  assert.equal(b.supports("bf16"), false);
  assert.throws(() => b.cast(b.fromHost(host("f32", [1], [1])), "f16"));
});

test("fromHost widens f16 and bf16 to f32", async () => {
  const b = createCpuBackend();
  const f16 = b.fromHost(host("f16", [3], [1.5, -2, 0.0999755859375]));
  const bf16 = b.fromHost(host("bf16", [2], [1.5, -3]));
  assert.equal(f16.dtype, "f32");
  assert.deepEqual([...toF32(await b.read(f16))], [1.5, -2, 0.0999755859375]);
  assert.deepEqual([...toF32(await b.read(bf16))], [1.5, -3]);
});

test("nested scopes keep returned tensors (array and object) and free the rest", () => {
  const b = createCpuBackend();
  const x = b.fromHost(host("f32", [2], [1, 2]));
  let inner: CpuTensor | undefined, tmp: CpuTensor | undefined;
  const out = b.scope(() => {
    const kept = b.scope(() => {
      tmp = b.exp(x);
      inner = b.add(tmp, x);
      return { y: inner, n: 3 };
    });
    assert.equal(tmp!.disposed, true);
    assert.equal(kept.y.disposed, false);
    const z = b.mul(kept.y, x);
    return [z];
  });
  assert.equal(inner!.disposed, true, "inner result is freed by the outer scope");
  assert.equal(out[0]!.disposed, false);
  assert.equal(x.disposed, false);
  assert.throws(() => b.scope(() => { b.exp(x); throw new Error("boom"); }), /boom/);
  b.dispose(x);
  b.dispose(x);
  assert.throws(() => b.exp(x), /after dispose/);
});

test("erf / erfc / gelu match reference values", () => {
  // values from Python math.erf / math.erfc (libm)
  const cases: [number, number, number][] = [
    [0.5, 0.5204998778130465, 0.4795001221869535],
    [1, 0.8427007929497148, 0.15729920705028516],
    [2.4, 0.999311486103355, 0.0006885138966450788],
    [3, 0.9999779095030015, 2.2090496998585438e-05],
    [-4.5, -0.9999999998033839, 1.999999999803384],
    [6, 1, 2.1519736712498913e-17],
  ];
  for (const [x, e, c] of cases) {
    assert.ok(Math.abs(erf(x) - e) <= 1e-15 * Math.max(1, Math.abs(e)), `erf(${x})`);
    assert.ok(Math.abs(erfc(x) - c) <= 1e-12 * Math.abs(c), `erfc(${x}) ${erfc(x)} vs ${c}`);
  }
  assert.equal(geluScalar(0), 0);
  assert.ok(Math.abs(geluScalar(-10)) < 1e-20);
  assert.ok(Math.abs(geluScalar(1) - 0.8413447460685429) < 1e-15);
});

test("broadcast, reshape(-1), slice with negative bounds, sort", async () => {
  const b = createCpuBackend();
  const a = b.fromHost(host("f32", [2, 3], [1, 2, 3, 4, 5, 6]));
  const r = b.reshape(a, [-1, 2]);
  assert.deepEqual(r.shape, [3, 2]);
  const s = b.slice(a, [0, -2], [2, 3]);
  assert.deepEqual([...toF32(await b.read(s))], [2, 3, 5, 6]);
  const m = b.where(b.fromHost(host("bool", [1, 3], [1, 0, 1])), a, b.fromHost(host("f32", [1], [-1])));
  assert.deepEqual([...toF32(await b.read(m))], [1, -1, 3, 4, -1, 6]);
  const srt = b.sort(b.fromHost(host("f32", [2, 2], [3, 1, -1, -5])), 0);
  assert.deepEqual([...toF32(await b.read(srt))], [-1, -5, 3, 1]);
});
});
