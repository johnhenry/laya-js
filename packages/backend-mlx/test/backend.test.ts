import * as nodeTest from "node:test";
// Bun 1.2's node:test shim only registers tests from the first file of a run,
// and bun:test binds describe/it per importing file, so import it here.
// @ts-ignore -- bun types are not installed
const bunTest: unknown = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const { describe, it } = (bunTest ?? nodeTest) as Pick<typeof nodeTest, "describe" | "it">;
import assert from "node:assert/strict";
import { host, toF32 } from "@johnhenry/tensor-backend";
import { createMlxBackend, type MlxBackend, type MlxTensor } from "../src/index.ts";
import { skipReason } from "./env.ts";

if (skipReason) {
  describe("backend-mlx", () => it.skip(`skipped: ${skipReason}`, () => {}));
} else {
  describe("backend-mlx", () => {
    let b: MlxBackend;
    const get = () => (b ??= createMlxBackend());
    const f32 = (shape: number[], data: number[]) => get().fromHost(host("f32", shape, data));

    it("reports which libmlxc and ABI it loaded", () => {
      const { info } = get();
      assert.match(info.libPath, /libmlxc\.dylib$/);
      assert.ok(info.runtime === "node" || info.runtime === "bun");
      assert.equal(get().name, "mlx");
      assert.equal(get().device, "gpu");
    });

    it("supports f32/f16/bf16/i32/bool and round-trips each", async () => {
      const be = get();
      for (const d of ["f32", "f16", "bf16", "i32", "bool"] as const) {
        assert.equal(be.supports(d), true);
        const vals = d === "bool" ? [1, 0, 1] : d === "i32" ? [7, -2, 3] : [1.5, -2, 3];
        const t = await be.fromHost(host(d, [3], vals));
        assert.equal(t.dtype, d);
        const r = await be.read(t);
        assert.equal(r.dtype, d);
        assert.deepEqual([...toF32(r)], vals);
        be.dispose(t);
      }
    });

    it("computes in bf16 and f16 and keeps the dtype", async () => {
      const be = get();
      for (const d of ["bf16", "f16"] as const) {
        const x = be.cast(await f32([2, 2], [1, 2, 3, 4]), d);
        const y = be.matmul(x, x);
        assert.equal(y.dtype, d);
        assert.deepEqual([...toF32(await be.read(y))], [7, 10, 15, 22]);
        assert.equal(be.scale(x, 0.5).dtype, d, "scale keeps storage dtype");
      }
    });

    it("scalars (shape []) and strided views read back densely", async () => {
      const be = get();
      const s = await be.fromHost(host("f32", [], [4]));
      assert.deepEqual(s.shape, []);
      assert.deepEqual([...toF32(await be.read(be.exp(s)))].map((v) => +v.toFixed(3)), [+Math.exp(4).toFixed(3)]);
      const x = (await f32([2, 3], [0, 1, 2, 3, 4, 5]));
      const t = be.transpose(x, [1, 0]);
      assert.deepEqual(t.shape, [3, 2]);
      assert.deepEqual([...toF32(await be.read(t))], [0, 3, 1, 4, 2, 5]);
      const sl = be.slice(x, [0, 1], [2, 3]);
      assert.deepEqual([...toF32(await be.read(sl))], [1, 2, 4, 5]);
    });

    it("mlx-c errors become exceptions and the backend stays usable", async () => {
      const be = get();
      const [a2, a3] = await Promise.all([f32([2], [1, 2]), f32([3], [1, 2, 3])]);
      assert.throws(() => be.add(a2, a3), /backend-mlx add:.*broadcast/i);
      assert.throws(() => be.reshape(a2, [3]), /backend-mlx reshape/);
      assert.deepEqual([...toF32(await be.read(be.add((await f32([1], [1])), (await f32([1], [2])))))], [3]);
    });

    it("dispose is idempotent, use-after-dispose throws, scope frees intermediates", async () => {
      const be = get();
      const before = be.liveTensors();
      const x = (await f32([2], [1, 2]));
      const kept = be.scope(() => {
        const a = be.exp(x);
        const inner = be.scope(() => ({ y: be.add(a, x), n: 1 }));
        return [inner.y, be.mul(a, a)];
      });
      assert.equal(be.liveTensors(), before + 3); // x + 2 kept
      be.dispose(x);
      be.dispose(x);
      assert.throws(() => be.exp(x), /after dispose/);
      for (const k of kept) be.dispose(k);
      assert.equal(be.liveTensors(), before);
      // a throwing scope frees everything it created
      const one = await f32([1], [1]);
      assert.throws(() => be.scope(() => { be.exp(one); throw new Error("boom"); }), /boom/);
      be.dispose(one);
      assert.equal(be.liveTensors(), before);
    });

    it("scope/dispose keep MLX memory flat across iterations", async () => {
      const be = get();
      const w = be.cast(await be.fromHost(host("f32", [256, 256], new Array(256 * 256).fill(0.01))), "f16");
      const step = () => be.scope(() => { const y = be.linear(be.gelu(w), w); be.flush(y); return null; });
      for (let i = 0; i < 5; i++) step();
      const m0 = be.memory().active, live0 = be.liveTensors();
      for (let i = 0; i < 50; i++) step();
      assert.equal(be.liveTensors(), live0);
      assert.ok(be.memory().active - m0 < 1 << 20, `active memory grew ${be.memory().active - m0} B`);
      be.dispose(w);
    });

    it("sdpa accepts a [B,1,1,L] bool padding mask", async () => {
      const be = get();
      const B = 1, H = 2, L = 3, D = 4;
      const q = await be.fromHost(host("f32", [B, H, L, D], Array.from({ length: B * H * L * D }, (_, i) => Math.sin(i))));
      const v = await be.fromHost(host("f32", [B, H, L, D], Array.from({ length: B * H * L * D }, (_, i) => i % 7)));
      const mask = await be.fromHost(host("bool", [B, 1, 1, L], [1, 1, 0]));
      const out = toF32(await be.read(be.sdpa(q, q, v, mask, 0.5)));
      // With the last key masked, every output is a convex mix of values rows 0 and 1 only.
      const vv = toF32(await be.read(v));
      for (let h = 0; h < H; h++) for (let i = 0; i < L; i++) for (let d = 0; d < D; d++) {
        const o = out[((h * L) + i) * D + d]!;
        const a = vv[(h * L) * D + d]!, c = vv[(h * L + 1) * D + d]!;
        assert.ok(o >= Math.min(a, c) - 1e-4 && o <= Math.max(a, c) + 1e-4);
      }
    });

    it("compile matches eager, supports tensor/array/object outputs and retraces per shape", async () => {
      const be = get();
      let traces = 0;
      const fn = (x: MlxTensor, y: MlxTensor) => {
        traces++;
        return { s: be.add(be.gelu(x), y), p: be.mul(x, y) };
      };
      const cf = be.compile(fn);
      const x = (await f32([3], [-1, 0, 2])), y = (await f32([3], [1, 2, 3]));
      const eager = fn(x, y);
      const r = cf(x, y);
      assert.deepEqual([...toF32(await be.read(r.s))], [...toF32(await be.read(eager.s))]);
      assert.deepEqual([...toF32(await be.read(r.p))], [-1, 0, 6]);
      cf(x, y);
      assert.equal(traces, 2, "one eager call + one trace for this shape");
      cf((await f32([2], [1, 1])), (await f32([2], [1, 1])));
      assert.equal(traces, 3, "new shape retraces");
      const arr = be.compile((a: typeof x) => [be.exp(a), a])(x);
      assert.equal(arr.length, 2);
      assert.deepEqual([...toF32(await be.read(arr[1]!))], [-1, 0, 2]);
      const bad = be.compile((a: typeof x) => { throw new Error("inside trace"); return a; });
      assert.throws(() => bad(x), /inside trace/);
      // backend still healthy afterwards
      assert.deepEqual([...toF32(await be.read(be.add(x, y)))], [0, 2, 5]);
    });

    it("cpu device computes the same results", async () => {
      const cpu = createMlxBackend({ device: "cpu" });
      const x = await cpu.fromHost(host("f32", [2, 2], [1, 2, 3, 4]));
      assert.deepEqual([...toF32(await cpu.read(cpu.matmul(x, x)))], [7, 10, 15, 22]);
      const ident = (a: typeof x) => a;
      assert.equal(cpu.compile(ident), ident, "compile is the identity on cpu");
      cpu.destroy();
    });
  });
}
