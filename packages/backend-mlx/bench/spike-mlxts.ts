// Not a dependency any more: run `npm i --no-save @nielspeter/mlx-ts` first (macOS arm64 only).
// Spike: @nielspeter/mlx-ts — import cost, per-op overhead, matmul.
const t0 = performance.now();
const mx: any = await import("@nielspeter/mlx-ts");
const t1 = performance.now();
console.log("runtime", mx.backend.version, "lib", mx.LIBMLXC, "import ms", (t1 - t0).toFixed(1));
const { fromF32, tidy, evalAll } = mx;
const a = fromF32(new Float32Array([1, 2, 3, 4]), [4]);
const b = fromF32(new Float32Array([10, 20, 30, 40]), [4]);
console.log("add", Array.from(a.add(b).toF32()));
// per-op overhead: 10k tiny adds (graph build only, then one eval)
for (let rep = 0; rep < 3; rep++) {
  const s = performance.now();
  tidy(() => { let x = a; for (let i = 0; i < 10000; i++) x = x.add(b); evalAll(x); return null; });
  const e = performance.now();
  console.log(`10k chained adds build+eval: ${(e - s).toFixed(1)} ms (${((e - s) * 1000 / 10000).toFixed(2)} us/op)`);
}
for (let rep = 0; rep < 3; rep++) {
  const s = performance.now();
  tidy(() => { for (let i = 0; i < 10000; i++) a.add(b); return null; });
  const e = performance.now();
  console.log(`10k independent adds (build+free only): ${((e - s) * 1000 / 10000).toFixed(2)} us/op`);
}
