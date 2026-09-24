# Changelog

## 0.5.1

### Patch Changes

- 5067f71: `sleepWhileWaiting`: the readback-wait estimate is now keyed by dispatch count **and** total workgroups (i.e. per shape). Before, a small input after a large one with the same op graph slept for most of the large input's GPU time (measured 594 ms instead of 31 ms, ~15 calls to recover) on Node/Bun. New `rt.waitEstimates` exposes the estimates for diagnostics.

## 0.5.0

### Minor Changes

- 0d5072b: backend-webgpu: quantized Linears (q8/q4 kept on the device) are now at least as fast as fp16 end to end on WebGPU (M2: English q8 one question 51.3 vs 54.6 ms, 16 questions 641 vs 654 ms; 0.4.0 was 9–16% slower).

  - New `qmv` kernel for M ≤ 48 (64 for N < 2048): 8 threads per weight row, one word load per 8 (or 16) values, unpacked and scaled in registers, f32 partials reduced in workgroup memory or with subgroup shuffles (the `subgroups` feature is now requested when available). 1.1–5.7× fp16 GFLOP/s for M ≤ 8, median 1.06× for 16–64, and used at every M when subgroup matrices are unavailable (browsers without the flag), where 0.4.0 fell back to the direct kernel.
  - Subgroup-matrix kernel with quantized B: 8-value loads, raw words and scale fetched into registers and dequantized when stored to workgroup memory (after the MMAs), A converted late too, and quantized-specific tile rules (BM 64/48/32, split-K). median 1.02× fp16 for M ≥ 93 (0.80–0.92× before).
  - Dequantization without int→float conversions (f16 exponent trick). With f16 activations every path still multiplies by exactly fl16(q·scale + bias); new tests check this weight by weight (subnormals included) and that random products round like the exact sum.
  - `GemmConfig.quant` (`QUANT_GEMM_DEFAULT`, `QUANT_GEMM_NAVIGATOR` for `navigator.gpu`, `quant: null` restores the 0.4.0 choice), `QmvGemmConfig`, `WebGpuBackend.subgroupSize`, `GemmChoice` "qmv", and `tuneGemm(shapes, { quantized: { bits, groupSize?, mode? } })`.
  - Remaining gaps (documented): 0.92–0.97× on the large-N Laya shapes around M = 33 and 0.83–0.97× in a few cells at M = 93–128.

  laya: `load(url)` (http(s) checkpoints) now passes `quantized` to the weight reader, so quantized checkpoints loaded from a URL stay quantized on MLX/WebGPU instead of always being dequantized on the host.

## 0.4.0

### Minor Changes

- 07941a4: Native quantized weights: q8/q4 matrices stay packed in a `u32` buffer with f16 scales/biases; every Linear kernel (skinny, subgroup-matrix incl. split-K, direct, tiled) dequantizes in its B tile load and accumulates in f32; `quantizedEmbedding` is a dequantizing gather. New `bench/quantized-gemm.ts`.

### Patch Changes

- Updated dependencies [07941a4]
  - @johnhenry/tensor-backend@0.3.0

## 0.3.1

### Patch Changes

- 03ff794: Runtime hooks for math-plus's WebGPU convergence (johnhenry/math-plus#146, RFC 0001 §12 Q6 path (a), under which math-plus-tensor-webgpu becomes a facade over this package). All changes are additive. This is a patch so that `^0.3.0` peer ranges keep matching.

  - `elementwise(expr, xs, { outDtype?, helpers? })`: a custom n-ary elementwise kernel with broadcasting over `x0, x1, …`, cached per expression. math-plus compiles its tensor-compile IR to it.
  - `empty(shape, dtype)`: a scope-tracked output tensor from the pool, for custom kernels.
  - `wrapBuffer(buffer, shape, dtype, offset?)`: a view of a caller-owned `GPUBuffer`, which is never pooled or destroyed (`Storage.external`).
  - `Runtime`, `Storage` and the kernel types (`KernelSource`, `BindingSpec`, `ParamSpec`, `ParamType`, `CompiledKernel`, `RuntimeStats`) are exported, and `rt.kernel` / `rt.dispatch` are documented.
  - `createWebGpuBackend({ device, adapter })`: subgroup-matrix detection now works for a device you pass in. Dawn doesn't mirror `subgroupMatrixConfigs` onto `device.adapterInfo`.
  - `sleepThresholdMs` option (default 3, unchanged): short readbacks can lose 15–60% latency to the pre-read sleep.
  - Fix: `sdpa` now respects `maxComputeWorkgroupStorageSize`. The fast kernel (~20 KiB at head dim 64) runs only where it fits, and the generic kernel's tiles halve until they fit. On a device with the 16 KiB WebGPU default, both previously failed validation for most head dims: 48, 64 and 128 at ~20–26 KiB.

## 0.3.0

### Minor Changes

- f52f7c4: Performance: faster subgroup-matrix GEMM and cheaper dispatch.

  - Subgroup-matrix Linears: 64×64 workgroup tiles (8×4 fragments per subgroup) for large M, chosen per shape by grid size and row padding (`minGroups`, `maxPad` in `GemmConfig.sg`); f16 operands are loaded 8 halves at a time (`vec4<u32>` + `unpack2x16float`); unpadded staging panels. Still f32 accumulation. Optional split-K (`splitK`), double buffering (`db`) and a whole-block epilogue (`epi: "block"`) are available for tuning but off by default (slower on Apple M2).
  - `backend.tuneGemm(shapes)` measures the Linear kernel choice per exact shape on the current device; results are shared by backends on the same `GPUDevice` (`backend.gemmTuning`) and can be restored with the `gemmTuning` option.
  - The buffer pool hands out the lowest-id free buffer of a size class instead of LIFO, so repeated forwards reuse the same buffers and bind groups are always cache hits (previously ~170 new bind groups per ModernBERT-large forward).
  - `firstBatch` option (default 24): the first submit after the GPU goes idle carries fewer dispatches, so the GPU starts sooner.
  - Bool uploads convert with a plain loop (the B×L×L sliding-window masks were ~40 ms at B=16, L=256).
  - `rope` computes both halves of a rotation pair per thread.
  - `GEMM_V020` exports the previous GEMM defaults for comparisons.

## 0.2.0

### Minor Changes

- 790e1e0: **Breaking: `fromHost` returns a Promise**, per `@johnhenry/tensor-backend` 0.2. Each backend still copies the host data when `fromHost` is called (CPU: a typed-array copy; MLX: `mlx_array_new_data`; WebGPU: `queue.writeBuffer`), and the Promise is already settled, so a batch of uploads awaited together costs one microtask. The benches and the WebGPU demo await their uploads.

  Migration: `const x = await backend.fromHost(h)` (or `Promise.all` over several).

- 790e1e0: **General-numerics ops** (math-plus RFC 0001 §12 Q7, closes #2): the contract gains optional `equal`, `notEqual`, `less`, `lessEqual`, `greater`, `greaterEqual`, `logicalAnd`, `logicalOr`, `logicalNot`, `sqrt`, `rsqrt`, `pow`, `neg`, `abs`, `tanh`, `sigmoid`, `erf`, `argmax`, `argmin` (i32), `mean`, `min` and `cumsum`. Call them through the new `compose.ts` helpers (`erf(b, x)`, `less(b, x, y)`, …): each uses the native kernel when the backend has one and a default composition otherwise. `cumsum` has no composition (required if used) and composed `argmax`/`argmin` need a native `cumsum` (`NATIVE_ONLY_OPS`, `COMPOSITION_NEEDS`, `hasNative`). The composed `erf` evaluates math-plus tensor-core's canonical algorithm.

  - backend-cpu, backend-mlx and backend-webgpu implement every op natively (CPU in f64 with the double-precision `erf`; MLX through mlx-c, extending the FFI bindings over both mlx-c ABIs; WebGPU with n-ary/reduction WGSL kernels, a C-semantics `pow`, and an f32 lowering of the canonical `erf`, < 2.5e-7 absolute).
  - `createCpuBackend()` now returns `CpuBackend` (`Backend<CpuTensor>` with those optional ops required); `MlxBackend` and `WebGpuBackend` declare them too.
  - Conformance: a second fixture file, `fixtures/ops-numerics.json` (54 cases over the 22 ops, generated with MLX on Metal by `scripts/gen_numerics_cases.py`), loaded by `loadOpCases()` alongside `ops.json`; a **bf16 pass** (tolerance floor 5e-2) for backends that `supports("bf16")`, skipping `f32Only` cases; `withoutOptionalOps(b)` to check the default compositions on a real backend; `nativeOnly` cases skipped when an op runs composed.

### Patch Changes

- Updated dependencies [790e1e0]
- Updated dependencies [790e1e0]
  - @johnhenry/tensor-backend@0.2.0

## 0.1.2

### Patch Changes

- 1adc4c7: Republish with an npm provenance attestation (built and published by GitHub Actions). The earlier versions were published from a local machine without provenance. No code changes.
- Updated dependencies [1adc4c7]
  - @johnhenry/tensor-backend@0.1.2

## 0.1.1

### Patch Changes

- a6d61c7: Licensing by origin. Original packages (tensor-backend, backend-cpu, backend-mlx, backend-webgpu, pyjson, hf-cache) are relicensed from Apache-2.0 to MIT. Packages that port laya-mlx / Laya code stay Apache-2.0 with their NOTICE. laya, laya-router and laya-cli are included to ship alongside `@johnhenry/math-plus-safetensors` on npm.
- Updated dependencies [a6d61c7]
  - @johnhenry/tensor-backend@0.1.1

## 0.1.0

### Minor Changes

- 7fecf44: Initial release: the laya-js package family (0.1.0). See each package's CHANGELOG.md for what ships, and the root README for parity results.

### Patch Changes

- Updated dependencies [7fecf44]
  - @johnhenry/tensor-backend@0.1.0

First npm distribution of `@johnhenry/backend-webgpu` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **WebGPU backend**: tiled f16 GEMM, fused SDPA (flash attention), rope and layerNorm in WGSL; conformance on Node and Bun (Dawn), Deno and Chromium; English checkpoint 63/63 argmax in f32 and f16. [7b327a2](https://github.com/johnhenry/laya-js/commit/7b327a2).
- **Bun upload fix.** Dawn's `writeBuffer` under Bun ignored a TypedArray view's `byteOffset`, so weight views into a shared buffer uploaded the wrong bytes; uploads now copy such views, with a regression test. Fixed in [c15f489](https://github.com/johnhenry/laya-js/commit/c15f489).
- **`@webgpu/types` is now a dependency**, because the published `.d.ts` files mention `GPUDevice` and friends.
