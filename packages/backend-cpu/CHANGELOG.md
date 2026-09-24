# Changelog

## 0.3.0

### Minor Changes

- d6ef9ce: `@johnhenry/backend-cpu` is now a thin re-export of `@johnhenry/math-plus-tensor-cpu`, the CPU reference backend math-plus owns (math-plus RFC 0001 §12 Q3, johnhenry/math-plus#144), built on math-plus tensor-core's kernels. `createCpuBackend`, `CpuTensor`, `CpuBackend`, `erf`, `erfc` and `geluScalar` are unchanged. Removed: the raw `gemmNT` export (the GEMM lives in `@johnhenry/math-plus-tensor-core/kernels`). NaN handling in `max`/`min`/`argmax`/`argmin` now follows tensor-core (backend-defined per the contract). New code should depend on `@johnhenry/math-plus-tensor-cpu` directly; this package will be deprecated later.

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

First npm distribution of `@johnhenry/backend-cpu` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **Pure-TypeScript f32 reference backend.** Passes the full conformance suite; every stage of the tiny Laya checkpoint within 1e-6 of MLX fp32; 63/63 argmax on all three published checkpoints (opt-in, slow). [9bfc247](https://github.com/johnhenry/laya-js/commit/9bfc247).
