# Changelog

## 0.2.0

### Minor Changes

- 790e1e0: **Breaking: `Backend.fromHost` is async** (`fromHost(t): Promise<T>`), like `read`, so device transfers are async-visible in both directions (math-plus RFC 0001 §12 Q2, closes #1). The conformance harness awaits uploads (and starts a case's uploads together), and the `compose.ts` helpers no longer upload: `meanPool`'s constant is derived on the device, with the new `zerosLike` / `onesLike` / `fullLike` helpers, so every helper stays synchronous and traceable by `compile`.

  Migration: `const x = await backend.fromHost(h)`; batch independent uploads with `Promise.all`. Code that uploaded constants in the middle of a synchronous op sequence (inside `scope` or a `compile`d function) should upload them beforehand or use `onesLike` / `fullLike`. Backend implementers: make `fromHost` `async` (copying the host data at call time is fine).

- 790e1e0: **General-numerics ops** (math-plus RFC 0001 §12 Q7, closes #2): the contract gains optional `equal`, `notEqual`, `less`, `lessEqual`, `greater`, `greaterEqual`, `logicalAnd`, `logicalOr`, `logicalNot`, `sqrt`, `rsqrt`, `pow`, `neg`, `abs`, `tanh`, `sigmoid`, `erf`, `argmax`, `argmin` (i32), `mean`, `min` and `cumsum`. Call them through the new `compose.ts` helpers (`erf(b, x)`, `less(b, x, y)`, …): each uses the native kernel when the backend has one and a default composition otherwise. `cumsum` has no composition (required if used) and composed `argmax`/`argmin` need a native `cumsum` (`NATIVE_ONLY_OPS`, `COMPOSITION_NEEDS`, `hasNative`). The composed `erf` evaluates math-plus tensor-core's canonical algorithm.

  - backend-cpu, backend-mlx and backend-webgpu implement every op natively (CPU in f64 with the double-precision `erf`; MLX through mlx-c, extending the FFI bindings over both mlx-c ABIs; WebGPU with n-ary/reduction WGSL kernels, a C-semantics `pow`, and an f32 lowering of the canonical `erf`, < 2.5e-7 absolute).
  - `createCpuBackend()` now returns `CpuBackend` (`Backend<CpuTensor>` with those optional ops required); `MlxBackend` and `WebGpuBackend` declare them too.
  - Conformance: a second fixture file, `fixtures/ops-numerics.json` (54 cases over the 22 ops, generated with MLX on Metal by `scripts/gen_numerics_cases.py`), loaded by `loadOpCases()` alongside `ops.json`; a **bf16 pass** (tolerance floor 5e-2) for backends that `supports("bf16")`, skipping `f32Only` cases; `withoutOptionalOps(b)` to check the default compositions on a real backend; `nativeOnly` cases skipped when an op runs composed.

## 0.1.2

### Patch Changes

- 1adc4c7: Republish with an npm provenance attestation (built and published by GitHub Actions). The earlier versions were published from a local machine without provenance. No code changes.

## 0.1.1

### Patch Changes

- a6d61c7: Licensing by origin. Original packages (tensor-backend, backend-cpu, backend-mlx, backend-webgpu, pyjson, hf-cache) are relicensed from Apache-2.0 to MIT. Packages that port laya-mlx / Laya code stay Apache-2.0 with their NOTICE. laya, laya-router and laya-cli are included to ship alongside `@johnhenry/math-plus-safetensors` on npm.

## 0.1.0

### Minor Changes

- 7fecf44: Initial release: the laya-js package family (0.1.0). See each package's CHANGELOG.md for what ships, and the root README for parity results.

First npm distribution of `@johnhenry/tensor-backend` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **The backend op contract.** `Backend<T>` with transfer/lifetime (`fromHost`, async `read`, `dispose`, `scope`), shape, elementwise, reduction and fused transformer ops (`linear`, `layerNorm`, `rope`, `sdpa`, `gelu`), plus optional `geglu`/`meanPool`/`compile`. Introduced in [8b963a2](https://github.com/johnhenry/laya-js/commit/8b963a2); `TestApi.it` signature and `fromHost` widening settled in [c2f272d](https://github.com/johnhenry/laya-js/commit/c2f272d).
- **Conformance suite.** `runConformance` over 49 golden cases (30 ops) generated from Python MLX 0.32.2, run in f32 and f16. [8b963a2](https://github.com/johnhenry/laya-js/commit/8b963a2).
