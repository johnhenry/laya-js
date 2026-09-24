---
"@johnhenry/tensor-backend": minor
"@johnhenry/backend-cpu": minor
"@johnhenry/backend-mlx": minor
"@johnhenry/backend-webgpu": minor
---

**General-numerics ops** (math-plus RFC 0001 §12 Q7, closes #2): the contract gains optional `equal`, `notEqual`, `less`, `lessEqual`, `greater`, `greaterEqual`, `logicalAnd`, `logicalOr`, `logicalNot`, `sqrt`, `rsqrt`, `pow`, `neg`, `abs`, `tanh`, `sigmoid`, `erf`, `argmax`, `argmin` (i32), `mean`, `min` and `cumsum`. Call them through the new `compose.ts` helpers (`erf(b, x)`, `less(b, x, y)`, …): each uses the native kernel when the backend has one and a default composition otherwise. `cumsum` has no composition (required if used) and composed `argmax`/`argmin` need a native `cumsum` (`NATIVE_ONLY_OPS`, `COMPOSITION_NEEDS`, `hasNative`). The composed `erf` evaluates math-plus tensor-core's canonical algorithm.

- backend-cpu, backend-mlx and backend-webgpu implement every op natively (CPU in f64 with the double-precision `erf`; MLX through mlx-c, extending the FFI bindings over both mlx-c ABIs; WebGPU with n-ary/reduction WGSL kernels, a C-semantics `pow`, and an f32 lowering of the canonical `erf`, < 2.5e-7 absolute).
- `createCpuBackend()` now returns `CpuBackend` (`Backend<CpuTensor>` with those optional ops required); `MlxBackend` and `WebGpuBackend` declare them too.
- Conformance: a second fixture file, `fixtures/ops-numerics.json` (54 cases over the 22 ops, generated with MLX on Metal by `scripts/gen_numerics_cases.py`), loaded by `loadOpCases()` alongside `ops.json`; a **bf16 pass** (tolerance floor 5e-2) for backends that `supports("bf16")`, skipping `f32Only` cases; `withoutOptionalOps(b)` to check the default compositions on a real backend; `nativeOnly` cases skipped when an op runs composed.
