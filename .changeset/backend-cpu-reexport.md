---
"@johnhenry/backend-cpu": minor
---

`@johnhenry/backend-cpu` is now a thin re-export of `@johnhenry/math-plus-tensor-cpu`, the CPU reference backend math-plus owns (math-plus RFC 0001 §12 Q3, johnhenry/math-plus#144), built on math-plus tensor-core's kernels. `createCpuBackend`, `CpuTensor`, `CpuBackend`, `erf`, `erfc` and `geluScalar` are unchanged. Removed: the raw `gemmNT` export (the GEMM lives in `@johnhenry/math-plus-tensor-core/kernels`). NaN handling in `max`/`min`/`argmax`/`argmin` now follows tensor-core (backend-defined per the contract). New code should depend on `@johnhenry/math-plus-tensor-cpu` directly; this package will be deprecated later.
