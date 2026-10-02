---
"@johnhenry/tensor-backend": minor
---

`runConformance` now runs a case pinned to its own dtype (any non-float input or output, or a `cast` target) only on a backend whose `supports()` includes that dtype. Before, only the run dtype (f32/f16/bf16) was checked, so every backend had to implement all 13 dtypes to pass the shared suite, contradicting the contract ("check `supports(dtype)`"). This unblocks backends that permanently lack a dtype (f64 on WebGPU) and backends whose engine has not gained the new dtypes yet (backend-cpu on the published math-plus-tensor-cpu).
