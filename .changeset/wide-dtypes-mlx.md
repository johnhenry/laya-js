---
"@johnhenry/backend-mlx": minor
---

Full dtype parity with the widened `@johnhenry/tensor-backend` contract: `u8 i8 u16 i16 u32 u64 i64` work on both `gpu` and `cpu` devices; `f64` works on `cpu` only (`supports("f64")` is `this.device === "cpu"`) since no Apple GPU has double-precision hardware and MLX's own `float64` throws if evaluated on the GPU stream. `MLX_DTYPE`/`FROM_MLX`/`BYTES`/`HOST_CTOR` now cover all 13 dtypes, using the real mlx-c `mlx_dtype` enum values (confirmed against `ml-explore/mlx-c`'s `mlx/c/array.h`, not assumed). `u64`/`i64` host storage is `BigUint64Array`/`BigInt64Array`. Verified against real MLX 0.32.2 on Apple Silicon: 40/40 tests, 0 skipped, including the full conformance suite on both devices with new fixture cases generated directly from MLX (`scripts/gen_numerics_cases.py`).
