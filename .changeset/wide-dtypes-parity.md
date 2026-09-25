---
"@johnhenry/tensor-backend": minor
---

`DType` grows from 5 to the full 13-dtype set that `@johnhenry/math-plus-tensor-core` already supports: `u8 i8 u16 i16 u32 u64 i64 f64` are new, alongside the existing `f32 f16 bf16 i32 bool`. `u64`/`i64` host storage is `BigUint64Array`/`BigInt64Array` (JS `bigint` elements) -- a first for this contract, mirroring tensor-core's own `isBigIntDType` handling. `div` excludes `u64`/`i64` (integer division semantics differ from NumPy's true division; cast to a float dtype first, matching tensor-core's `BIGINT_OPS`). `cumsum`'s doc comment is corrected: confirmed against real MLX that integer dtypes wider than bool wrap on overflow within their own width rather than auto-promoting (only bool promotes, to i32). Not every backend implements every new dtype -- check `supports(dtype)`; `f64` and the smaller integer dtypes are hardware/spec-capped on some backends (see each backend's own changelog).
