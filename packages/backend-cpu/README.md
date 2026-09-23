# @johnhenry/backend-cpu

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Fbackend-cpu.svg)](https://www.npmjs.com/package/@johnhenry/backend-cpu)

Pure-TypeScript f32 reference backend for
[`@johnhenry/tensor-backend`](../tensor-backend). No native code, no WASM,
no dependencies beyond the contract package: it runs anywhere JavaScript
runs and is the numerical oracle the MLX and WebGPU backends are compared
against.

## Install

```bash
npm install @johnhenry/backend-cpu
bun add @johnhenry/backend-cpu
deno add jsr:@johnhenry/backend-cpu
```

Runs anywhere JavaScript runs: Node ≥ 24, Bun ≥ 1.2, Deno and browsers. No native code.

```ts
import { createCpuBackend } from "@johnhenry/backend-cpu";

const cpu = createCpuBackend();
const x = cpu.fromHost({ dtype: "f32", shape: [2, 3], data: new Float32Array([1, 2, 3, 4, 5, 6]) });
const y = cpu.scope(() => cpu.softmax(cpu.scale(x, 2), -1)); // intermediates freed
console.log(await cpu.read(y));
```

## API

- `createCpuBackend(): Backend<CpuTensor>` — independent instance; implements
  every required op plus the optional `geglu`, `meanPool`, `flush` (no-op)
  and `destroy`.
- `class CpuTensor` — `shape`, `dtype`, `data` (row-major `Float32Array` |
  `Int32Array` | `Uint8Array`; may be shared between tensors, never mutate),
  `disposed`.
- `erf(x)`, `erfc(x)`, `geluScalar(x)` — double-precision special functions
  (≈1e-15 relative vs libm) used by `gelu`.
- `gemmNT(A, aOff, lda, B, bOff, ldb, C, cOff, ldc, M, N, K)` — the single
  matmul kernel (C = A·Bᵀ, 4×4 register blocking, f64 accumulation).

## Behaviour

- Storage: f32 for all floats. `fromHost` widens f16 (`Float16Array`) and
  bf16 (raw `Uint16Array` bits) to f32; i32 → `Int32Array`; bool → `Uint8Array`.
- `supports("f16" | "bf16")` is **false**; `cast(x, "f16" | "bf16")` throws.
- Eager and synchronous; `read` resolves immediately with a copy.
- `reshape` and same-dtype `cast` share the buffer (no copy).
- Reductions, softmax, LayerNorm, attention and matmul accumulate in f64,
  then round to f32 once. GELU is exact erf GELU.
- `rope` emulates the f32 angle computation of MLX/PyTorch
  (`inv_freq` and `pos · inv_freq` rounded to f32).
- `sdpa` accepts bool masks (true = attend) or additive float masks,
  broadcast to `[B, H, Lq, Lk]`; K/V may have fewer heads (GQA). Fully
  masked rows return zeros (undefined behaviour in the contract).
- `scope(fn)` tracks every tensor created inside `fn` (nested scopes
  supported); returned tensors (directly, or one level deep in an
  array/object) move to the enclosing scope, everything else is released.
  `dispose` is idempotent; using a disposed tensor throws.

## Performance

About 5–8 GFLOP/s single-threaded on an Apple M2 (Node 24 / Bun 1.2).
See `@johnhenry/laya`'s README for measured end-to-end timings on the
published checkpoints. Good for tests, parity checks and small models; use
`@johnhenry/backend-mlx` or `@johnhenry/backend-webgpu` for real workloads.

## Limitations

- Single-threaded; no SIMD/WASM kernels.
- f32 only (no f16/bf16 compute), so f16-specific numerics are not reproduced.
- Sliding-window attention computes the full L×L score matrix and masks it.
- Float weights are held as f32 in memory (2× the size of fp16 checkpoints).

## Tests

`npm test` (typecheck + node:test) and `npm run test:bun` run the shared
conformance suite (`@johnhenry/tensor-backend/conformance`, MLX-generated
cases) plus backend-specific tests.

## Family

Part of **[laya-js](https://github.com/johnhenry/laya-js#readme)**, Laya typed decisions in JavaScript on MLX, WebGPU and CPU — see its [package map](https://github.com/johnhenry/laya-js#which-package-do-i-want) and [results](https://github.com/johnhenry/laya-js#results).

- Implements [`@johnhenry/tensor-backend`](https://github.com/johnhenry/laya-js/tree/main/packages/tensor-backend) and is the numerical reference the MLX and WebGPU backends are tested against.
- [`@johnhenry/laya`](https://github.com/johnhenry/laya-js/tree/main/packages/laya) depends on it as the always-available fallback.

## License

MIT.
