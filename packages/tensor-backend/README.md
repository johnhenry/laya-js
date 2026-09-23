# @johnhenry/tensor-backend

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Ftensor-backend.svg)](https://www.npmjs.com/package/@johnhenry/tensor-backend)

The op contract every transformer-inference backend in laya-js implements
(CPU reference, native MLX, WebGPU), plus a data-driven conformance suite
that checks a backend against golden outputs from Python MLX.

## Install

```bash
npm install @johnhenry/tensor-backend
bun add @johnhenry/tensor-backend
deno add jsr:@johnhenry/tensor-backend
```

Types plus a small host-side helper library: Node ≥ 24, Bun ≥ 1.2, Deno and
browsers. The conformance runner's default fixture loader uses `node:fs`
(browsers fetch the JSON and pass it in).

```ts
import type { Backend, Tensor } from "@johnhenry/tensor-backend";
import { host } from "@johnhenry/tensor-backend";

// Code written against Backend runs on every implementation.
function attentionScores<T extends Tensor>(b: Backend<T>, q: T, k: T): T {
  return b.scope(() => b.softmax(b.scale(b.matmul(q, b.transpose(k, [0, 2, 1])), 0.125), -1));
}

const x = host("f32", [2, 3], [1, 2, 3, 4, 5, 6]); // a HostTensor for backend.fromHost(x)
```

Checking a new backend (Node's `node:test` or `bun:test`):

```ts
import { describe, it } from "node:test";
import { loadOpCases, runConformance, type TestApi } from "@johnhenry/tensor-backend/conformance";
import { createMyBackend } from "./my-backend.ts";

runConformance(() => createMyBackend(), loadOpCases(), { describe, it: it as unknown as TestApi["it"] });
```

## API

- **`Backend<T>`** — the contract. Transfer and lifetime: `supports(dtype)`,
  `fromHost`, `read` (the only async op), `dispose`, `scope(fn)`, optional
  `flush` and `destroy`. Shape ops: `reshape`, `transpose`, `slice`, `split`,
  `concat`, `cast`. Elementwise: `add`, `sub`, `mul`, `div`, `maximum`,
  `where`, `scale`, `exp`, `log`, `relu`, `gelu`. Reductions: `sum`, `max`,
  `softmax`, `sort`. Linear algebra and fused kernels: `matmul`, `linear`,
  `layerNorm`, `embedding`, `gatherRows`, `rope`, `sdpa`. Optional:
  `geglu`, `meanPool`, `compile`.
- **Types** — `DType` (`"f32" | "f16" | "bf16" | "i32" | "bool"`), `Shape`,
  `HostData`, `HostTensor`, `Tensor`.
- **Host helpers** — `host(dtype, shape, data?)`, `allocHost`, `sizeOf`,
  `toF32`, `f32ToBf16Bits` / `bf16BitsToF32`, `toMathPlusArgs` (a
  `HostTensor` in the argument shape of math-plus `Tensor.fromTypedArray`).
- **Default compositions** — `geglu(b, x)` and `meanPool(b, x, mask)` for
  backends without the fused op.
- **`@johnhenry/tensor-backend/conformance`** — `loadOpCases(url?)`,
  `runConformance(make, cases, { describe, it })`, `callOp`, `decodeTensor`,
  `assertClose`. The bundled `fixtures/ops.json` has 49 cases over 30 ops,
  generated from Python MLX 0.32.2; each case runs in f32 and, unless marked
  `f32Only`, in f16 with its own tolerances.

Design rules (from `src/index.ts`): backends are passed explicitly (there is
no global default); ops are synchronous and return opaque handles, so a
backend may evaluate lazily (MLX graphs, queued WebGPU passes); masks and
index tensors are built on the host, so the interface has no comparison or
iota ops.

## Limitations

- The contract covers what a ModernBERT-style encoder and the Laya decision
  heads need, not a general tensor library. Use
  [math-plus](https://github.com/johnhenry/math-plus) for that.
- `sdpa` takes a bool mask only (true = attend); additive float masks are
  not part of the contract. Fully masked rows are undefined behaviour.
- `loadOpCases()` without an argument reads the bundled JSON with `node:fs`;
  from JSR or in a browser, fetch `fixtures/ops.json` yourself and pass the
  parsed cases.
- Changing this contract means updating every backend in the repo.

## Family

Part of **[laya-js](https://github.com/johnhenry/laya-js#readme)**, Laya typed decisions in JavaScript on MLX, WebGPU and CPU — see its [package map](https://github.com/johnhenry/laya-js#which-package-do-i-want) and [results](https://github.com/johnhenry/laya-js#results).

- Every backend implements this contract and runs its conformance suite: [`@johnhenry/backend-cpu`](https://github.com/johnhenry/laya-js/tree/main/packages/backend-cpu) (the f32 oracle), [`@johnhenry/backend-mlx`](https://github.com/johnhenry/laya-js/tree/main/packages/backend-mlx), [`@johnhenry/backend-webgpu`](https://github.com/johnhenry/laya-js/tree/main/packages/backend-webgpu).
- [`@johnhenry/modernbert`](https://github.com/johnhenry/laya-js/tree/main/packages/modernbert) and [`@johnhenry/laya`](https://github.com/johnhenry/laya-js/tree/main/packages/laya) are written only against `Backend`, so they run on any of them.
- dtype names and host layouts match `@johnhenry/math-plus-tensor-core` ([math-plus](https://github.com/johnhenry/math-plus)); `toMathPlusArgs` hands a `HostTensor` to `Tensor.fromTypedArray` without a copy.

## License

Apache-2.0.
