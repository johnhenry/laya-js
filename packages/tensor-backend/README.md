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

const x = host("f32", [2, 3], [1, 2, 3, 4, 5, 6]); // a HostTensor for `await backend.fromHost(x)`
```

Optional ops are called through their helper, which uses the backend's
native kernel or a default composition:

```ts
import { argmax, erf, less } from "@johnhenry/tensor-backend";

const e = erf(b, x);          // b.erf(x) when the backend has it, else composed
const small = less(b, x, y);  // bool
const best = argmax(b, x, -1); // i32
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
  `fromHost` and `read` (both async: device transfers return Promises in
  both directions), `dispose`, `scope(fn)`, optional `flush` and `destroy`. Shape ops: `reshape`, `transpose`, `slice`, `split`,
  `concat`, `cast`. Elementwise: `add`, `sub`, `mul`, `div`, `maximum`,
  `where`, `scale`, `exp`, `log`, `relu`, `gelu`. Reductions: `sum`, `max`,
  `softmax`, `sort`. Linear algebra and fused kernels: `matmul`, `linear`,
  `layerNorm`, `embedding`, `gatherRows`, `rope`, `sdpa`. Optional:
  `geglu`, `meanPool`, `compile`, and the general-numerics ops below.
- **Types** — `DType` (`"f32" | "f16" | "bf16" | "i32" | "bool"`), `Shape`,
  `HostData`, `HostTensor`, `Tensor`.
- **Host helpers** — `host(dtype, shape, data?)`, `allocHost`, `sizeOf`,
  `toF32`, `f32ToBf16Bits` / `bf16BitsToF32`, `toMathPlusArgs` (a
  `HostTensor` in the argument shape of math-plus `Tensor.fromTypedArray`).
- **Optional-op helpers** (`compose.ts`) — `geglu(b, x)`, `meanPool(b, x, mask)`
  and one helper per general-numerics op. Each calls the native op when the
  backend has it and a default composition from required ops otherwise.
  Compositions never upload (constants come from `zerosLike` / `onesLike` /
  `fullLike`, derived on the device), so they stay synchronous and
  traceable by `compile`. `hasNative(b, op)`, `NUMERICS_OPS`,
  `NATIVE_ONLY_OPS`, `COMPOSITION_NEEDS`.
- **General numerics** (optional; math-plus RFC 0001 §12 Q7). Elementwise ops
  broadcast like numpy.

  | Op | Result | Default composition |
  | --- | --- | --- |
  | `equal`, `notEqual`, `less`, `lessEqual`, `greater`, `greaterEqual` | bool | `sub` → `relu` → `cast(…, "bool")`; exact for finite inputs |
  | `logicalAnd`, `logicalOr`, `logicalNot` (nonzero is true) | bool | 0/1 arithmetic + `cast` |
  | `sqrt`, `rsqrt` | float | exp(±½·log x) |
  | `pow(a, b)` | float | exp(b·log a); a > 0 only (a native `pow` handles a negative base with an integral exponent) |
  | `neg`, `abs` | keeps f32/f16/bf16/i32 | `scale(x, −1)` / `maximum(x, −x)` |
  | `tanh`, `sigmoid` | float | 2σ(2x) − 1, 1 / (1 + e⁻ˣ) |
  | `erf` | float | math-plus's canonical algorithm (series + continued fraction), ≈1e-7 absolute |
  | `argmax`, `argmin` (first index on ties) | i32 | `max`/`min` + `equal` + `where` over an iota; **needs a native `cumsum`** |
  | `mean` | float (integer input → f32) | `sum` · 1/n |
  | `min` | keeps dtype | −max(−x) |
  | `cumsum` (inclusive) | floats keep dtype; i32/bool → i32 | **none: required if used** |

  Integer inputs of float-valued ops compute in f32. NaN handling is
  backend-defined (WebGPU may assume no NaNs).
- **`@johnhenry/tensor-backend/conformance`** — `loadOpCases(url?)`,
  `runConformance(make, cases, { describe, it })`, `withoutOptionalOps(b, ops?)`
  (hides optional ops so the suite checks the default compositions),
  `callOp`, `decodeTensor`, `assertClose`, `OP_CASE_FILES`,
  `REDUCED_TOLERANCE`. `loadOpCases()` loads both bundled files:
  `fixtures/ops.json` (49 cases over 30 ops, from laya-mlx's
  `scripts/dump_js_fixtures.py`) and `fixtures/ops-numerics.json` (54 cases
  over the 22 general-numerics ops, from `scripts/gen_numerics_cases.py`),
  both generated with Python MLX 0.32.2 on Metal. Every case runs in f32; unless marked
  `f32Only`, it also runs in f16 (tolerance floor 2e-2) and bf16 (5e-2,
  for bf16's 8-bit mantissa) when the backend `supports` them. Cases marked
  `nativeOnly` (outside a composition's domain) are skipped when the op
  runs composed.

### Regenerating the numerics fixtures

```bash
cd packages/tensor-backend
uv run --with mlx==0.32.2 --with numpy python scripts/gen_numerics_cases.py
```

The script computes every output with MLX on the GPU, cross-checks it
against NumPy (or `math.erf`), and writes `fixtures/ops-numerics.json`.
Float inputs are exactly representable in bf16, so the f16/bf16 runs see
the same inputs as the f32 run (no rounding-induced comparison or argmax
flips). Never hand-edit the JSON.

## Migrating from 0.1

`fromHost` returns a `Promise` (math-plus RFC 0001 §12 Q2):

```ts
// 0.1
const x = backend.fromHost(h);
// 0.2
const x = await backend.fromHost(h);
// several uploads: start them all, await once
const [ids, mask] = await Promise.all([backend.fromHost(hIds), backend.fromHost(hMask)]);
```

Code that built constants with `fromHost` inside a synchronous op sequence
(for example inside `scope` or a function passed to `compile`) should
upload them beforehand, or derive them on the device with `onesLike` /
`fullLike`. Backend implementers: make `fromHost` `async` (copying the host
data at call time is fine) and run the conformance suite, which now also
has a bf16 pass. The general-numerics ops are optional; the helpers cover
backends that do not implement them.

Design rules (from `src/index.ts`): backends are passed explicitly (there is
no global default); ops are synchronous and return opaque handles, so a
backend may evaluate lazily (MLX graphs, queued WebGPU passes); device
transfers are async in both directions (`fromHost` and `read`); masks and
index tensors for the transformer hot path are built on the host; the
optional general-numerics ops serve device-side code elsewhere.

## Limitations

- The contract covers what a ModernBERT-style encoder and the Laya decision
  heads need, plus a small optional general-numerics section. It is not a
  general tensor library. Use [math-plus](https://github.com/johnhenry/math-plus) for that.
- `cumsum` has no default composition, and composed `argmax`/`argmin` need a
  native `cumsum`: the helpers throw on a backend without it.
- Default compositions are for finite inputs: composed comparisons can
  misjudge NaN/±inf and overflowing i32 differences, composed `pow` returns
  NaN for a negative base, and composed `erf` costs ~150 dispatches.
- `sdpa` takes a bool mask only (true = attend); additive float masks are
  not part of the contract. Fully masked rows are undefined behaviour.
- `loadOpCases()` without an argument reads the bundled JSON with `node:fs`;
  from JSR or in a browser, fetch `fixtures/ops.json` and
  `fixtures/ops-numerics.json` yourself and pass the concatenated cases.
- Changing this contract means updating every backend in the repo.

## Family

Part of **[laya-js](https://github.com/johnhenry/laya-js#readme)**, Laya typed decisions in JavaScript on MLX, WebGPU and CPU — see its [package map](https://github.com/johnhenry/laya-js#which-package-do-i-want) and [results](https://github.com/johnhenry/laya-js#results).

- Every backend implements this contract and runs its conformance suite: [`@johnhenry/backend-cpu`](https://github.com/johnhenry/laya-js/tree/main/packages/backend-cpu) (the f32 oracle), [`@johnhenry/backend-mlx`](https://github.com/johnhenry/laya-js/tree/main/packages/backend-mlx), [`@johnhenry/backend-webgpu`](https://github.com/johnhenry/laya-js/tree/main/packages/backend-webgpu).
- [`@johnhenry/modernbert`](https://github.com/johnhenry/laya-js/tree/main/packages/modernbert) and [`@johnhenry/laya`](https://github.com/johnhenry/laya-js/tree/main/packages/laya) are written only against `Backend`, so they run on any of them.
- dtype names and host layouts match `@johnhenry/math-plus-tensor-core` ([math-plus](https://github.com/johnhenry/math-plus)); `toMathPlusArgs` hands a `HostTensor` to `Tensor.fromTypedArray` without a copy.

## License

MIT.
