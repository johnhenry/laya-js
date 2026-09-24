# @johnhenry/backend-cpu

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Fbackend-cpu.svg)](https://www.npmjs.com/package/@johnhenry/backend-cpu)

> **Now an alias.** Since 0.3.0 this package re-exports
> [`@johnhenry/math-plus-tensor-cpu`](https://github.com/johnhenry/math-plus/tree/main/packages/tensor-cpu),
> the CPU reference backend that math-plus owns (math-plus RFC 0001 §12 Q3,
> [math-plus#144](https://github.com/johnhenry/math-plus/issues/144)). New
> code should depend on `@johnhenry/math-plus-tensor-cpu` directly. This
> package will be deprecated in a later release.

Pure-TypeScript f32 reference backend for
[`@johnhenry/tensor-backend`](../tensor-backend). It is the numerical oracle
that the MLX and WebGPU backends are compared against.

## Install

```bash
npm install @johnhenry/backend-cpu        # or, preferred: @johnhenry/math-plus-tensor-cpu
```

```ts
import { createCpuBackend } from "@johnhenry/backend-cpu";

const cpu = createCpuBackend();
const x = await cpu.fromHost({ dtype: "f32", shape: [2, 3], data: new Float32Array([1, 2, 3, 4, 5, 6]) });
const y = cpu.scope(() => cpu.softmax(cpu.scale(x, 2), -1)); // intermediates freed
console.log(await cpu.read(y));
```

## API

Unchanged from 0.2.0 except for one removal:

- `createCpuBackend()`, `CpuTensor`, `type CpuBackend`, `erf`, `erfc`,
  `geluScalar` are re-exported from `@johnhenry/math-plus-tensor-cpu`.
- **Removed:** `gemmNT`. The GEMM now lives once, in
  `@johnhenry/math-plus-tensor-core/kernels`, and works over packed f64
  panels.

The backend's behaviour (dtype rules, f16/bf16 widening, scopes, the ops it
implements natively, NaN semantics, limitations and benchmarks against 0.2.0)
is documented in
[math-plus-tensor-cpu's README](https://github.com/johnhenry/math-plus/tree/main/packages/tensor-cpu#readme).
Differences from 0.2.0 that callers might notice:

- `max`, `min`, `argmax` and `argmin` no longer propagate NaN NumPy-style.
  They use tensor-core's strict comparisons, and NaN results are
  backend-defined in the contract.
- `where` with a bool value operand and an i32 value operand now returns
  i32.

## Tests

`npm test` and `npm run test:bun` run the shared conformance suite and this
package's tests against the re-exported backend.

## Family

Part of **[laya-js](https://github.com/johnhenry/laya-js#readme)**, Laya typed decisions in JavaScript on MLX, WebGPU and CPU — see its [package map](https://github.com/johnhenry/laya-js#which-package-do-i-want) and [results](https://github.com/johnhenry/laya-js#results).

- Implements [`@johnhenry/tensor-backend`](https://github.com/johnhenry/laya-js/tree/main/packages/tensor-backend) and is the numerical reference the MLX and WebGPU backends are tested against.
- [`@johnhenry/laya`](https://github.com/johnhenry/laya-js/tree/main/packages/laya) depends on it as the always-available fallback.

## License

MIT.
