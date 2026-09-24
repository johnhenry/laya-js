# Changelog

## 0.4.0

### Minor Changes

- 07941a4: Native quantized weights: `fromHostQuantized` repacks laya-js q8/q4 matrices into MLX's affine uint32 layout without dequantizing (q4 uploads as-is; symmetric q8 flips each byte's sign bit with bias = −128·scale, bit-exact), `quantizedLinear` is `mlx_quantized_matmul`, `quantizedEmbedding` gathers packed rows and runs `mlx_dequantize`. Groups of 32/64/128.

### Patch Changes

- Updated dependencies [07941a4]
  - @johnhenry/tensor-backend@0.3.0

## 0.3.0

### Minor Changes

- 2b97c2c: Deno 2 support: a `Deno.dlopen` loader next to the `bun:ffi` and koffi ones (same ~60 mlx-c signatures, both mlx-c ABIs; handles cross as `usize`, callbacks are `Deno.UnsafeCallback`s). `backend.info.runtime` can now be `"deno"`. Library resolution keeps its order; under Deno the platform package is also found from the working directory's `node_modules` or Deno's npm cache (`deno add npm:@johnhenry/backend-mlx-darwin-arm64`), and loading the module over https (JSR) no longer throws in `createRequire`/`fileURLToPath`. The conformance suite (f32/f16/bf16 incl. numerics, GPU and CPU) and `@johnhenry/laya` parity on all three checkpoints pass under Deno 2.9.7 (`npm run test:deno`). Now also published to JSR as `jsr:@johnhenry/backend-mlx`.

## 0.2.0

### Minor Changes

- 790e1e0: **Breaking: `fromHost` returns a Promise**, per `@johnhenry/tensor-backend` 0.2. Each backend still copies the host data when `fromHost` is called (CPU: a typed-array copy; MLX: `mlx_array_new_data`; WebGPU: `queue.writeBuffer`), and the Promise is already settled, so a batch of uploads awaited together costs one microtask. The benches and the WebGPU demo await their uploads.

  Migration: `const x = await backend.fromHost(h)` (or `Promise.all` over several).

- 790e1e0: **General-numerics ops** (math-plus RFC 0001 §12 Q7, closes #2): the contract gains optional `equal`, `notEqual`, `less`, `lessEqual`, `greater`, `greaterEqual`, `logicalAnd`, `logicalOr`, `logicalNot`, `sqrt`, `rsqrt`, `pow`, `neg`, `abs`, `tanh`, `sigmoid`, `erf`, `argmax`, `argmin` (i32), `mean`, `min` and `cumsum`. Call them through the new `compose.ts` helpers (`erf(b, x)`, `less(b, x, y)`, …): each uses the native kernel when the backend has one and a default composition otherwise. `cumsum` has no composition (required if used) and composed `argmax`/`argmin` need a native `cumsum` (`NATIVE_ONLY_OPS`, `COMPOSITION_NEEDS`, `hasNative`). The composed `erf` evaluates math-plus tensor-core's canonical algorithm.

  - backend-cpu, backend-mlx and backend-webgpu implement every op natively (CPU in f64 with the double-precision `erf`; MLX through mlx-c, extending the FFI bindings over both mlx-c ABIs; WebGPU with n-ary/reduction WGSL kernels, a C-semantics `pow`, and an f32 lowering of the canonical `erf`, < 2.5e-7 absolute).
  - `createCpuBackend()` now returns `CpuBackend` (`Backend<CpuTensor>` with those optional ops required); `MlxBackend` and `WebGpuBackend` declare them too.
  - Conformance: a second fixture file, `fixtures/ops-numerics.json` (54 cases over the 22 ops, generated with MLX on Metal by `scripts/gen_numerics_cases.py`), loaded by `loadOpCases()` alongside `ops.json`; a **bf16 pass** (tolerance floor 5e-2) for backends that `supports("bf16")`, skipping `f32Only` cases; `withoutOptionalOps(b)` to check the default compositions on a real backend; `nativeOnly` cases skipped when an op runs composed.

### Patch Changes

- Updated dependencies [790e1e0]
- Updated dependencies [790e1e0]
  - @johnhenry/tensor-backend@0.2.0

## 0.1.2

### Patch Changes

- 1adc4c7: Republish with an npm provenance attestation (built and published by GitHub Actions). The earlier versions were published from a local machine without provenance. No code changes.
- Updated dependencies [1adc4c7]
  - @johnhenry/tensor-backend@0.1.2

## 0.1.1

### Patch Changes

- a6d61c7: Licensing by origin. Original packages (tensor-backend, backend-cpu, backend-mlx, backend-webgpu, pyjson, hf-cache) are relicensed from Apache-2.0 to MIT. Packages that port laya-mlx / Laya code stay Apache-2.0 with their NOTICE. laya, laya-router and laya-cli are included to ship alongside `@johnhenry/math-plus-safetensors` on npm.
- Updated dependencies [a6d61c7]
  - @johnhenry/tensor-backend@0.1.1

## 0.1.0

### Minor Changes

- 7fecf44: Initial release: the laya-js package family (0.1.0). See each package's CHANGELOG.md for what ships, and the root README for parity results.

### Patch Changes

- Updated dependencies [7fecf44]
  - @johnhenry/tensor-backend@0.1.0

First npm distribution of `@johnhenry/backend-mlx` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **Native MLX backend over our own mlx-c FFI binding** (`bun:ffi` on Bun, `koffi` on Node): lazy graph, fused MLX kernels, `compile` via `mlx_compile`, two mlx-c ABIs detected at load time. Conformance green in f32/f16 on Node and Bun. [8acf587](https://github.com/johnhenry/laya-js/commit/8acf587); decision record in `docs/mlx-binding-decision.md`.
- **Native distribution.** `koffi` and the new platform package `@johnhenry/backend-mlx-darwin-arm64` are optional dependencies; `libmlxc` resolution order is `libPath` → `$LAYA_MLXC_PATH` (file or directory; `$LAYA_MLXC_LIB` kept as an alias; a set-but-missing path now throws) → platform package → local build → `@nielspeter/mlx-ts-darwin-arm64` → Homebrew. `scripts/build-mlxc.sh` takes `MLXC_OUT` and writes `SHA256SUMS` and Apple's license texts next to the binaries.
