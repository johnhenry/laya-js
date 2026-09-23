# Changelog

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
