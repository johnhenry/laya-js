# Changelog

## 0.1.1

### Patch Changes

- a6d61c7: Licensing by origin. Original packages (tensor-backend, backend-cpu, backend-mlx, backend-webgpu, pyjson, hf-cache) are relicensed from Apache-2.0 to MIT. Packages that port laya-mlx / Laya code stay Apache-2.0 with their NOTICE. laya, laya-router and laya-cli are included to ship alongside `@johnhenry/math-plus-safetensors` on npm.
- Updated dependencies [a6d61c7]
  - @johnhenry/backend-cpu@0.1.1
  - @johnhenry/hf-cache@0.1.1
  - @johnhenry/pyjson@0.1.1
  - @johnhenry/tensor-backend@0.1.1

## 0.1.0

### Minor Changes

- 7fecf44: Initial release: the laya-js package family (0.1.0). See each package's CHANGELOG.md for what ships, and the root README for parity results.

### Patch Changes

- Updated dependencies [7fecf44]
  - @johnhenry/backend-cpu@0.1.0
  - @johnhenry/hf-cache@0.1.0
  - @johnhenry/laya-core@0.1.0
  - @johnhenry/modernbert@0.1.0
  - @johnhenry/pyjson@0.1.0
  - @johnhenry/tensor-backend@0.1.0

First npm distribution of `@johnhenry/laya` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **`load()` / `predict()` / `createAgent()` / shortlist**, a port of laya-mlx `Agent`/`load`: MLX results bit-identical to Python; 63/63 argmax on all three checkpoints on MLX and WebGPU in f32 and f16 (12 configurations). [8bfcd6c](https://github.com/johnhenry/laya-js/commit/8bfcd6c).
- **`@johnhenry/math-plus-safetensors` is a semver dependency (`^0.0.0`)** instead of a `file:` link into a local math-plus worktree.
