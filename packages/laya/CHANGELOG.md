# Changelog

## 0.2.0

### Minor Changes

- 790e1e0: **Breaking (lower-level API only): uploads are async** (`@johnhenry/tensor-backend` 0.2). `load()` and `predict()` are unchanged. `createAgent` returns `Promise<LayaAgent>`; `loadDecisionModel` returns `Promise<DecisionModel>` and uploads the encoder and head weights in one batch; `DecisionModel.uploadBatch` and `forwardTensors` are async (a batch's eight tensors upload together); `compiled()` returns an async function. The forward pass's constants (−1e4, 1e-9, 255) are uploaded once with the weights (`DecisionWeights.constants`), so `forwardCore` stays synchronous and the MLX graph is unchanged (English parity 63/63, MLX f32 exact). Load time and single-question `predict` latency are unchanged (see the PR's before/after table). New: `DecisionModel.disposeInputs`.

  Migration: `await createAgent(parts)`, `await loadDecisionModel(...)`, `await model.forwardTensors(batch)`, `await model.uploadBatch(batch)`; code constructing `DecisionWeights` by hand must add `constants`.

- 944e3c0: Quantized checkpoints (q8/q4, dequantize-on-load). `load()` / `readWeights()` detect `__metadata__.laya_quant` in `model.safetensors` and dequantize 8-bit (symmetric, groups of 64) and 4-bit (affine, groups of 64, optional q8 tensors) weights to f16/f32 tensor by tensor while loading, so downloads shrink to ~52% (q8) or ~28–35% (q4) with no backend changes. New `quantizeMatrix` / `dequantizeMatrix` / `quantizeSafetensors` / `quantMetadata` exports; `load()` in Node/Bun now also accepts an http(s) base URL (Range reads, as in browsers). New `laya quantize --model <repo|dir> --bits 8|4 --out <dir>` command writes a complete quantized checkpoint directory.

### Patch Changes

- Updated dependencies [790e1e0]
- Updated dependencies [790e1e0]
- Updated dependencies [790e1e0]
- Updated dependencies [790e1e0]
  - @johnhenry/backend-cpu@0.2.0
  - @johnhenry/tensor-backend@0.2.0
  - @johnhenry/modernbert@0.2.0

## 0.1.2

### Patch Changes

- 1adc4c7: Republish with an npm provenance attestation (built and published by GitHub Actions). The earlier versions were published from a local machine without provenance. No code changes.
- Updated dependencies [1adc4c7]
  - @johnhenry/backend-cpu@0.1.2
  - @johnhenry/hf-cache@0.1.2
  - @johnhenry/laya-core@0.1.1
  - @johnhenry/modernbert@0.1.1
  - @johnhenry/pyjson@0.1.2
  - @johnhenry/tensor-backend@0.1.2

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
