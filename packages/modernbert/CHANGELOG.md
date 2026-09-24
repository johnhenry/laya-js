# Changelog

## 0.3.0

### Minor Changes

- 07941a4: Quantized checkpoints now stay quantized on the device on MLX and WebGPU (less device memory): `load(..., { quantized: "device" | "dequantize" })`, default `"device"` when the backend has native quantized ops, else host dequantization as before. `agent.model.quantizedOnDevice` reports which. modernbert and `DecisionModel` accept `HostQuantized` Linear weights and token/type embeddings (`HostWeight`, `MatrixWeight`, `disposeWeight`); `readWeights(src, { quantized: "device" })` and `hostQuantized(matrix)` hand out packed matrices. Optional peer ranges on backend-mlx/backend-webgpu widened to `^0.4.0`.

### Patch Changes

- Updated dependencies [07941a4]
  - @johnhenry/tensor-backend@0.3.0

## 0.2.0

### Minor Changes

- 790e1e0: **Breaking: loading and running the encoder is async**, because uploads are (`@johnhenry/tensor-backend` 0.2). `loadModernBert` returns `Promise<ModernBert>` and starts every weight upload before awaiting any (validation first; on failure the uploaded tensors are disposed), so load time is unchanged. `forward` and `embed` return Promises; `uploadAs` returns `Promise<T>`.

  New: `uploadInputs` / `encode` / `disposeInputs` (upload a batch's ids and masks together, then run the encoder synchronously, e.g. under `compile`), `EncoderInputs`, and the batch-loading helpers `loadInBatch` and `settleUploads`.

  Migration: `await loadModernBert(...)`, `await encoder.forward(...)`, `await encoder.embed(...)`, `await uploadAs(...)`; for a synchronous encoder pass use `uploadInputs` + `encode`.

### Patch Changes

- Updated dependencies [790e1e0]
- Updated dependencies [790e1e0]
  - @johnhenry/tensor-backend@0.2.0

## 0.1.1

### Patch Changes

- 1adc4c7: Republish with an npm provenance attestation (built and published by GitHub Actions). The earlier versions were published from a local machine without provenance. No code changes.
- Updated dependencies [1adc4c7]
  - @johnhenry/tensor-backend@0.1.2

## 0.1.0

### Minor Changes

- 7fecf44: Initial release: the laya-js package family (0.1.0). See each package's CHANGELOG.md for what ships, and the root README for parity results.

### Patch Changes

- Updated dependencies [7fecf44]
  - @johnhenry/tensor-backend@0.1.0

First npm distribution of `@johnhenry/modernbert` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **ModernBERT / mmBERT encoder on any tensor backend**, loading safetensors; every stage within 1e-6 of MLX fp32 on the tiny checkpoint. [9bfc247](https://github.com/johnhenry/laya-js/commit/9bfc247).
- **`@johnhenry/math-plus-safetensors` moved to devDependencies**: the encoder only uses a structural `SafetensorsFile` type, so it has no runtime dependency on it.
