# Changelog

## 0.2.1

### Patch Changes

- Optional backend peer ranges are now `^0.2.0 || ^0.3.0`: they accept the 0.3 backends, and drop pre-0.2 backends, which predate the async-upload contract.
- Updated dependencies
  - @johnhenry/laya@0.2.1

## 0.2.0

### Minor Changes

- 944e3c0: Quantized checkpoints (q8/q4, dequantize-on-load). `load()` / `readWeights()` detect `__metadata__.laya_quant` in `model.safetensors` and dequantize 8-bit (symmetric, groups of 64) and 4-bit (affine, groups of 64, optional q8 tensors) weights to f16/f32 tensor by tensor while loading, so downloads shrink to ~52% (q8) or ~28–35% (q4) with no backend changes. New `quantizeMatrix` / `dequantizeMatrix` / `quantizeSafetensors` / `quantMetadata` exports; `load()` in Node/Bun now also accepts an http(s) base URL (Range reads, as in browsers). New `laya quantize --model <repo|dir> --bits 8|4 --out <dir>` command writes a complete quantized checkpoint directory.

### Patch Changes

- 790e1e0: Depend on `@johnhenry/laya` 0.2 (async device uploads). No API change: both use `load()` / `predict()`, which are unchanged.
- Updated dependencies [790e1e0]
- Updated dependencies [790e1e0]
- Updated dependencies [944e3c0]
  - @johnhenry/laya-router@0.1.3
  - @johnhenry/laya@0.2.0

## 0.1.2

### Patch Changes

- 1adc4c7: Republish with an npm provenance attestation (built and published by GitHub Actions). The earlier versions were published from a local machine without provenance. No code changes.
- Updated dependencies [1adc4c7]
  - @johnhenry/laya-router@0.1.2
  - @johnhenry/laya@0.1.2
  - @johnhenry/pyjson@0.1.2

## 0.1.1

### Patch Changes

- a6d61c7: Licensing by origin. Original packages (tensor-backend, backend-cpu, backend-mlx, backend-webgpu, pyjson, hf-cache) are relicensed from Apache-2.0 to MIT. Packages that port laya-mlx / Laya code stay Apache-2.0 with their NOTICE. laya, laya-router and laya-cli are included to ship alongside `@johnhenry/math-plus-safetensors` on npm.
- Updated dependencies [a6d61c7]
  - @johnhenry/pyjson@0.1.1
  - @johnhenry/laya@0.1.1
  - @johnhenry/laya-router@0.1.1

## 0.1.0

### Minor Changes

- 7fecf44: Initial release: the laya-js package family (0.1.0). See each package's CHANGELOG.md for what ships, and the root README for parity results.

### Patch Changes

- Updated dependencies [7fecf44]
  - @johnhenry/laya@0.1.0
  - @johnhenry/laya-router@0.1.0
  - @johnhenry/pyjson@0.1.0

First npm distribution of `@johnhenry/laya-cli` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **`laya predict` and `laya bench`**, mirroring the laya-mlx CLI and benchmark worker; with `--backend mlx` the README example prints byte-identical JSON. [8bfcd6c](https://github.com/johnhenry/laya-js/commit/8bfcd6c).
- **The `laya` bin runs from the published tarball** (`dist/bin.js`), and re-executes under `--conditions=source` in an unbuilt checkout.
