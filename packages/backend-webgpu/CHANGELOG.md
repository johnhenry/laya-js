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

First npm distribution of `@johnhenry/backend-webgpu` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **WebGPU backend**: tiled f16 GEMM, fused SDPA (flash attention), rope and layerNorm in WGSL; conformance on Node and Bun (Dawn), Deno and Chromium; English checkpoint 63/63 argmax in f32 and f16. [7b327a2](https://github.com/johnhenry/laya-js/commit/7b327a2).
- **Bun upload fix.** Dawn's `writeBuffer` under Bun ignored a TypedArray view's `byteOffset`, so weight views into a shared buffer uploaded the wrong bytes; uploads now copy such views, with a regression test. Fixed in [c15f489](https://github.com/johnhenry/laya-js/commit/c15f489).
- **`@webgpu/types` is now a dependency**, because the published `.d.ts` files mention `GPUDevice` and friends.
