# Changelog

## 0.1.0 (Unreleased)

First npm distribution of `@johnhenry/backend-webgpu` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **WebGPU backend**: tiled f16 GEMM, fused SDPA (flash attention), rope and layerNorm in WGSL; conformance on Node and Bun (Dawn), Deno and Chromium; English checkpoint 63/63 argmax in f32 and f16. [7b327a2](https://github.com/johnhenry/laya-js/commit/7b327a2).
- **Bun upload fix.** Dawn's `writeBuffer` under Bun ignored a TypedArray view's `byteOffset`, so weight views into a shared buffer uploaded the wrong bytes; uploads now copy such views, with a regression test. Fixed in [c15f489](https://github.com/johnhenry/laya-js/commit/c15f489).
- **`@webgpu/types` is now a dependency**, because the published `.d.ts` files mention `GPUDevice` and friends.
