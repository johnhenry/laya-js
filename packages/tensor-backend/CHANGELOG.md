# Changelog

## 0.1.0

### Minor Changes

- 7fecf44: Initial release: the laya-js package family (0.1.0). See each package's CHANGELOG.md for what ships, and the root README for parity results.


First npm distribution of `@johnhenry/tensor-backend` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **The backend op contract.** `Backend<T>` with transfer/lifetime (`fromHost`, async `read`, `dispose`, `scope`), shape, elementwise, reduction and fused transformer ops (`linear`, `layerNorm`, `rope`, `sdpa`, `gelu`), plus optional `geglu`/`meanPool`/`compile`. Introduced in [8b963a2](https://github.com/johnhenry/laya-js/commit/8b963a2); `TestApi.it` signature and `fromHost` widening settled in [c2f272d](https://github.com/johnhenry/laya-js/commit/c2f272d).
- **Conformance suite.** `runConformance` over 49 golden cases (30 ops) generated from Python MLX 0.32.2, run in f32 and f16. [8b963a2](https://github.com/johnhenry/laya-js/commit/8b963a2).
