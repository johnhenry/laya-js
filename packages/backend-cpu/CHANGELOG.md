# Changelog

## 0.1.0 (Unreleased)

First npm distribution of `@johnhenry/backend-cpu` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **Pure-TypeScript f32 reference backend.** Passes the full conformance suite; every stage of the tiny Laya checkpoint within 1e-6 of MLX fp32; 63/63 argmax on all three published checkpoints (opt-in, slow). [9bfc247](https://github.com/johnhenry/laya-js/commit/9bfc247).
