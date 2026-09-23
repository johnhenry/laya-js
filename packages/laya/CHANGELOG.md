# Changelog

## 0.1.0 (Unreleased)

First npm distribution of `@johnhenry/laya` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **`load()` / `predict()` / `createAgent()` / shortlist**, a port of laya-mlx `Agent`/`load`: MLX results bit-identical to Python; 63/63 argmax on all three checkpoints on MLX and WebGPU in f32 and f16 (12 configurations). [8bfcd6c](https://github.com/johnhenry/laya-js/commit/8bfcd6c).
- **`@johnhenry/math-plus-safetensors` is a semver dependency (`^0.0.0`)** instead of a `file:` link into a local math-plus worktree.
