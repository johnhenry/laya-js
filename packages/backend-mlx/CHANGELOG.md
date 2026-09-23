# Changelog

## 0.1.0 (Unreleased)

First npm distribution of `@johnhenry/backend-mlx` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **Native MLX backend over our own mlx-c FFI binding** (`bun:ffi` on Bun, `koffi` on Node): lazy graph, fused MLX kernels, `compile` via `mlx_compile`, two mlx-c ABIs detected at load time. Conformance green in f32/f16 on Node and Bun. [8acf587](https://github.com/johnhenry/laya-js/commit/8acf587); decision record in `docs/mlx-binding-decision.md`.
- **Native distribution.** `koffi` and the new platform package `@johnhenry/backend-mlx-darwin-arm64` are optional dependencies; `libmlxc` resolution order is `libPath` → `$LAYA_MLXC_PATH` (file or directory; `$LAYA_MLXC_LIB` kept as an alias; a set-but-missing path now throws) → platform package → local build → `@nielspeter/mlx-ts-darwin-arm64` → Homebrew. `scripts/build-mlxc.sh` takes `MLXC_OUT` and writes `SHA256SUMS` and Apple's license texts next to the binaries.
