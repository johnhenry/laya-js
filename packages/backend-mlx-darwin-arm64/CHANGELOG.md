# Changelog


First npm distribution of `@johnhenry/backend-mlx-darwin-arm64` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **First npm distribution of the MLX runtime for @johnhenry/backend-mlx**: `libmlxc.dylib` (mlx-c `d4afaec`, "Support MLX v0.32.2") built by `backend-mlx/scripts/build-mlxc.sh` against Apple's MLX 0.32.2 wheel, plus `libmlx.dylib`, `libjaccl.dylib` and `mlx.metallib` from that wheel (64 MB packed, 207 MB unpacked). `os: darwin`, `cpu: arm64`; `prepack` refuses to pack unless every file matches `lib/SHA256SUMS`. Apple's MIT notices ship in `NOTICE` and `lib/licenses/`.
