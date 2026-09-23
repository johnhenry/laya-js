# Changelog

Per-package histories live in each `packages/*/CHANGELOG.md`; this file
groups them by release.

## 0.1.0 (Unreleased)

The first release of the laya-js family. Every package is a first npm
distribution (nothing was published before under any name). Parity: MLX and
WebGPU both give 63/63 argmax agreement with Python laya-mlx on all three
published checkpoints, in f32 and f16.

**`@johnhenry/tensor-backend`**

- **The backend op contract.** `Backend<T>` with transfer/lifetime (`fromHost`, async `read`, `dispose`, `scope`), shape, elementwise, reduction and fused transformer ops (`linear`, `layerNorm`, `rope`, `sdpa`, `gelu`), plus optional `geglu`/`meanPool`/`compile`. Introduced in [8b963a2](https://github.com/johnhenry/laya-js/commit/8b963a2); `TestApi.it` signature and `fromHost` widening settled in [c2f272d](https://github.com/johnhenry/laya-js/commit/c2f272d).
- **Conformance suite.** `runConformance` over 49 golden cases (30 ops) generated from Python MLX 0.32.2, run in f32 and f16. [8b963a2](https://github.com/johnhenry/laya-js/commit/8b963a2).

**`@johnhenry/backend-cpu`**

- **Pure-TypeScript f32 reference backend.** Passes the full conformance suite; every stage of the tiny Laya checkpoint within 1e-6 of MLX fp32; 63/63 argmax on all three published checkpoints (opt-in, slow). [9bfc247](https://github.com/johnhenry/laya-js/commit/9bfc247).

**`@johnhenry/backend-mlx`**

- **Native MLX backend over our own mlx-c FFI binding** (`bun:ffi` on Bun, `koffi` on Node): lazy graph, fused MLX kernels, `compile` via `mlx_compile`, two mlx-c ABIs detected at load time. Conformance green in f32/f16 on Node and Bun. [8acf587](https://github.com/johnhenry/laya-js/commit/8acf587); decision record in `docs/mlx-binding-decision.md`.
- **Native distribution.** `koffi` and the new platform package `@johnhenry/backend-mlx-darwin-arm64` are optional dependencies; `libmlxc` resolution order is `libPath` → `$LAYA_MLXC_PATH` (file or directory; `$LAYA_MLXC_LIB` kept as an alias; a set-but-missing path now throws) → platform package → local build → `@nielspeter/mlx-ts-darwin-arm64` → Homebrew. `scripts/build-mlxc.sh` takes `MLXC_OUT` and writes `SHA256SUMS` and Apple's license texts next to the binaries.

**`@johnhenry/backend-mlx-darwin-arm64`**

- **First npm distribution of the MLX runtime for @johnhenry/backend-mlx**: `libmlxc.dylib` (mlx-c `d4afaec`, "Support MLX v0.32.2") built by `backend-mlx/scripts/build-mlxc.sh` against Apple's MLX 0.32.2 wheel, plus `libmlx.dylib`, `libjaccl.dylib` and `mlx.metallib` from that wheel (64 MB packed, 207 MB unpacked). `os: darwin`, `cpu: arm64`; `prepack` refuses to pack unless every file matches `lib/SHA256SUMS`. Apple's MIT notices ship in `NOTICE` and `lib/licenses/`.

**`@johnhenry/backend-webgpu`**

- **WebGPU backend**: tiled f16 GEMM, fused SDPA (flash attention), rope and layerNorm in WGSL; conformance on Node and Bun (Dawn), Deno and Chromium; English checkpoint 63/63 argmax in f32 and f16. [7b327a2](https://github.com/johnhenry/laya-js/commit/7b327a2).
- **Bun upload fix.** Dawn's `writeBuffer` under Bun ignored a TypedArray view's `byteOffset`, so weight views into a shared buffer uploaded the wrong bytes; uploads now copy such views, with a regression test. Fixed in [c15f489](https://github.com/johnhenry/laya-js/commit/c15f489).
- **`@webgpu/types` is now a dependency**, because the published `.d.ts` files mention `GPUDevice` and friends.

**`@johnhenry/hf-cache`**

- **Hugging Face Hub file resolution with the `huggingface_hub` cache layout** on Node/Bun/Deno (files Python downloaded are reused offline and vice versa) and the Cache API in browsers. [af2d5dc](https://github.com/johnhenry/laya-js/commit/af2d5dc).
- **No runtime dependencies**: the unused `@huggingface/hub` dependency was removed.

**`@johnhenry/pyjson`**

- **Byte-identical CPython `json.dumps`, `repr(float)`, `"%g"` and `round()`** (round-half-even on the exact double). [855f699](https://github.com/johnhenry/laya-js/commit/855f699).

**`@johnhenry/langdetect-lite`**

- **Port of laya-mlx `lang.py`**: script detection and Latin-language guessing with no model or data files; identical decisions on the laya-mlx fixtures. [855f699](https://github.com/johnhenry/laya-js/commit/855f699).

**`@johnhenry/laya-core`**

- **Tensor-free Laya logic ported from laya-mlx** (`common.py`, `agent.py`, `prepared.py`, `tokenizer.py`): validation, prompts, tokenization, collation, calibration and result formatting; exact item and result parity on all three checkpoints. [855f699](https://github.com/johnhenry/laya-js/commit/855f699).

**`@johnhenry/modernbert`**

- **ModernBERT / mmBERT encoder on any tensor backend**, loading safetensors; every stage within 1e-6 of MLX fp32 on the tiny checkpoint. [9bfc247](https://github.com/johnhenry/laya-js/commit/9bfc247).
- **`@johnhenry/math-plus-safetensors` moved to devDependencies**: the encoder only uses a structural `SafetensorsFile` type, so it has no runtime dependency on it.

**`@johnhenry/laya`**

- **`load()` / `predict()` / `createAgent()` / shortlist**, a port of laya-mlx `Agent`/`load`: MLX results bit-identical to Python; 63/63 argmax on all three checkpoints on MLX and WebGPU in f32 and f16 (12 configurations). [8bfcd6c](https://github.com/johnhenry/laya-js/commit/8bfcd6c).
- **`@johnhenry/math-plus-safetensors` is a semver dependency (`^0.0.0`)** instead of a `file:` link into a local math-plus worktree.

**`@johnhenry/laya-router`**

- **Port of laya-mlx `router.py`**: the same routing decisions, reasons, precedence and LRU residency, made async-safe. [8bfcd6c](https://github.com/johnhenry/laya-js/commit/8bfcd6c).

**`@johnhenry/laya-presets`**

- **Port of laya-mlx `presets.py` and `email.py`** with the exact question text. [8bfcd6c](https://github.com/johnhenry/laya-js/commit/8bfcd6c).

**`@johnhenry/laya-cli`**

- **`laya predict` and `laya bench`**, mirroring the laya-mlx CLI and benchmark worker; with `--backend mlx` the README example prints byte-identical JSON. [8bfcd6c](https://github.com/johnhenry/laya-js/commit/8bfcd6c).
- **The `laya` bin runs from the published tarball** (`dist/bin.js`), and re-executes under `--conditions=source` in an unbuilt checkout.

**`@johnhenry/laya-fixtures`** (internal, unpublished)

- **Golden fixtures from Python laya-mlx** (`scripts/dump_js_fixtures.py`). Private; never published. [8b963a2](https://github.com/johnhenry/laya-js/commit/8b963a2).

### Housekeeping

- Engines: `node >=24.0.0`, `bun >=1.2.0` in every manifest, `.nvmrc` and the CI matrix
  (enforced by `test/manifest-drift.test.ts`).
- `@johnhenry/math-plus-safetensors` resolves to a vendored tarball
  (`vendor/`, root `overrides`) until math-plus publishes it.
- JSR configs generated by `scripts/sync-jsr-configs.mjs`; Changesets config;
  CI (`ci.yml`) and a manual-only release skeleton (`release.yml`).
