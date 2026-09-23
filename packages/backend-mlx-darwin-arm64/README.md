# @johnhenry/backend-mlx-darwin-arm64

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Fbackend-mlx-darwin-arm64.svg)](https://www.npmjs.com/package/@johnhenry/backend-mlx-darwin-arm64)

The prebuilt MLX runtime that
[`@johnhenry/backend-mlx`](../backend-mlx) loads on macOS/Apple Silicon.
You normally never install this directly: it is an `optionalDependency` of
`@johnhenry/backend-mlx`, and npm installs it only where `os` is `darwin`
and `cpu` is `arm64`.

## Install

```bash
npm install @johnhenry/backend-mlx   # pulls this package in on darwin/arm64
bun add @johnhenry/backend-mlx
```

Contents (`lib/`, 64 MB packed, 207 MB unpacked):

| File | From |
|---|---|
| `libmlxc.dylib` | mlx-c `d4afaec` ("Support MLX v0.32.2"), compiled against the wheel below by `backend-mlx/scripts/build-mlxc.sh` |
| `libmlx.dylib`, `libjaccl.dylib`, `mlx.metallib` | Apple's `mlx` / `mlx-metal` 0.32.2 Python wheels — the same MLX that Python laya-mlx runs |
| `VERSION`, `SHA256SUMS` | exact sources and file hashes |
| `licenses/MLX.LICENSE`, `licenses/mlx-c.LICENSE` | Apple's MIT license texts |

The dylibs are relocatable (`@rpath` / `@loader_path`) and ad-hoc signed.

```js
import { libDir, libmlxc } from "@johnhenry/backend-mlx-darwin-arm64";
// backend-mlx finds these on its own; the exports are for tools and for
// pointing $LAYA_MLXC_PATH at the bundle explicitly.
```

## Why the whole bundle ships on npm

`mlx.metallib` (Apple's compiled Metal kernels) is 182 MB of the 207 MB. We
ship it in the tarball instead of downloading it on first use because:

- npm accepts it: the same four files at the same size are already on the
  registry as `@nielspeter/mlx-ts-darwin-arm64` (199 MB unpacked).
- Installs stay offline-capable, cached, and integrity-checked by npm itself
  (the lockfile's `sha512`), with no extra download code, cache directory or
  network access at runtime.
- The `os`/`cpu` gate means Linux, Windows and Intel Macs never download it.

Fallbacks if the registry ever rejects the size: attach the bundle to a
GitHub Release and point `$LAYA_MLXC_PATH` at it, or build locally with
`npm run build:mlxc -w @johnhenry/backend-mlx`.

## Building and verifying (maintainers)

```bash
pip install mlx==0.32.2                                   # or MLX_PY_DIR=…/site-packages/mlx
npm run build:native --prefix packages/backend-mlx-darwin-arm64
npm run verify --prefix packages/backend-mlx-darwin-arm64  # sha256 of every file
npm pack ./packages/backend-mlx-darwin-arm64              # prepack re-verifies
```

This directory is deliberately **outside** npm workspaces (root
`"!packages/backend-mlx-darwin-arm64"`): npm hard-fails with `EBADPLATFORM`
on other platforms when an `os`/`cpu`-gated package is a workspace member.
The repo root links it as an optional `file:` dependency plus an
`overrides` entry instead, which npm skips silently elsewhere. Changesets
therefore does not version it; bump it together with
`@johnhenry/backend-mlx` (`test/manifest-drift.test.ts` enforces equal
versions and the `^<version>` optional range).

## Limitations

- darwin/arm64 only; there is no MLX for other platforms.
- Pins MLX 0.32.2 and one mlx-c commit. For another MLX, build locally or
  set `$LAYA_MLXC_PATH`.
- The bundle is built on the CI's macOS image; the minimum macOS version is
  whatever Apple's 0.32.2 wheel supports (macOS 14+).

## Family

Part of **[laya-js](https://github.com/johnhenry/laya-js#readme)** — see its [package map](https://github.com/johnhenry/laya-js#which-package-do-i-want).

- [`@johnhenry/backend-mlx`](https://github.com/johnhenry/laya-js/tree/main/packages/backend-mlx) resolves `@johnhenry/backend-mlx-darwin-arm64/package.json` and loads `lib/libmlxc.dylib` (step 3 of its resolution order, after `libPath` and `$LAYA_MLXC_PATH`).

## License

MIT. The binaries are Apple's MLX and mlx-c (MIT, © 2023-2024 Apple Inc.);
see [NOTICE](NOTICE) and `lib/licenses/`.
