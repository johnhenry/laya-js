# Changelog

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
