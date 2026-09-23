# Agent playbook

npm workspaces monorepo: 14 packages under `packages/` (13 published as
`@johnhenry/<dir>`, 1 private), 3 private apps under `examples/`, Node ≥ 24
and Bun ≥ 1.2, `node --test` and `bun test`. Packages build to `dist/` with
plain `tsc`, but in development every cross-package import resolves to the
TypeScript **source** through the `source` export condition
(`--conditions=source`, `customConditions` in `tsconfig.base.json`) — so
tests never need a build, and a stale `dist/` never masks a change.
Conventions mirror `~/Projects/@johnhenry/math-plus/AGENTS.md`; read it once.

`CLAUDE.md` in this directory is a symlink to this file.

## MANDATORY: Use td for Task Management

Run `td usage --new-session` at conversation start (or after `/clear`) to
see current work, and `td usage -q` for subsequent reads.

## Workspace structure and build order

Dependency order, leaves first (the root `build`/`test` scripts iterate over
every workspace, so there is no hand-maintained list to keep in sync):

| Package | Role |
| --- | --- |
| [`pyjson`](packages/pyjson) | Python-identical `json.dumps` / rounding (no deps) |
| [`tensor-backend`](packages/tensor-backend) | Backend op contract + conformance suite (no deps) |
| [`hf-cache`](packages/hf-cache) | Hugging Face resolution + cache (no deps) |
| [`langdetect-lite`](packages/langdetect-lite) | Script / language detection → pyjson |
| [`laya-core`](packages/laya-core) | Tensor-free Laya logic → pyjson, `@huggingface/tokenizers` |
| [`backend-cpu`](packages/backend-cpu), [`backend-mlx`](packages/backend-mlx), [`backend-webgpu`](packages/backend-webgpu) | Backends → tensor-backend |
| [`backend-mlx-darwin-arm64`](packages/backend-mlx-darwin-arm64) | Native MLX bundle; **not a workspace** (see gotchas) |
| [`modernbert`](packages/modernbert) | Encoder → tensor-backend |
| [`laya`](packages/laya) | `load`/`predict` → laya-core, modernbert, hf-cache, backend-cpu, math-plus-safetensors; optional peers backend-mlx/-webgpu |
| [`laya-presets`](packages/laya-presets), [`laya-router`](packages/laya-router) | → laya-core (router also → laya, langdetect-lite) |
| [`laya-cli`](packages/laya-cli) | `laya` bin → laya, laya-router |
| [`laya-fixtures`](packages/laya-fixtures) | Private golden data from `../laya-mlx/scripts/dump_js_fixtures.py` (`npm run fixtures`). Never hand-edit. |

The contracts are `packages/tensor-backend/src/index.ts` (Backend interface)
and `packages/laya-core/src/types.ts`. Changing them requires updating every
backend; say so loudly in your report.

## Rules

1. **Canonical implementation.** Never implement the same construct twice.
   Safetensors and fp16 conversion live in math-plus
   (`@johnhenry/math-plus-safetensors`), not here.
2. **Oracle discipline.** Numeric code is tested against the Python/MLX
   fixtures, never hand-typed expected values. Tests whose oracle/fixture or
   hardware (MLX, GPU) is unavailable must *skip*, not fail — but a skip in
   your final verification run is a red flag you must report.
3. **Backends are passed explicitly.** No global default backend.
4. **Disclose limits** in the package README (`## Limitations`).
5. **Only touch the packages your workstream owns.** If you need a change
   in a contract or another package, describe it in your final report.
6. Before WebGPU/MLX-heavy test runs, take `~/gpu.lock`
   (`shlock -f ~/gpu.lock -p $$` or equivalent) so parallel agents don't
   contend for the GPU.

## The verification loop (before every push)

Per package while iterating:
`npm run typecheck -w <pkg> && npm test -w <pkg> && npm run test:bun -w <pkg>`

Whole repo, in CI's order (`.github/workflows/ci.yml`):

```bash
npm ci
node scripts/sync-jsr-configs.mjs --check   # jsr.json files match package.json
npm run typecheck
npm run build        # dist/ + .d.ts for every package; examples bundle with bun
npm test             # test/manifest-drift.test.ts, then every workspace (Node)
npm run test:bun     # every workspace under Bun
```

A genuinely fresh clone before a release:
`git clone . /tmp/laya-js-verifyN && cd $_ && npm ci && npm run build && npm test`.
This catches "works on my checked-out tree" bugs (missing `files`,
undeclared deps, paths outside the repo).

Opt-in suites (slow, need the HF cache and the GPU lock):
`LAYA_REAL=1 npm test -w @johnhenry/laya` (3 checkpoints × MLX/WebGPU ×
f32/f16), `LAYA_REAL_CPU=1`, `LAYA_REAL_CPU_FULL=1`.

## Repo-specific gotchas

- **`@johnhenry/math-plus-safetensors` is not on npm yet.** Packages declare
  `^0.0.0` (what they will get from npm); the root `devDependencies` +
  `overrides` (`"$@johnhenry/math-plus-safetensors"`) map it to the vendored
  tarball in `vendor/`. `overrides` never publish, so consumers get the
  registry package. Refresh from a math-plus checkout with
  `npm run vendor:math-plus` (`MATH_PLUS_DIR=…`, `BUILD=1`), then
  `npm install`. A relative `file:` path directly *in* `overrides` does not
  work — npm resolves it against each dependent workspace.
- **`packages/backend-mlx-darwin-arm64` must stay out of `workspaces`**
  (root `"!packages/backend-mlx-darwin-arm64"`). npm hard-fails the whole
  install with `EBADPLATFORM` on Linux/Intel when an `os`/`cpu`-gated package
  is a workspace member. Instead the root links it as an
  `optionalDependency` (`file:packages/backend-mlx-darwin-arm64`) with a
  matching `overrides` entry, so backend-mlx's `^0.0.0` optional edge
  resolves locally: the lockfile records it as optional + `os`/`cpu`, `npm ci`
  skips it silently on other platforms, and it does not 404 before it is
  published (an unresolvable optional edge makes `npm ci` reject the lockfile
  as out of sync). Changesets does not see it: bump its version by hand
  with backend-mlx (the drift test enforces equality).
- **The native bundle is gitignored and built, not committed.**
  `npm run build:native --prefix packages/backend-mlx-darwin-arm64` (needs
  `pip install mlx==0.32.2` and cmake) writes `lib/` with `SHA256SUMS`;
  `prepack` refuses to pack a bundle that does not verify.
  `npm run build:mlxc` builds the same thing into
  `packages/backend-mlx/prebuilds/darwin-arm64/` for local tests.
- **`$LAYA_MLXC_PATH` set to a missing path is an error**, not a fallback,
  so a CI job cannot silently test a different MLX than it meant to.
- **Install scripts don't run under npm ≥ 11's `allow-scripts`.** koffi's
  and `webgpu`'s scripts are skipped; both work from their prebuilt
  binaries. Don't add a dependency that needs a postinstall.
- **Engines are `node >=24.0.0`, `bun >=1.2.0`** in every manifest, `.nvmrc`
  and CI. The family floor is `>=26.0.0`; this repo stays on 24 because
  that is what every suite was verified on. Raise all of them together (the
  drift test checks they agree).
- **`@webgpu/types`**: backend-webgpu's `.d.ts` mention `GPUDevice`, so the
  types are a runtime `dependency`; TypeScript consumers may still need
  `"types": ["@webgpu/types"]`.
- **The published `exports` keep the `source` condition first**, and `src/`
  ships in every tarball so it resolves. Bun honours it; Node refuses to
  strip types under `node_modules` (`ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING`),
  so a Node consumer must not run with `--conditions=source`. Built `.d.ts`
  files get their `./x.ts` specifiers rewritten to `.js` by
  `scripts/rewrite-dts-extensions.mjs` (tsc leaves them; Deno's `npm:`
  type check fails on them).
- **Bun's `node:test` shim registers only the first file**; package tests
  use `bun:test` under Bun (`test:bun` scripts).

## New-package definition of done

See the README's [`## Adding a new package`](README.md#adding-a-new-package).
Mechanically: `package.json` (source-first `exports`, `files` with `dist` and
`CHANGELOG.md`, root `engines`), the `tsconfig.json` + `tsconfig.typecheck.json`
pair, `README.md` + `CHANGELOG.md` + `LICENSE` (+ `NOTICE` if it ports
laya-mlx code), and an entry in `scripts/sync-jsr-configs.mjs`'s
`PACKAGE_DIRS` or `JSR_EXCLUDED`. `test/manifest-drift.test.ts` fails
loudly if any of that is missing.

## Releases

Nothing is published yet. Mechanism: Changesets (as math-plus), plus a
macOS job for the native platform package. `.github/workflows/release.yml`
is **manual-only** (`workflow_dispatch`, `dry_run` defaults to true) until
the first release is done.

Before the first real publish, in order:

1. Create `github.com/johnhenry/laya-js` and push (badges, `repository`,
   provenance and commit links in the CHANGELOGs all point there).
2. Publish `@johnhenry/math-plus-safetensors` from math-plus
   (`feat/f16-safetensors`). Then delete `vendor/`, the root
   `devDependencies`/`overrides` entries for it, run `npm install`, and
   run the full suite against the registry package. Note `^0.0.0` matches
   only `0.0.0`: if math-plus publishes `0.0.1`+, bump the range in
   `laya` and `modernbert`.
3. Set a scope-capable `NPM_TOKEN` repo secret (granular tokens are
   per-package; a new scoped name 404s on PUT otherwise).
4. On jsr.io: claim/confirm the `@johnhenry` scope, create the ten JSR
   packages, and link this repo as their trusted publisher.
5. Run `release.yml` with `dry_run: true`, read the tarball lists, then
   `changeset version` (the pending `.changeset/initial-release.md` bumps
   all 13 workspace packages to 0.1.0; bump
   `backend-mlx-darwin-arm64` to 0.1.0 by hand, and drop the hand-written
   `## 0.1.0 (Unreleased)` headings Changesets duplicates), commit, and
   run `release.yml` with `dry_run: false`.
6. After the platform package is on npm, run `npm install` once so the
   lockfile records it (until then its optional edge 404s harmlessly).
7. Switch `release.yml` to the family trigger (`release: published` +
   `push: tags: ["v*"]` + `workflow_dispatch`) and add the
   `Full documentation:` line to the READMEs once an
   opensource.johnhenry.me section exists.
