# laya-js

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Flaya.svg)](https://www.npmjs.com/package/@johnhenry/laya)
[![CI](https://github.com/johnhenry/laya-js/actions/workflows/ci.yml/badge.svg)](https://github.com/johnhenry/laya-js/actions/workflows/ci.yml)
[![license](https://img.shields.io/npm/l/%40johnhenry%2Flaya.svg)](LICENSE)

Laya typed decisions (choice / score / noul) in JavaScript. It loads the
published Laya MLX checkpoints (fp16 safetensors from Hugging Face) and runs
them on **native MLX** (Node and Bun on Apple Silicon, through our own
mlx-c FFI binding), on **WebGPU** (browsers, Deno, and Node/Bun via Dawn), or
on a pure-TypeScript **CPU** reference. On MLX the answers are bit-identical
to Python [laya-mlx](https://github.com/mizorewww/laya-mlx).

It is a family of packages, not a single port: the tensor-backend contract,
the three backends, the ModernBERT encoder, the Hugging Face cache, the
language detector and the Python-compatible JSON are each usable without
Laya.

**Status:** 0.1.0 is prepared but not yet published. Nothing below is on npm
or JSR yet.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Which package do I want?](#which-package-do-i-want)
- [Packages](#packages)
- [Backends](#backends)
- [Results](#results)
- [Limitations](#limitations)
- [Adding a new package](#adding-a-new-package)
- [Credits](#credits)
- [Family](#family)

## Install

**Node ≥ 24** (TypeScript types are stripped natively; no build step needed
to run the sources in this repo):

```bash
npm install @johnhenry/laya @johnhenry/backend-mlx      # Apple Silicon: native MLX
npm install @johnhenry/laya @johnhenry/backend-webgpu   # anywhere with a GPU: WebGPU via Dawn
npm install @johnhenry/laya                             # CPU reference only
```

On darwin/arm64, `@johnhenry/backend-mlx` pulls in the prebuilt MLX runtime
(`@johnhenry/backend-mlx-darwin-arm64`, a 64 MB optional dependency) and
`koffi` for Node's FFI (a prebuilt addon). Nothing is compiled on your machine.

**Bun ≥ 1.2**: the same packages with `bun add`. Bun uses its built-in
`bun:ffi` for MLX.

**Browsers**: `npm install @johnhenry/laya @johnhenry/backend-webgpu` and
bundle as usual. The `browser` export condition swaps in the browser I/O
(Cache API model cache, `fetch`), so bundlers never see `node:fs`, koffi or
libmlxc. WebGPU needs Chrome/Edge ≥ 113 (f16 from 120); Safari 26 and
Firefox 141 are expected to work but are not verified.

**Deno**: the backend-agnostic packages (`tensor-backend`, `backend-cpu`,
`backend-webgpu`, `modernbert`, `laya-core`, `pyjson`, `langdetect-lite`,
`hf-cache`, …) are prepared for JSR (`deno add jsr:@johnhenry/<name>`);
`@johnhenry/laya` itself and MLX are Node/Bun/browser only for now.

## Quick start

The laya-mlx README example, in JavaScript:

```ts
import { load } from "@johnhenry/laya";

const agent = await load("aac6fef/laya-mlx"); // MLX → WebGPU → CPU, whichever is available
const result = await agent.predict("I was billed twice. Please refund the duplicate.", {
  department: {
    type: "choice",
    instructions: "Who should handle this?",
    criteria: ["billing", "technical", "sales"],
  },
});
console.log(result.answers.department); // { type: "choice", choice: "billing", confidence, probabilities, … }
agent.dispose();
```

The first `load` downloads the checkpoint into the standard
`~/.cache/huggingface/hub` layout (shared with Python's `hf download`);
later runs are fully local. From a shell:

```bash
npx @johnhenry/laya-cli predict --state "I was billed twice. Please refund the duplicate." \
  --questions '{"department": {"type": "choice", "instructions": "Who should handle this?", "criteria": ["billing", "technical", "sales"]}}'
```

## Which package do I want?

| I want to... | Start with |
|---|---|
| Ask Laya typed questions from JS/TS | [`laya`](./packages/laya) — `load()` + `predict()`; add a GPU backend next to it |
| Ask from a shell or benchmark a machine | [`laya-cli`](./packages/laya-cli) (`npx @johnhenry/laya-cli predict`) |
| Pick English vs multilingual automatically | [`laya-router`](./packages/laya-router) |
| Use ready-made triage / email / moderation questions | [`laya-presets`](./packages/laya-presets) |
| Run on Apple Silicon at native speed | [`backend-mlx`](./packages/backend-mlx) — macOS arm64 only; the native bundle installs itself |
| Run on a GPU in the browser (or Node/Bun/Deno) | [`backend-webgpu`](./packages/backend-webgpu) |
| Run anywhere, slowly but exactly | [`backend-cpu`](./packages/backend-cpu) — the f32 reference the GPU backends are tested against |
| Write a new backend | [`tensor-backend`](./packages/tensor-backend) — the contract plus a conformance suite to run against it |
| Run a ModernBERT / mmBERT encoder | [`modernbert`](./packages/modernbert) — any backend, safetensors weights |
| Download/cache Hugging Face files like `huggingface_hub` | [`hf-cache`](./packages/hf-cache) — disk cache on Node/Bun, Cache API in browsers |
| Build prompts/calibration without tensors | [`laya-core`](./packages/laya-core) |
| Detect script / "is this English?" without a model | [`langdetect-lite`](./packages/langdetect-lite) |
| Produce exactly Python's `json.dumps` / `round()` output | [`pyjson`](./packages/pyjson) |
| Read safetensors | `@johnhenry/math-plus-safetensors` — lives in [math-plus](https://github.com/johnhenry/math-plus) |

## Packages

### Laya

| Package | Role |
|---|---|
| [`@johnhenry/laya`](./packages/laya) | `load()` / `predict()` / `createAgent()` / shortlist — port of laya-mlx `Agent`, `load`, `shortlist.py` |
| [`@johnhenry/laya-core`](./packages/laya-core) | Tensor-free logic: validation, prompts, tokenization, collation, calibration, result formatting |
| [`@johnhenry/laya-router`](./packages/laya-router) | Language/task routing across the three checkpoints (`router.py`) |
| [`@johnhenry/laya-presets`](./packages/laya-presets) | Question sets and email cleaning (`presets.py`, `email.py`) |
| [`@johnhenry/laya-cli`](./packages/laya-cli) | `laya predict` / `laya bench` |

### Tensor backends

| Package | Role |
|---|---|
| [`@johnhenry/tensor-backend`](./packages/tensor-backend) | Op contract + conformance suite (golden cases from Python MLX) |
| [`@johnhenry/backend-cpu`](./packages/backend-cpu) | Pure-TS f32 reference |
| [`@johnhenry/backend-mlx`](./packages/backend-mlx) | Native MLX through mlx-c (`bun:ffi` / `koffi`) |
| [`@johnhenry/backend-mlx-darwin-arm64`](./packages/backend-mlx-darwin-arm64) | Prebuilt MLX 0.32.2 runtime (optional dependency of backend-mlx) |
| [`@johnhenry/backend-webgpu`](./packages/backend-webgpu) | WGSL kernels, f16, flash attention |

### Building blocks

| Package | Role |
|---|---|
| [`@johnhenry/modernbert`](./packages/modernbert) | ModernBERT / mmBERT encoder on any backend |
| [`@johnhenry/hf-cache`](./packages/hf-cache) | Hugging Face Hub resolution + `huggingface_hub`-compatible cache |
| [`@johnhenry/langdetect-lite`](./packages/langdetect-lite) | Script + Latin-language detection (`lang.py`) |
| [`@johnhenry/pyjson`](./packages/pyjson) | Byte-identical CPython `json.dumps`, `repr(float)`, `round()` |

Private: [`laya-fixtures`](./packages/laya-fixtures) (golden data from
Python) and three [examples](./examples): a terminal Snake driven by Laya
(Node/Bun), the same game in the browser on WebGPU, and a web playground.

## Backends

| Runtime | MLX | WebGPU | CPU |
|---|---|---|---|
| Node ≥ 24, macOS arm64 | f32, f16 (bf16 storage) | f32, f16 (Dawn → Metal) | f32 |
| Node ≥ 24, Linux / Windows | — | f32, f16 where Dawn finds an adapter | f32 |
| Bun ≥ 1.2 | as Node (`bun:ffi`) | as Node (same Dawn addon) | f32 |
| Deno 2 | — (no `Deno.dlopen` adapter yet) | f32, f16 (built-in wgpu); backend tested, `laya` not | f32 |
| Chrome / Edge ≥ 113 | — | f32; f16 from 120 (`shader-f16`) | f32 |
| Safari 26, Firefox ≥ 141 | — | expected, not verified | f32 |

- `load(…, { backend: "auto" })` picks MLX (macOS arm64 with a libmlxc) →
  WebGPU (adapter available) → CPU; in browsers WebGPU → CPU.
- `dtype: "f16"` is the default. The CPU backend, and WebGPU without
  `shader-f16`, compute in f32; `agent.dtype` reports what is used.
- MLX runs the same `libmlx` and Metal kernels as Python, so results match
  exactly. WebGPU has its own kernels; its f16 results differ from the third
  decimal on, but choices agree.

## Results

**Parity with Python laya-mlx**: the 63-question laya-mlx validation set
(16 cases) on all three published checkpoints, compared with Python's
`result_fp32` (tolerance 1e-4) and `result_fp16` (0.02). "argmax" is the
chosen label / most likely score level / noul side; "exact" means the
whole answer object deep-equals Python's, including 4-decimal rounding.

| checkpoint | MLX f32 | MLX f16 | WebGPU f32 | WebGPU f16 |
|---|---|---|---|---|
| English (`aac6fef/laya-mlx`, ModernBERT-large 421M) | 63/63 argmax, 63/63 exact | 63/63, 63/63 | 63/63, 62/63 (max \|Δ\| 1.0e-4) | 63/63, 14/63 (3.2e-3) |
| Multilingual (`aac6fef/laya-multilingual-mlx`, mmBERT-base 322M) | 63/63, 63/63 | 63/63, 63/63 | 63/63, 63/63 | 63/63, 27/63 (1.1e-3) |
| Typed decisions (`aac6fef/laya-typed-decisions-mlx`) | 63/63, 63/63 | 63/63, 63/63 | 63/63, 63/63 | 63/63, 8/63 (3.8e-3) |

All 12 configurations agree with Python on every one of the 63 answers
(756/756). The CPU reference also gives 63/63 argmax on all three
checkpoints (max |Δlogit| ≤ 4e-5, single-threaded and slow). Reproduce with
`LAYA_REAL=1 npm test -w @johnhenry/laya` (details in the
[laya README](./packages/laya/README.md#parity)).

**Speed — Apple M2, preliminary.** Measured on a shared development
machine (±30% noise); a separate benchmark document will supersede these.

| One short English question (33 tokens, B = 1) | MLX | WebGPU |
|---|---:|---:|
| Forward pass, f16, median | 23.0 ms | 25.9 ms |
| Forward pass, f32, median | 34.6 ms | 43.1 ms |
| End-to-end `predict` (`laya bench`), f16 | ≈ 38 ms | ≈ 250 ms (end-to-end WebGPU path being optimized) |

MLX op dispatch costs 0.80 µs (Node/koffi) and 0.55 µs (Bun/`bun:ffi`) per
op, against 0.40 µs in Python; GPU time is identical because it is the same
`libmlx`. Per-backend numbers: [backend-mlx](./packages/backend-mlx/README.md#performance),
[backend-webgpu](./packages/backend-webgpu/README.md).

## Limitations

- **MLX is macOS/arm64 only** and pins MLX 0.32.2 (the prebuilt bundle).
  Deno has no MLX adapter yet.
- **WebGPU f16 is not bit-exact** with MLX (its own kernels); f32 is within
  1e-4. Kernels are tuned on Apple M2 and untested on discrete or mobile GPUs.
  Every op is its own dispatch (no graph fusion; `compile` is MLX-only).
- **The browser loader** is tested in Node against a local HTTP server and
  bundle-checked with Bun; the full model has not been run in a real browser
  in CI.
- **CPU reference** is exact but slow (tens of seconds per case on the large
  checkpoints). It is an oracle and a fallback, not a production path.
- `embed` returns the zero vector for texts that tokenize to nothing
  (Python divides by zero there).
- Node consumers must not run with `--conditions=source`: the packages
  export TypeScript sources under that condition (for this monorepo's
  development) and Node does not strip types inside `node_modules`. Bun
  handles it.
- No `convert` command: converting upstream PyTorch checkpoints to MLX
  format stays in Python laya-mlx.
- Each package README has its own `## Limitations` section with the details.

## Adding a new package

1. `packages/<name>/package.json` — copy a sibling's (e.g. `pyjson`): name
   `@johnhenry/<name>`, version matching the family, `exports` with the
   `source` condition first, `files` with `dist` and `CHANGELOG.md`,
   `engines` equal to the root's.
2. `tsconfig.json` + `tsconfig.typecheck.json` — copy a sibling's pair.
3. `README.md` (badge, install, example, API, `## Limitations`,
   `## Family`), `CHANGELOG.md`, and `LICENSE` (a copy of the root's; also
   `NOTICE` when it ports laya-mlx code).
4. Add the directory to `scripts/sync-jsr-configs.mjs`'s `PACKAGE_DIRS`, or
   to `JSR_EXCLUDED` with the reason, then `npm run sync:jsr`.
5. `npm test` — `test/manifest-drift.test.ts` fails loudly on anything
   missed above. Root `build`/`test` iterate over all workspaces, so there is
   no script list to update.

## Credits

- **[Laya](https://github.com/NandhaKishorM/laya)** by Convai Innovations —
  the model, weights, prompt format and calibration (Apache-2.0).
- **[laya-mlx](https://github.com/mizorewww/laya-mlx)** — the Apple MLX port
  this project follows line by line, the published MLX checkpoints on
  Hugging Face, and the Python oracle every fixture comes from (Apache-2.0).
- **[MLX](https://github.com/ml-explore/mlx)** and
  **[mlx-c](https://github.com/ml-explore/mlx-c)** by Apple (MIT), shipped
  unmodified in `@johnhenry/backend-mlx-darwin-arm64`.
- [`@nielspeter/mlx-ts`](https://github.com/nielspeter/mlx-ts), which showed
  the mlx-c-over-FFI route works; [koffi](https://koffi.dev) for Node FFI;
  [Dawn / `webgpu`](https://github.com/dawn-gpu/node-webgpu) for WebGPU in
  Node; [`@huggingface/tokenizers`](https://www.npmjs.com/package/@huggingface/tokenizers).

This is an independent port, not an official Convai Innovations or Apple
release. Model weights are not distributed here; they download from Hugging
Face under their own licenses. See [NOTICE](NOTICE).

## Family

laya-js is a consumer of the **math** family, not a member of it: it reuses
math-plus's safetensors reader and dtype conventions rather than growing its
own tensor library.

- **[`@johnhenry/math-plus`](https://github.com/johnhenry/math-plus)** — the
  JS/TS numeric runtime. `@johnhenry/laya` reads every checkpoint through
  `openSafetensors` from `@johnhenry/math-plus-safetensors` (lazy header +
  coalesced reads, F16 as `Float16Array`), a real dependency — the only
  safetensors implementation in either repo. `@johnhenry/tensor-backend`'s
  dtype names and `HostTensor` layout match `@johnhenry/math-plus-tensor-core`,
  so `toMathPlusArgs` hands results to `Tensor.fromTypedArray` without a copy.
  math-plus's `tensor-webgpu` is a general tensor library; this repo's
  `backend-webgpu` implements only the inference op contract and is not a
  dependency either way.
- **[`@johnhenry/math`](https://github.com/johnhenry/math)** — math-plus's
  scalar/CAS sibling. No direct dependency from laya-js; listed so the
  family map stays complete.

## License

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).
`@johnhenry/backend-mlx-darwin-arm64` is MIT: it contains Apple's MLX binaries.
