# Laya Playground (WebGPU, browser)

**Live:** <https://johnhenry.github.io/laya-js/playground/> (GitHub Pages, deployed from `main` by `.github/workflows/pages.yml`).

A single-page app that runs Laya typed decisions in the browser with
[`@johnhenry/laya`](../../packages/laya) on the WebGPU backend (f16 when the
adapter has `shader-f16`).

![Playground, dark theme](docs/screenshot-dark.png)

- **Checkpoint picker**: Multilingual (mmBERT-base, 644 MB, the smallest),
  English and Typed decisions (ModernBERT-large, 843 MB each).
- **Download progress**: weights come from
  `https://huggingface.co/<repo>/resolve/<commit>/…` through
  `@johnhenry/hf-cache`'s browser store (Cache API, cache name `hf-cache`).
  A reload loads from the cache (about 1 s instead of about 35 s), and the
  card shows "cached in this browser". "Clear cached weights" deletes them (click twice to confirm).
- **WebGPU detection**: explains a missing `navigator.gpu`, a missing adapter,
  or a non-secure context. Without `shader-f16`, the pill says "f32 only" and
  the f32 checkbox is ticked.
- **State editor**: a typed field builder (text/number/boolean rows, add/
  remove/rename) that collapses to a plain string when it's just the
  default single text field -- so free text still works with zero setup.
  "Edit as JSON" is always available as an escape hatch (objects go through
  Python-identical `json.dumps`, as in laya-mlx).
- **Question builder**: add choice, score and yes/no (`noul`) questions with
  criteria. Choice options can have descriptions, score levels are ordered,
  and yes/no questions have optional true/false meanings. You can also
  "Edit as JSON", which is sent verbatim so the agent's Python-compatible
  validation messages appear.
- **Compare mode**: run the same state/questions on the WebGPU backend plus
  any of: `@johnhenry/laya`'s pure-TS CPU reference backend (in this tab);
  native **MLX**; **ONNX** via
  [`@receptron/laya`](https://github.com/receptron/laya), an independent
  export of the same upstream model; or **Jev**, TypeSafe's hosted
  typed-decision API -- side by side, with per-column latency, answers and
  raw response, and the fastest column marked. The CPU agent (a second
  checkpoint download) loads lazily, only while "CPU" is checked, and is
  disposed when unchecked. MLX/ONNX/Jev all run through
  [`laya-server`](../laya-server) -- a separate local process, since none of
  the three can run in a browser tab (native FFI, `onnxruntime-node`, and a
  `TYPESAFE_API_KEY` that must never reach client JS, respectively) -- with
  a "Backend server" URL field and a live `/health`-driven pill per backend
  that disables its checkbox until the server reports it available. Jev's
  answers have no equivalent to Laya's act probability; that stat reads
  "N/A" for Jev rather than a misleading `0.0000`.
- **Batch mode**: run one `predict()` call per line of text sequentially,
  each feeding the queue below. Enabled only when the state is exactly one
  text field, since "one value per line" doesn't generalize to more fields.
- **Result queue**: every run (single, compare, or batch) is added to an
  in-memory queue, newest first, for this page load only. Pick any score/
  yes-no question as the sort priority; click an entry to view its stored
  results again. **Export JSON** downloads the full queue; **Export CSV**
  derives columns from the union of state and question keys actually seen
  across queued runs, not a fixed schema.
- **Presets**: the README triage example (the `en` case of the parity
  fixtures), a German variant and a plain-text example, plus every
  `*Questions()` export of `@johnhenry/laya-presets` when it has any.
- **Results**: a card per question with the verdict, a probability bar per
  option (score shows the expected value on a scale), confidence and act
  probability. Also shown: per-run latency, input tokens, question count and
  the engine. The raw request and response are both collapsible.
  Cmd/Ctrl+Enter runs.
- Accessible (labels, `role=meter` bars, `aria-live` results, focus rings,
  reduced motion) and responsive (one column below 900 px), with light, dark
  and system themes.

## Run

```bash
npm run dev -w @johnhenry/example-web-playground    # bun build → dist/, then serve http://localhost:5173/
# or step by step
npm run build -w @johnhenry/example-web-playground  # bun scripts/build.ts
npm start -w @johnhenry/example-web-playground      # node scripts/serve.ts dist --port 5173
```

Open <http://localhost:5173/> in Chrome or Edge 113+ (f16 needs 120+), or
Safari 26. `localhost` is a secure context, so WebGPU and the Cache API
work. Serving it from another host needs https.

The build uses Bun's bundler (`scripts/build.ts`, conditions `browser` +
`source`, so workspace packages are bundled from `src/`). It then **fails if
the bundle references a Node-only or native module**, such as
`node:*`/`fs`/`child_process` imports, `koffi`, `bun:ffi`,
`@johnhenry/backend-mlx` or the Dawn addon.

`scripts/serve.ts` is a dependency-free static server (Node or Bun). The
snake-web example reuses both scripts.

### Tests

```bash
npm test -w @johnhenry/example-web-playground   # typecheck, then node --test test/*.test.ts
```

The question-draft, state-field and queue/export conversions live in
`src/lib/{questions,queue,state-fields}.ts` with no DOM dependency, so
they're covered directly by `node:test` (round-trips, validation errors,
priority-value normalization, CSV column derivation) -- 40 cases, 0
skipped. `main.ts` itself (DOM wiring, agent loading, WebGPU) has no
automated coverage; see "Verified" below for how it's checked instead.

### GitHub Pages

`node scripts/build-pages.mjs` (from the repo root) builds this app and
snake-web and assembles `_site/` (`/` landing page, `/playground/`,
`/snake/`). `.github/workflows/pages.yml` runs it on every push to `main`
and deploys to <https://johnhenry.github.io/laya-js/>. Every asset URL is
relative (`./main.js`, relative chunk imports), so the `/laya-js/` prefix
needs no configuration; the script fails if an HTML or CSS file uses a
root-absolute URL. Weights still come from huggingface.co, whose `resolve`
URLs and CDN redirects send `access-control-allow-origin` for any origin.

## Verified (Apple M2, Chromium in the Claude browser pane and headless Chrome)

- Multilingual download 647 MB in 34–38 s, and 1.1 s from the cache after a reload.
- README triage example: every probability, score, noul, confidence and act
  probability is within **3e-4** of the Python fp16 fixture
  (`packages/laya-fixtures/data/real/multilingual.json`, case `en`), with 179
  input tokens (identical).
- Latency for 3 questions and one batch: about 700 ms on the first run
  (pipeline compilation), then 70–100 ms.
- No console errors.

Compare mode, the structured state builder, batch mode, and the result
queue/export (added after the above was recorded) have **not** been
exercised interactively in a real browser -- they were built and checked in
a sandbox without one. Each was verified by: a clean `tsc --noEmit`, a real
`bun scripts/build.ts` (which fails on any Node-only/native import leaking
into the bundle), a structural check that every new DOM id referenced from
`main.ts` exists in the built HTML/JS and no removed id lingers, `node
--check` on the bundle for syntax, and the `node:test` suite above for the
pure logic. If you're the first to click through them, treat that as the
real verification and update this section.

The multi-backend compare mode's remote path (`src/lib/remoteBackend.ts`)
got one further real check beyond the above: with a real
[`laya-server`](../laya-server) running locally, a Node script imported
`checkHealth`/`createRemoteAgent` directly (the same functions the browser
bundle calls) and ran a real MLX prediction through them end to end --
`checkHealth` correctly reported `mlx`/`onnx` available and `jev`
unavailable (no `TYPESAFE_API_KEY` in this sandbox), and the MLX prediction
matched the same server's own direct-`curl` result exactly. This confirms
the wrapper's request/response handling is correct; it is **not** a
substitute for clicking the actual checkboxes and pills in a browser.

## Limits

- The Cache API is per origin. Locally, the playground (port 5173) and
  snake-web (port 5174) each keep their own copy of the weights; on GitHub
  Pages both live on `johnhenry.github.io` and share one copy.
- The first run after loading compiles the WebGPU pipelines.
- A downloaded file is buffered into a Blob before it is cached (see
  hf-cache's limitations), so the browser briefly needs about the checkpoint
  size in memory.
- The native MLX backend is not available in browsers -- run
  [`laya-server`](../laya-server) locally and use compare mode's MLX
  checkbox instead. Same reasoning for ONNX and Jev (a native runtime and a
  server-side-only API key, respectively).
