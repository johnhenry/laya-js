# Laya server (local backend proxy for web-playground)

A tiny local HTTP server that runs Laya predictions on three backends
[`web-playground`](../web-playground) can't run in the browser: **native
MLX**, **ONNX** (via [`@receptron/laya`](https://github.com/receptron/laya)),
and **Jev** ([TypeSafe](https://docs.typesafe.ai)'s hosted typed-decision
API). Point the playground's Compare mode at it (default
`http://localhost:5199`) to run any of MLX/WebGPU/CPU/ONNX/Jev side by side
against the same state and questions.

Why a server at all: MLX is native FFI and ONNX runs on
`onnxruntime-node` -- neither can run in a browser tab, full stop. Jev needs
a `TYPESAFE_API_KEY`, and a paid API key must never ship inside browser JS.
Since two of the three backends already need Node, all three are proxied
through this one process instead of three different mechanisms -- the key
only ever lives in this server's environment.

## The backends

- **`mlx`** -- [`@johnhenry/laya`](../../packages/laya) with `backend: "mlx"`,
  loaded offline from the local Hugging Face cache (same convention as every
  terminal demo in this repo). Any of the three published checkpoints work;
  pass `repo` in the request body (default `aac6fef/laya-mlx`). Apple Silicon
  only.
- **`onnx`** -- [`@receptron/laya`](https://github.com/receptron/laya)'s
  `Laya.load()`, an **independent** ONNX export of the same upstream model
  (`convaiinnovations/laya`), verified against the PyTorch reference to
  ~1e-5 max logit difference. It publishes **one** checkpoint today: the
  English/ModernBERT-large one, the ONNX equivalent of `aac6fef/laya-mlx`
  specifically -- not the multilingual or typed-decisions checkpoints. First
  use downloads ~1.7 GB to `~/.cache/receptron-laya` (override with
  `LAYA_CACHE`).
- **`jev`** -- [`@typesafe-ai/sdk`](https://docs.typesafe.ai)'s
  `TypeSafeClient`, reading `TYPESAFE_API_KEY` from the environment. A real,
  separate, paid commercial API (~$0.042 / 1M input tokens, output free) --
  not a Laya checkpoint at all, just structurally close enough to compare
  answers side by side. **Jev's answers have no equivalent to Laya's
  `action.act_probability`** (it has no RL-agent "should I act" concept) --
  mapped to `NaN` here (`mapJevResult` in `src/mapping.ts`), rendered as
  "N/A" by the playground rather than a misleading `0.0000`.

## Run

```bash
npm start -w @johnhenry/example-laya-server                # node, port 5199
npm run start:bun -w @johnhenry/example-laya-server         # same under Bun
TYPESAFE_API_KEY=sk-... npm start -w @johnhenry/example-laya-server
PORT=5555 npm start -w @johnhenry/example-laya-server        # different port
```

```
GET  /health            -> { mlx: {available, reason?}, onnx: {...}, jev: {...} }
POST /predict           body: { backend: "mlx"|"onnx"|"jev", repo?, dtype?, state, questions }
                        -> { result, seconds, engine } | { error }
```

`onnx` and `jev` each hold one cached instance/client; `mlx` caches one
agent per `repo:dtype` pair. CORS is wide open (`Access-Control-Allow-Origin: *`)
-- this is a single-user local dev tool, not a public service.

## Tests (`npm test` / `npm run test:bun`)

`test/mapping.test.ts` covers the two pure response-mapping functions
(`mapOnnxResult`, `mapJevResult`) and the choice-criteria adapter
(`toJevQuestions`) against canned SDK responses -- no network or native call
in the suite. **4 cases, 0 skipped.**

## Verified so far

- `npm run typecheck` clean; `node --test test/*.test.ts` 4/4 pass.
- A real end-to-end `POST /predict` smoke test against all three backends,
  started fresh (`node --conditions=source src/server.ts`):
  - **`mlx`**, `aac6fef/laya-mlx`: a real `noul` + `choice` question pair
    ("Is this about billing?" / "Which department?") on the state "I was
    charged twice for the same order." answered correctly (`noul: 0.9228`,
    `choice: "billing"` at 98.3% probability) in 1.25 s including agent load.
  - **`onnx`**: the same `noul` question against the same state answered
    **`noul: 0.9228`** -- identical to MLX to 4 decimal places, a real,
    direct confirmation that `@receptron/laya`'s independent export is
    faithful to the same model -- in 5.47 s including `Laya.load()` (weights
    were already cached from earlier research in this session).
  - **`jev`**: no `TYPESAFE_API_KEY` was available in this sandbox, so only
    the missing-key path was verified for real (`/health` reports
    `{available: false, reason: "TYPESAFE_API_KEY is not set"}`, and
    `/predict` returns the same error rather than a crash). A live paid call
    needs a real key -- set `TYPESAFE_API_KEY` and try it yourself; treat
    that as the real verification of this path.
- `bun ../web-playground/scripts/build.ts` (the browser bundle's native/
  Node-module leak guard) still passes unchanged -- this server lives
  entirely outside `web-playground/src/`, so it was never in scope for that
  guard to scan in the first place.

## Limits

- Not a production service: no auth, permissive CORS, in-memory-only caches
  that reset on restart. Meant to run on `localhost` next to the playground,
  nothing else.
- `onnx` and `jev`'s `usage.input_tokens` is whatever each SDK reports, not
  independently re-derived here -- their tokenizers aren't guaranteed to
  count the same way Laya's own does, so token counts across backends are
  informative, not directly comparable.
- No batching across concurrent requests -- each `/predict` call is one
  forward pass (mlx/onnx) or one HTTP round trip (jev), same as the
  playground's own sequential compare-mode design.
