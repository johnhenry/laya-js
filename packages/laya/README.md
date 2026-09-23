# @johnhenry/laya

Laya typed decisions (choice / score / noul) in JavaScript: load a Laya
checkpoint and answer questions about a state, on **native MLX** (Node/Bun on
Apple Silicon), **WebGPU** (browsers; Node/Bun via Dawn) or the pure-TS **CPU**
reference. Port of laya-mlx `Agent` / `load` (`agent.py`, `shortlist.py`).

```ts
import { load } from "@johnhenry/laya";

const agent = await load("aac6fef/laya-mlx"); // HF repo or local directory
const result = await agent.predict("I was billed twice. Please refund the duplicate.", {
  department: { type: "choice", instructions: "Who should handle this?", criteria: ["billing", "technical", "sales"] },
});
console.log(result.answers.department); // { type, confidence, action, choice: "billing", probabilities }
agent.dispose();
```

On MLX the result is bit-identical to Python laya-mlx (see Parity).

## API

### `load(modelIdOrPath, opts?) → Promise<LayaAgent>`

| option | default | |
|---|---|---|
| `backend` | `"auto"` | a `Backend` instance (you own it), or `"auto"` \| `"mlx"` \| `"webgpu"` \| `"cpu"`. `auto` = MLX (macOS arm64 with a libmlxc) → WebGPU (adapter available) → CPU. Browsers: WebGPU → CPU. |
| `dtype` | `"f16"` | `"f16"` \| `"f32"`. The CPU backend (and WebGPU without `shader-f16`) computes in f32; `agent.dtype` reports what is used. |
| `device` | — | Python's `device`: `"gpu"`/`"metal"`/`"cpu"` selects the MLX device. |
| `revision`, `token`, `subfolder`, `offline`, `onProgress`, `fetch` | | passed to `@johnhenry/hf-cache` |
| `batchSize` | 16 | rows per forward pass |
| `padToMultiple` | null | bucket the padded length (capped at `max_len`) |
| `cachePrompts` | false | reuse tokenized question prefixes (laya-core `PrefixCache`) |
| `compile` | false | trace the forward pass with `backend.compile` (MLX `mlx_compile`, once per input shape); ignored by backends without it |
| `warn` | `console.warn` | receives the temperature-clamping warning |

Model resolution (`resolve_model`): an existing local directory is read from
disk; a path-looking string (`/`, `./`, `../`, `~`) that does not exist throws
`Local model directory does not exist: …`; anything else is a Hub repo fetched
into the huggingface_hub cache (`model.safetensors`, `rl_agent_config.json`,
`encoder/config.json`, `tokenizer/tokenizer.json`,
`tokenizer/tokenizer_config.json`). `offline: true` (or `$HF_HUB_OFFLINE`)
never touches the network. In browsers a Hub repo goes through the Cache API,
and a URL or URL path (`/models/tiny/`) is read with `fetch`.

Validation and messages follow Python: `subfolder must be a relative path…`,
`Not a complete Laya checkpoint: … is missing`, `Laya config must specify
encoder and head_layers`, `Expected 4 < head_max_len < max_len <=
max_position_embeddings`, `Calibration temperatures must be finite and
positive`, `batch_size must be a positive integer`, and so on. Loading is
strict, as with MLX `load_weights(strict=True)`: a checkpoint tensor the model
does not use throws `Received parameters not in model: …`. The `temperature`
buffer is ignored, as in laya-mlx.

Weights are opened with `openSafetensors` (the header, then coalesced reads).
Each tensor is released as soon as the backend has its copy, so the file is
not held twice.

### `LayaAgent`

- `predict(state, questions) → Promise<PredictResult>` (alias `systemOne`) is
  Python's `system_one`: `prepare` → chunks of `batchSize` → `collate` →
  forward → `formatResults` (all from `@johnhenry/laya-core`).
- `prepare(state, questions) → { items, internal }` and `forward(batch) → BatchOutputs`.
- `embed(texts, { maxLength = 512, batchSize = 32 }) → Float32Array[]` is
  `embed_fn_from_agent`: encoder states mean-pooled over real tokens, with special tokens.
- Properties: `modelId`, `backend`, `dtype`, `config`, `encoderConfig`,
  `tokenizer`, `batchSize`, `padToMultiple`, `model` (the `DecisionModel`),
  `temperature` and `temperatureByOptions` (clamped; these are applied),
  `temperatureRaw` and `temperatureByOptionsRaw` (as shipped).
- `dispose()` frees the weights, and the backend when `load` created it. It is idempotent.

### `createAgent(parts) → LayaAgent`

This does no I/O. `parts` is `{ backend, encoderConfig, agentConfig, weights,
tokenizer, dtype?, batchSize?, padToMultiple?, cachePrompts?, compile?,
modelId?, ownsBackend?, warn? }`. `weights` is any `WeightSource`, such as
`safetensorsWeights(readSafetensors(bytes))` or
`(await readWeights(pathOrUrlOrBlob)).get`. `tokenizer` comes from
`loadTokenizer(json, config)` in `@johnhenry/laya-core`.

### Shortlist (`shortlist.py`)

- `predictShortlist(agent, state, questions, { k = 20, embedFn?, predictOptions? })`
  keeps the top `k` options of every choice question by cosine similarity, runs
  one `predict`, and adds `shortlist[qid] = { labels, scores, k, n, passthrough }`.
  `embedFn` defaults to `agent.embed`. `agent` can be anything with `predict` or
  `systemOne`, such as a Router; `predictOptions` is passed as its third argument.
- `shortlistChoice(state, criteria, embedFn, k = 20, { instructions? }) → labels`.

### Lower level

- `loadDecisionModel(backend, { encoderConfig, agentConfig, weights, dtype })`
  returns a `DecisionModel` with `forward`, `forwardTensors`, `uploadBatch`,
  `forwardCore` (pure function of device tensors), `compiled`, `readOutputs`,
  `headLayer` and `dispose`.
- `readWeights`, `consumingWeights`, `sanitizeName` (upstream PyTorch names →
  laya-mlx names), `createBackend(name)`, `readCheckpoint(id)`, `validateConfig`.

## Packaging

- `@johnhenry/backend-mlx` and `@johnhenry/backend-webgpu` are **optional peer
  dependencies**. Each is loaded with a dynamic `import()` only when selected,
  so install the ones you use. `@johnhenry/backend-cpu` is a regular dependency.
- Runtime-specific code sits behind the package's `#io` import map. The
  `browser` condition selects `io-browser.ts`: hf-cache's Cache API store,
  WebGPU or CPU only, and no `node:` imports. Every other runtime gets
  `io-node.ts`: the filesystem, the huggingface_hub disk cache and MLX.
  Bundlers that honour the `browser` condition (Vite, esbuild with
  `platform: "browser"`, webpack) never see `node:fs`, koffi or libmlxc.
- `createAgent`, `predictShortlist`, `loadDecisionModel` and the weights
  helpers do not depend on the runtime.

## Parity

`npm test -w @johnhenry/laya` runs the tiny fixture checkpoint end to end.
`predict` deep-equals Python's `predict.json` on the CPU backend. When MLX and
WebGPU are available, their results are within 1e-4 (f32) and 0.02 (f16).

The published checkpoints are opt-in: `LAYA_REAL=1 npm test -w
@johnhenry/laya` takes about 2 minutes; take `~/gpu.lock` first. It loads each
of the three repos from the local HF cache with `offline: true`, then runs
every fixture case (16 cases, 63 questions) and compares with Python's
results. Measured on an Apple M2 (Node 24.9, macOS 27):

| checkpoint | backend | choices | argmax | exact answers | max \|Δ\| | embed (rel) |
|---|---|---|---|---|---|---|
| english | mlx f32 | 22/22 | 63/63 | 63/63 | 0 | 0 |
| english | mlx f16 | 22/22 | 63/63 | 63/63 | 0 | — |
| english | webgpu f32 | 22/22 | 63/63 | 62/63 | 1.00e-4 | 5.75e-6 |
| english | webgpu f16 | 22/22 | 63/63 | 14/63 | 3.20e-3 | — |
| multilingual | mlx f32 | 22/22 | 63/63 | 63/63 | 0 | 0 |
| multilingual | mlx f16 | 22/22 | 63/63 | 63/63 | 0 | — |
| multilingual | webgpu f32 | 22/22 | 63/63 | 63/63 | 0 | 5.77e-6 |
| multilingual | webgpu f16 | 22/22 | 63/63 | 27/63 | 1.10e-3 | — |
| typed-decisions | mlx f32 | 22/22 | 63/63 | 63/63 | 0 | 0 |
| typed-decisions | mlx f16 | 22/22 | 63/63 | 63/63 | 0 | — |
| typed-decisions | webgpu f32 | 22/22 | 63/63 | 63/63 | 0 | 5.57e-6 |
| typed-decisions | webgpu f16 | 22/22 | 63/63 | 8/63 | 3.80e-3 | — |

- f32 rows compare with Python `result_fp32` (tolerance 1e-4); f16 rows
  compare with `result_fp16` (tolerance 0.02).
- "exact" means the answer object deep-equals Python's, including the
  4-decimal rounding. "argmax" is the chosen label, the most likely score
  level, or the noul side.
- MLX runs the same libmlx kernels as Python and matches exactly in both
  dtypes.
- WebGPU f16 has its own f16 kernels, so results differ from the third
  decimal on. Choices still agree.
- `LAYA_REAL_CPU=1` adds one English case on the CPU backend (about 1 minute).
  `LAYA_REAL_CPU_FULL=1` runs the Phase 1 DecisionModel CPU test on all 63
  questions (7–19 minutes per checkpoint). Its results follow.

### DecisionModel on the CPU backend (Phase 1)

- Tiny fixture checkpoint: every stage (embeddings, 4 encoder layers, final
  norm, type_emb_added, 2 head layers, logits, act) within 1e-6 absolute of
  MLX fp32, valid and padded positions alike.
- Published checkpoints: `LAYA_REAL_CPU_FULL=1 npm test -w @johnhenry/laya` (opt-in,
  slow; `LAYA_REAL_MODELS=english,...` to select) compares all 63 validation
  questions per checkpoint against the MLX fp32 fixture outputs.
  Measured once (Apple M2, Node 24, single thread, fp16 weights widened to f32):

  | checkpoint | argmax | max abs Δlogit | max abs Δact (rel) | forward time |
  |---|---|---|---|---|
  | english (ModernBERT-large) | 63/63 | 2.9e-5 | 2.7e-2 (7.9e-6) | 981 s (61 s/case) |
  | multilingual (mmBERT-base) | 63/63 | 3.1e-5 | 1.5e-3 (9.4e-7) | 430 s (27 s/case) |
  | typed-decisions (ModernBERT-large) | 63/63 | 4.0e-5 | 1.8e-2 (3.8e-6) | 1117 s (70 s/case)* |

  \* ran concurrently with multilingual. Act logits reach |x| ≈ 4000, so
  their error is best read relatively.


## Limitations

- `embed` does not run texts that tokenize to nothing; it returns the zero
  vector for them. Python computes `sum(h·0) / max(0, 1)`, and an all-masked
  attention row is undefined on the tensor backends.
- Under Bun, backend-webgpu uploads TypedArray views with a non-zero
  `byteOffset` wrongly, because the Dawn binding's `writeBuffer` ignores the
  offset. On that combination only, `createAgent` copies such weight views.
- The browser loader is tested in Node against a local HTTP server (base-URL
  mode), and a Bun browser bundle is checked for node-only modules. It has not
  been run in a real browser here, and the Hub/Cache API path is covered only
  by hf-cache's own tests.
- `compile` needs a backend with `compile` (MLX on the GPU). Other backends
  ignore it.
- The clamping warning starts with `laya: …` (laya-core's text). Python
  laya-mlx starts it with `laya-mlx: …`; the rest of the message is the same.
- Integer-like choice labels in a criteria *object* follow JS key order (see
  laya-core). Pass a list to keep your order.
- WebGPU is much slower than MLX on an Apple M2: one short question takes
  about 250 ms, against 38 ms on MLX (see `@johnhenry/laya-cli` bench).
