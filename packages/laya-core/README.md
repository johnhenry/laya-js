# @johnhenry/laya-core

Tensor-free Laya typed-decision logic: question validation, prompt construction,
tokenization, collation, calibration and result formatting. It is the part of
laya-mlx (`common.py`, `agent.py`, `prepared.py`, `tokenizer.py`) that does not touch a
tensor, ported so that a backend only has to run the encoder: given the same logits,
`formatResults` returns the same result object Python does, bit for bit.

Works in browsers (no `fs` in the main entry); `@johnhenry/laya-core/node` adds a
filesystem loader.

```js
import { prepare, collate, resolveTemperatures, formatResults, chunkItems } from "@johnhenry/laya-core";
import { loadTokenizerFromDir } from "@johnhenry/laya-core/node";

const tok = await loadTokenizerFromDir(`${checkpoint}/tokenizer`);
const cfg = JSON.parse(await readFile(`${checkpoint}/rl_agent_config.json`, "utf8"));
const temps = resolveTemperatures(cfg);             // clamped, like Agent.__init__
if (temps.warning) console.warn(temps.warning);

const { items, internal } = prepare(tok, state, questions, cfg);
const outputs = chunkItems(items, 16).map((chunk) => {
  const batch = collate(chunk, tok.padTokenId, { maxLength: cfg.max_len ?? 512 });
  return runModel(batch);                            // your backend -> { logits, act, nAct }
});
const result = formatResults(internal, items, outputs, temps);
```

## API

Contract types (`types.ts`): `Question`, `Questions`, `State`, `InternalQuestion`,
`LayaTokenizer`, `PreparedItem`, `Batch`, `BatchOutputs`, `AgentConfig`, `Answer`,
`PredictResult`, `QTYPES`.

| Export | Python |
|---|---|
| `toInternal(qdef)` | `Agent._to_internal` (same validation messages) |
| `prepare(tok, state, questions, cfg?, { cache? })` → `{ items, internal }` | `Agent.prepare`; `internal[i].id` is the question id |
| `PrefixCache(capacity = 128)` | `prepared.PrefixCache` (LRU of tokenized question prefixes) |
| `collate(items, padId, { padToMultiple?, maxLength? })` → `Batch` | `collate_items` |
| `chunkItems(items, batchSize)` | the `system_one` chunk loop |
| `resolveTemperatures(cfg)` → `{ temperature, temperatureByOptions, temperatureRaw, temperatureByOptionsRaw, rejected, warning }` | `Agent.__init__` calibration block |
| `temperatureFor(temps, qtype, k)` | bucket lookup in `system_one` |
| `formatResults(internal, items, outputs, temps)` → `PredictResult` | the result half of `system_one` |
| `serializeState`, `renderCriterion`, `renderOptions`, `buildPrefix`, `buildSequence` | `common.py` |
| `confidenceFromProbs`, `tempBucket`, `clampTemperature`, `TEMP_MIN`, `TEMP_MAX` | `common.py` |
| `loadTokenizer(tokenizerJson, tokenizerConfig)` → `HFLayaTokenizer` | `tokenizer.Tokenizer` |
| `loadTokenizerFromDir(dir)` (from `/node`) | `tokenizer.Tokenizer(path)` |
| `softmaxF32`, `sumF32`, `sumF64` | the numpy expressions used for probabilities |

`formatResults` takes one `BatchOutputs` per chunk, in item order; each chunk's row count is
`act.length / nAct` and its marker stride `logits.length / rows`, so any chunk size works.
Probabilities are computed like numpy 2 on f32 logits (f32 division by the f32-cast
temperature, f32 `exp`, numpy's pairwise f32 summation), then rounded with CPython
`round(x, 4)` via `@johnhenry/pyjson`.

## Parity

Tested against golden data from laya-mlx (`@johnhenry/laya-fixtures`):

- `prepare` reproduces the model inputs (ids, markers, qtype) of all 16 parity cases on all
  three published checkpoints (English, multilingual, typed-decisions), cached and uncached,
  plus the tiny WordLevel checkpoint.
- `formatResults` fed the recorded fp32 logits deep-equals Python's `predict()` result for
  all 48 case/checkpoint pairs.
- Tokenizer ids (with and without special tokens) match the Rust `tokenizers` crate on the
  per-checkpoint tokenizer tables.

## Tokenizer compatibility patches

`@huggingface/tokenizers` (tokenizers.js 0.2.0) differs from the Rust crate in ways that
change token ids. `loadTokenizer` applies these fixes, driven by `tokenizer.json`
(`tok.patches` lists what was applied):

- **`Metaspace.split`** — tokenizers.js ignores Metaspace `split: true` (the Rust default),
  so the Gemma-style multilingual tokenizer merged runs of spaces (`"  leading"` →
  `▁▁` + `leading` instead of `▁` + `▁leading`). The patch splits pre-tokens before each `▁`
  (Rust `MergedWithNext`).
- **`Whitespace.unicode`** — tokenizers.js' `Whitespace` pre-tokenizer uses ASCII `\w`;
  Rust's is Unicode-aware.
- **`WordLevel.unk`** — tokenizers.js has no WordLevel model; its fallback reads
  `unk_token` from `tokenizer_config.json`, so unknown words became `undefined`.

## Limitations

- **Integer-like choice labels.** A JS object orders integer-like keys (`"10"`, `"2"`) before
  other keys, numerically. A choice *list* keeps its order (`InternalQuestion.labels`), but
  a criteria *object* such as `{ b: …, "1": … }` is seen in JS order, which changes the option
  order the model sees compared with Python.
- **int vs float in JSON-rendered values.** State objects, structured criteria and
  non-string instructions go through `json.dumps`; a JS `1.0` renders as `1` (Python would
  write `1.0`). Wrap with `pyFloat` from `@johnhenry/pyjson` when it matters.
- `formatResults` emulates the fp32 path. Python's fp16 agent does the softmax in float16;
  feeding f32 logits from an fp16 model gives results within rounding of Python's fp16
  results, not bit-identical ones.
- `f32 exp`/`log` are `Math.fround(Math.exp(x))`; numpy uses the platform `expf`/`logf`.
  These agree except in astronomically rare double-rounding cases.
