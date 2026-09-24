# @johnhenry/modernbert

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Fmodernbert.svg)](https://www.npmjs.com/package/@johnhenry/modernbert)

ModernBERT / mmBERT encoder on any
[`@johnhenry/tensor-backend`](../tensor-backend) backend (CPU reference,
native MLX, WebGPU), loading safetensors checkpoints. A port of the encoder
in laya-mlx `laya_mlx/model.py` (which follows Hugging Face ModernBERT);
it uses only backend ops, so every backend gets it for free.

## Install

```bash
npm install @johnhenry/modernbert
bun add @johnhenry/modernbert
deno add jsr:@johnhenry/modernbert
```

Backend-agnostic: runs wherever the backend you pass runs (Node ≥ 24, Bun ≥ 1.2, Deno, browsers). To read checkpoints as in the example, also install `@johnhenry/math-plus-safetensors` (not a dependency: `safetensorsWeights` accepts any reader with the same structural shape).

```ts
import { readFile } from "node:fs/promises";
import { readSafetensors } from "@johnhenry/math-plus-safetensors";
import { createCpuBackend } from "@johnhenry/backend-cpu";
import { loadModernBert, parseModernBertConfig, safetensorsWeights } from "@johnhenry/modernbert";

const backend = createCpuBackend();
const config = parseModernBertConfig(JSON.parse(await readFile("encoder/config.json", "utf8")));
const weights = safetensorsWeights(readSafetensors(await readFile("model.safetensors")));
const encoder = await loadModernBert(backend, config, weights); // prefix auto-detected; uploads batched

// ids tokenized WITH special tokens, row-major [B, L]; mask 1 = token, 0 = padding
const vectors = await encoder.embedToHost(inputIds, attentionMask, B, L); // Float32Array [B*H]
```

## API

- `parseModernBertConfig(json): ModernBertConfig` — port of
  `EncoderConfig.from_dict` with its validation (model_type `modernbert`,
  `gelu`, even integral head dim, `layer_types` defaulting to
  `i % global_attn_every_n_layers (3) == 0 → full_attention`, default-type
  RoPE only). `ropeBase[kind]` = `rope_parameters[kind].rope_theta`, else
  `global_rope_theta` (160000) / `local_rope_theta` (10000). Also
  `normEps` (1e-5), `normBias`/`attentionBias`/`mlpBias` (false),
  `localAttention` (128), `headDim`.
- `loadModernBert(backend, config, weights, { dtype?: "f32"|"f16"|"bf16", prefix? }): Promise<ModernBert<T>>`
  - `weights`: `(name) => HostTensor | undefined` or `{ get(name) }`.
  - Names: laya-mlx `encoder.layers.N.attn.Wqkv.weight`, … or HF
    `model.layers.N.attn.Wqkv.weight`, `model.embeddings.*`, `model.final_norm.*`.
    `prefix` defaults to auto-detection (`encoder.`, `model.`, `""`).
    Layer 0 has no `attn_norm`. Shapes are validated.
  - Every upload is started before any is awaited (one batch), after the
    names and shapes are validated; if anything fails, the tensors that
    did upload are disposed.
  - Call it **outside** `backend.scope` (weights must outlive the scope).
- `class ModernBert<T>`
  - `forward(inputIds: Int32Array, attentionMask: Uint8Array, B, L, { onStage? }): Promise<T>` —
    hidden states `[B, L, H]` in the model dtype. Builds the full and
    sliding bool masks on the host and uploads them (`uploadInputs`); all
    intermediates are freed. `onStage(name, t)` sees `"embeddings"`,
    `"layers.<i>"`, `"final_norm"`; return `true` to keep a stage tensor
    (you dispose it).
  - `uploadInputs(inputIds, attentionMask, B, L): Promise<EncoderInputs<T>>`
    (ids, the bool padding mask and the attention masks, uploaded together),
    `encode(inputs, { onStage? }): T` (the synchronous forward pass on
    device tensors, e.g. for `compile`) and `disposeInputs(inputs)`.
  - `embed(inputIds, attentionMask, B, L): Promise<T>` — masked mean pool, `[B, H]` f32
    (laya-mlx `embed_fn_from_agent`, uses `meanPool` from tensor-backend).
  - `embedToHost(...)`: `Promise<Float32Array>`.
  - Building blocks: `embeddings(ids)`, `layer(i, x, mask)`, `finalNorm(x)`.
  - `dispose()` frees the weights.
- `attentionMasks(attentionMask, B, L, window): { full, sliding }` — host
  bool masks exactly like laya-mlx `attention_masks`: full `[B,1,1,L]`;
  sliding `[B,1,L,L]` = (|i−j| ≤ window // 2 **or** query i is padding)
  **and** key j valid. Padded query rows see all valid keys, avoiding
  all-masked softmax rows.
- `safetensorsWeights(file)`: adapts an in-memory
  `@johnhenry/math-plus-safetensors` file (F16 → f16, BF16 → bf16, F32 → f32).
- `uploadAs(backend, host, dtype): Promise<T>`, `toWeightGetter(src)`,
  `detectPrefix(get)`, and the batch-loading helpers `loadInBatch(backend,
  build, dtype)` (runs a pure weight-tree builder twice: once to validate
  and start every upload, once with the settled tensors) and
  `settleUploads(backend, jobs)` (awaits a `Map` of uploads; on failure
  disposes the ones that succeeded).

### Migrating from 0.1

Uploads are async in `@johnhenry/tensor-backend` 0.2, so everything that
uploads returns a Promise: `await loadModernBert(...)`, `await
encoder.forward(...)`, `await encoder.embed(...)`, `await uploadAs(...)`.
`embedToHost` was already async. To run the encoder synchronously (inside
`scope` or `compile`), upload first with `uploadInputs` and call `encode`.

## Accuracy

On the CPU backend, the tiny random checkpoint in `@johnhenry/laya-fixtures`
matches MLX fp32 at every stage (embeddings, each layer, final norm) within
1e-6 absolute, including padded positions. Mean-pooled embeddings of the
three published Laya checkpoints match the MLX fp32 fixture within ~8e-5
absolute (checked once with Python-tokenized ids; see Limitations).

## Limitations

- Inference only; no MLM head, no dropout, no unpadding/FlashAttention path.
- Only default (unscaled) RoPE; `hidden_activation` must be `gelu`.
- No tokenizer here: callers pass ids. The `embeddings` fixture test in
  `@johnhenry/laya` is skipped until a JS tokenizer exists.
- Sliding-window layers use a dense `[B,1,L,L]` bool mask (B·L² bytes).

## Family

Part of **[laya-js](https://github.com/johnhenry/laya-js#readme)**, Laya typed decisions in JavaScript on MLX, WebGPU and CPU — see its [package map](https://github.com/johnhenry/laya-js#which-package-do-i-want) and [results](https://github.com/johnhenry/laya-js#results).

- Runs on any [`@johnhenry/tensor-backend`](https://github.com/johnhenry/laya-js/tree/main/packages/tensor-backend) backend; [`@johnhenry/laya`](https://github.com/johnhenry/laya-js/tree/main/packages/laya) builds its decision heads on top of it.
- Weights come from [`@johnhenry/math-plus-safetensors`](https://github.com/johnhenry/math-plus/tree/main/packages/safetensors) (structural `SafetensorsFile` type, so any reader with the same shape works).

## License

Apache-2.0. Ports logic from [laya-mlx](https://github.com/mizorewww/laya-mlx) and [Laya](https://github.com/NandhaKishorM/laya) (both Apache-2.0); see [NOTICE](NOTICE).
