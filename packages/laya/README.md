# @johnhenry/laya

Laya typed decisions in JavaScript on MLX, WebGPU or CPU.

> Status: under construction. See the root README.

## DecisionModel (`src/model.ts`)

Backend-generic port of laya-mlx `DecisionModel.__call__`: ModernBERT
encoder (`@johnhenry/modernbert`) → `+ type_emb[qtype]` → `head_layers`
PyTorch pre-norm `TransformerEncoderLayer`s (ReLU FFN, `max(1, D // 64)`
heads, biases, key-padding mask) → gather marker rows at
`max(marker_pos, 0)` → scorer (LayerNorm → Linear → GELU → Linear) →
logits (masked slots = −1e4) → action features
`[top1, top1 − top2, entropy / log k, k / 255]` with `k = max(#markers, 2)`
per row → `act_head` on `[h[:, 0], features]`.

```ts
import { loadDecisionModel } from "@johnhenry/laya/src/model.ts"; // public export pending (see below)

const model = loadDecisionModel(backend, {
  encoderConfig,              // encoder/config.json (raw) or parseModernBertConfig(...)
  agentConfig,                // rl_agent_config.json
  weights: safetensorsWeights(readSafetensors(bytes)),
  dtype: "f32",               // or "f16"/"bf16" on backends that support it
});
const { logits, act, nAct } = await model.forward(batch); // Batch from @johnhenry/laya-core
```

- `loadDecisionModel(backend, { encoderConfig, agentConfig, weights, dtype? }): DecisionModel<T>`
  — laya-mlx weight names (`head.layers.N.self_attn.in_proj.weight`,
  `scorer.layers.{0,1,3}`, `act_head.layers.{0,2}`, `type_emb.weight`,
  `encoder.*`); upstream PyTorch spellings (`in_proj_weight`, `scorer.0.*`,
  `act_head.0.*`) are accepted too. Shapes are validated.
- `DecisionModel.forward(batch, { onStage? }): Promise<BatchOutputs>` —
  `logits` f32 `[B, M]`, `act` f32 `[B, nAct]` (raw logits, not softmaxed).
- `DecisionModel.forwardTensors(batch, { onStage? }): { logits: T, act: T }` —
  same, without reading back (caller disposes).
- `onStage(name, t)` observes `embeddings`, `encoder.layers.<i>`,
  `encoder.final_norm`, `type_emb_added`, `head.layers.<j>` (return `true`
  to keep the tensor).
- `headLayer(j, x, mask)`, `dispose()`, `nAct`, `headHeads`, `hiddenSize`.

### Parity (CPU backend, f32)

- Tiny fixture checkpoint: every stage (embeddings, 4 encoder layers, final
  norm, type_emb_added, 2 head layers, logits, act) within 1e-6 absolute of
  MLX fp32, valid and padded positions alike.
- Published checkpoints: `LAYA_REAL=1 npm test -w @johnhenry/laya` (opt-in,
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

### Limitations

- `markerCount` must be ≥ 2 (collation pads one-option questions to two slots).
- The `embeddings` fixture test is skipped: it needs a JS tokenizer.
