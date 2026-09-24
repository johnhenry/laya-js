---
"@johnhenry/laya": minor
---

**Breaking (lower-level API only): uploads are async** (`@johnhenry/tensor-backend` 0.2). `load()` and `predict()` are unchanged. `createAgent` returns `Promise<LayaAgent>`; `loadDecisionModel` returns `Promise<DecisionModel>` and uploads the encoder and head weights in one batch; `DecisionModel.uploadBatch` and `forwardTensors` are async (a batch's eight tensors upload together); `compiled()` returns an async function. The forward pass's constants (−1e4, 1e-9, 255) are uploaded once with the weights (`DecisionWeights.constants`), so `forwardCore` stays synchronous and the MLX graph is unchanged (English parity 63/63, MLX f32 exact). Load time and single-question `predict` latency are unchanged (see the PR's before/after table). New: `DecisionModel.disposeInputs`.

Migration: `await createAgent(parts)`, `await loadDecisionModel(...)`, `await model.forwardTensors(batch)`, `await model.uploadBatch(batch)`; code constructing `DecisionWeights` by hand must add `constants`.
