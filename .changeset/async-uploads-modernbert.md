---
"@johnhenry/modernbert": minor
---

**Breaking: loading and running the encoder is async**, because uploads are (`@johnhenry/tensor-backend` 0.2). `loadModernBert` returns `Promise<ModernBert>` and starts every weight upload before awaiting any (validation first; on failure the uploaded tensors are disposed), so load time is unchanged. `forward` and `embed` return Promises; `uploadAs` returns `Promise<T>`.

New: `uploadInputs` / `encode` / `disposeInputs` (upload a batch's ids and masks together, then run the encoder synchronously, e.g. under `compile`), `EncoderInputs`, and the batch-loading helpers `loadInBatch` and `settleUploads`.

Migration: `await loadModernBert(...)`, `await encoder.forward(...)`, `await encoder.embed(...)`, `await uploadAs(...)`; for a synchronous encoder pass use `uploadInputs` + `encode`.
