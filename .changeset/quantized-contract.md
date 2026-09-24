---
"@johnhenry/tensor-backend": minor
---

Optional quantized-weight ops: `fromHostQuantized`, `quantizedLinear` (x · dequant(W)ᵀ + bias, f32 accumulation) and `quantizedEmbedding`, with the `HostQuantized` (laya-js checkpoint layout: 8-bit or packed 4-bit values, per-group scales/biases, symmetric or affine) and `QuantizedTensor` types. Call them through the new `compose.ts` helpers `uploadQuantized`, `quantizedLinear`, `quantizedEmbedding`, `disposeQuantized`, `linearAny`, `embeddingAny`; backends without the native ops get a default composition that dequantizes on the device. New conformance fixture `fixtures/ops-quantized.json` (26 cases from `scripts/gen_quantized_cases.py`, cross-checked against `mx.quantized_matmul`), loaded by `loadOpCases()`. Additive: see "Migrating from 0.2" in the README.
