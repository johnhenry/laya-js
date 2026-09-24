---
"@johnhenry/modernbert": minor
"@johnhenry/laya": minor
---

Quantized checkpoints now stay quantized on the device on MLX and WebGPU (less device memory): `load(..., { quantized: "device" | "dequantize" })`, default `"device"` when the backend has native quantized ops, else host dequantization as before. `agent.model.quantizedOnDevice` reports which. modernbert and `DecisionModel` accept `HostQuantized` Linear weights and token/type embeddings (`HostWeight`, `MatrixWeight`, `disposeWeight`); `readWeights(src, { quantized: "device" })` and `hostQuantized(matrix)` hand out packed matrices. Optional peer ranges on backend-mlx/backend-webgpu widened to `^0.4.0`.
