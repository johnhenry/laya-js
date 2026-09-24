---
"@johnhenry/laya": minor
"@johnhenry/laya-cli": minor
---

Quantized checkpoints (q8/q4, dequantize-on-load). `load()` / `readWeights()` detect `__metadata__.laya_quant` in `model.safetensors` and dequantize 8-bit (symmetric, groups of 64) and 4-bit (affine, groups of 64, optional q8 tensors) weights to f16/f32 tensor by tensor while loading, so downloads shrink to ~52% (q8) or ~28–35% (q4) with no backend changes. New `quantizeMatrix` / `dequantizeMatrix` / `quantizeSafetensors` / `quantMetadata` exports; `load()` in Node/Bun now also accepts an http(s) base URL (Range reads, as in browsers). New `laya quantize --model <repo|dir> --bits 8|4 --out <dir>` command writes a complete quantized checkpoint directory.
