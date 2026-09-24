---
"@johnhenry/backend-mlx": minor
---

Native quantized weights: `fromHostQuantized` repacks laya-js q8/q4 matrices into MLX's affine uint32 layout without dequantizing (q4 uploads as-is; symmetric q8 flips each byte's sign bit with bias = −128·scale, bit-exact), `quantizedLinear` is `mlx_quantized_matmul`, `quantizedEmbedding` gathers packed rows and runs `mlx_dequantize`. Groups of 32/64/128.
