---
"@johnhenry/backend-webgpu": minor
---

Native quantized weights: q8/q4 matrices stay packed in a `u32` buffer with f16 scales/biases; every Linear kernel (skinny, subgroup-matrix incl. split-K, direct, tiled) dequantizes in its B tile load and accumulates in f32; `quantizedEmbedding` is a dequantizing gather. New `bench/quantized-gemm.ts`.
