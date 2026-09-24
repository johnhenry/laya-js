---
"@johnhenry/backend-webgpu": minor
"@johnhenry/laya": patch
---

backend-webgpu: quantized Linears (q8/q4 kept on the device) are now at least as fast as fp16 end to end on WebGPU (M2: English q8 one question 51.3 vs 54.6 ms, 16 questions 641 vs 654 ms; 0.4.0 was 9–16% slower).

- New `qmv` kernel for M ≤ 48 (64 for N < 2048): 8 threads per weight row, one word load per 8 (or 16) values, unpacked and scaled in registers, f32 partials reduced in workgroup memory or with subgroup shuffles (the `subgroups` feature is now requested when available). 1.1–5.7× fp16 GFLOP/s for M ≤ 8, median 1.06× for 16–64, and used at every M when subgroup matrices are unavailable (browsers without the flag), where 0.4.0 fell back to the direct kernel.
- Subgroup-matrix kernel with quantized B: 8-value loads, raw words and scale fetched into registers and dequantized when stored to workgroup memory (after the MMAs), A converted late too, and quantized-specific tile rules (BM 64/48/32, split-K). median 1.02× fp16 for M ≥ 93 (0.80–0.92× before).
- Dequantization without int→float conversions (f16 exponent trick). With f16 activations every path still multiplies by exactly fl16(q·scale + bias); new tests check this weight by weight (subnormals included) and that random products round like the exact sum.
- `GemmConfig.quant` (`QUANT_GEMM_DEFAULT`, `QUANT_GEMM_NAVIGATOR` for `navigator.gpu`, `quant: null` restores the 0.4.0 choice), `QmvGemmConfig`, `WebGpuBackend.subgroupSize`, `GemmChoice` "qmv", and `tuneGemm(shapes, { quantized: { bits, groupSize?, mode? } })`.
- Remaining gaps (documented): 0.92–0.97× on the large-N Laya shapes around M = 33 and 0.83–0.97× in a few cells at M = 93–128.

laya: `load(url)` (http(s) checkpoints) now passes `quantized` to the weight reader, so quantized checkpoints loaded from a URL stay quantized on MLX/WebGPU instead of always being dequantized on the host.
