---
"@johnhenry/backend-webgpu": minor
---

Performance: faster subgroup-matrix GEMM and cheaper dispatch.

- Subgroup-matrix Linears: 64×64 workgroup tiles (8×4 fragments per subgroup) for large M, chosen per shape by grid size and row padding (`minGroups`, `maxPad` in `GemmConfig.sg`); f16 operands are loaded 8 halves at a time (`vec4<u32>` + `unpack2x16float`); unpadded staging panels. Still f32 accumulation. Optional split-K (`splitK`), double buffering (`db`) and a whole-block epilogue (`epi: "block"`) are available for tuning but off by default (slower on Apple M2).
- `backend.tuneGemm(shapes)` measures the Linear kernel choice per exact shape on the current device; results are shared by backends on the same `GPUDevice` (`backend.gemmTuning`) and can be restored with the `gemmTuning` option.
- The buffer pool hands out the lowest-id free buffer of a size class instead of LIFO, so repeated forwards reuse the same buffers and bind groups are always cache hits (previously ~170 new bind groups per ModernBERT-large forward).
- `firstBatch` option (default 24): the first submit after the GPU goes idle carries fewer dispatches, so the GPU starts sooner.
- Bool uploads convert with a plain loop (the B×L×L sliding-window masks were ~40 ms at B=16, L=256).
- `rope` computes both halves of a rotation pair per thread.
- `GEMM_V020` exports the previous GEMM defaults for comparisons.
