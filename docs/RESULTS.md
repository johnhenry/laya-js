# Results: parity and performance

All numbers come from one machine: an **Apple M2 with a 10-core GPU, fanless,
running macOS 27**, with Node 24.9.0, Bun 1.2.17 and MLX 0.32.2. This is *not*
the M3 Max behind laya-mlx's published figures (13.4 ms), so compare against
the Python column measured here, not against laya-mlx's README.

> **Thermal caveat.** A fanless M2 drops to about 35% of its cold GPU speed
> after roughly 10 s of sustained load and recovers after about 5 s idle.
> Benchmarks that loop for a long time (`laya bench`, long Snake runs) measure
> the throttled machine. The latency grid below uses
> `packages/backend-webgpu/bench/grid.ts`: it idles 5 s before each cell, times
> for at most 1 s, and alternates backends cell by cell within one process.

## Parity against Python laya-mlx

**Question set:** the 16 cases / 63 questions from laya-mlx's
`benchmarks/common.py:parity_cases()`, covering 8 languages, empty and long
states, mask literals, structured criteria and 20 options.

**Reference:** golden outputs from `laya-mlx/scripts/dump_js_fixtures.py`:
- fp32 run on Metal;
- fp16 results from the fp16 agent.

**How to rerun:** `LAYA_REAL=1 npm test -w @johnhenry/laya`, which runs
`packages/laya/test/e2e-real.test.ts`.

**Tolerances:** within 1e-4 of the fp32 result for f32 runs, and within 0.02 of
the fp16 result for f16 runs. Choices and argmax must be identical.

| checkpoint | backend | choices | argmax | answers bit-identical to Python | max \|Δp\| | embed rel. err |
|---|---|---|---|---|---|---|
| english (421M) | mlx f32 | 22/22 | 63/63 | 63/63 | 0 | 0 |
| english | mlx f16 | 22/22 | 63/63 | 63/63 | 0 | — |
| english | webgpu f32 | 22/22 | 63/63 | 62/63 | 1.0e-4 | 5.8e-6 |
| english | webgpu f16 | 22/22 | 63/63 | 13/63 | 5.0e-3 | — |
| multilingual (322M) | mlx f32 | 22/22 | 63/63 | 63/63 | 0 | 0 |
| multilingual | mlx f16 | 22/22 | 63/63 | 63/63 | 0 | — |
| multilingual | webgpu f32 | 22/22 | 63/63 | 63/63 | 0 | 5.8e-6 |
| multilingual | webgpu f16 | 22/22 | 63/63 | 26/63 | 1.3e-3 | — |
| typed-decisions | mlx f32 | 22/22 | 63/63 | 63/63 | 0 | 0 |
| typed-decisions | mlx f16 | 22/22 | 63/63 | 63/63 | 0 | — |
| typed-decisions | webgpu f32 | 22/22 | 63/63 | 62/63 | 1.0e-4 | 5.6e-6 |
| typed-decisions | webgpu f16 | 22/22 | 63/63 | 16/63 | 1.3e-3 | — |
| english | cpu f32 (smallest case only, 21 s) | ✓ | ✓ | exact | 0 | — |

- **"Bit-identical"** means the whole answer object (4-decimal rounding
  included) equals Python's. On MLX that holds in fp16 too: the JS binding
  drives the same MLX kernels.
- **WebGPU f16 differences** come from different f16 kernels, not from a bug.
  They are well inside laya-mlx's own fp16 tolerance of 0.02.

The CPU reference backend also matches all 63 questions on all three checkpoints
(Workstream C). Its max logit error is 2.9e-5 to 4.0e-5, but it takes 27–70 s
per case.

**Lower levels** (tests that run by default):
- Token ids and markers are identical to Python for all 3 tokenizers × 16
  cases, including the 256k-vocabulary mmBERT tokenizer (after three
  tokenizers.js workarounds in `laya-core`).
- `formatResults` given the fixture logits deep-equals Python's result in all
  48 cases.
- On the tiny random checkpoint, every stage is within 1e-6 of MLX on cpu, mlx
  and webgpu.
- Every backend passes the op conformance suite in f32, and mlx and webgpu also
  pass it in f16.
- Snake prompts are token-identical to Python on 40 recorded frames. Replaying
  Python's moves reproduces all 2,440 recorded boards.

## Latency: English checkpoint, f16, one forward pass

These are medians in ms with a cold GPU. B is batch rows, L is tokens per row.
WebGPU runs on Node through Dawn; Bun is within ±3%.

| L | 16 | 33 | 64 | 93 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B=1 MLX (JS) | 20.0 | 23.6 | 23.6 | 41.7 | 39.0 | 80.4 | 154.4 |
| B=1 WebGPU | 14.6 | 24.1 | 37.9 | 53.8 | 67.2 | 130.2 | 252.2 |
| B=1 WebGPU 0.2.0 | 15.4 | 25.7 | 39.4 | 59.0 | 77.0 | 149.3 | 288.2 |
| B=3 MLX (JS) | 25.4 | 43.7 | 62.1 | 95.5 | 107.4 | 220.9 | 442.6 |
| B=3 WebGPU | 30.8 | 67.0 | 90.6 | 141.7 | 175.1 | 354.1 | 727.3 |
| B=3 WebGPU 0.2.0 | 32.1 | 71.1 | 106.0 | 148.3 | 199.6 | 398.7 | 836.0 |
| B=16 MLX (JS) | 79.7 | 159.4 | 271.7 | 413.2 | 545.9 | 1118 | 2367 |
| B=16 WebGPU | 119.6 | 252.5 | 435.7 | 653.4 | 886.3 | 1821 | 3843 |
| B=16 WebGPU 0.2.0 | 129.7 | 279.0 | 488.7 | 723.6 | 992.2 | 2067 | 4440 |

All three rows of a block come from one Node process, interleaved cell by cell
(`BACKEND=webgpu-main,webgpu,mlx`; "0.2.0" is the unmodified backend from
`main`, copied to `packages/backend-webgpu/.base/`). WebGPU is now 4–15% faster
than 0.2.0: faster subgroup-matrix GEMM tiles (≈1.95 instead of ≈1.75 TFLOP/s
f16 for large M, see the backend README for per-shape GFLOP/s), a buffer pool
that makes every bind group a cache hit, an earlier first submit, and faster
bool uploads and RoPE.

**Python laya-mlx reference:** a full `predict()` on the same machine (L=93,
B=1, 30 runs after warmup) has a P50 of **46.2 ms**. The JS MLX `predict()`
adds about 1 ms of prompt building and result formatting on top of the 39.5 ms
forward pass. The native path is therefore at parity with Python: both call the
same MLX kernels, and graph building over FFI costs about 1.4 ms per pass.

**Why WebGPU is still 1.3–1.7× slower than MLX at L ≥ 64:** at those sizes
the linear layers are about 85% of GPU time (B=16, L=256: 1545 of 1827 ms, at
≈1.85 TFLOP/s; MLX's GEMM runs at ≈3 TFLOP/s, which accounts for ≈600 of the
≈700 ms gap; attention is most of the rest, 169 ms).
- WGSL cannot reach Apple's matrix units directly. Dawn's experimental subgroup
  matrices can: a loop of multiply-accumulates alone runs at 3.2 TFLOP/s on
  the M2, but loading each multiply's 8×8 fragments from workgroup memory, as
  a GEMM must, caps it at 2.0–2.45 TFLOP/s. The f16 GEMM reaches ≈1.95.
- Dawn only offers f16 accumulation for f16 inputs. That would break parity, so
  the kernel converts f16 tiles to f32 in workgroup memory (twice the traffic
  of MLX's half tiles) and accumulates in f32.
- In browsers without the subgroup-matrix feature (and in Deno), the tiled WGSL
  kernel is used instead, at about 1.3 TFLOP/s.
- Host overhead is not the gap: ≈450 dispatches go out in 6 submits per
  forward, encoding (≈2.4 ms) overlaps GPU work, and profiled GPU time matches
  wall time within 2%.

## Snake (multilingual checkpoint, 3 questions per move, B=3)

| runtime | per-decision p50 | notes |
|---|---:|---|
| Node + MLX | 24.3 ms | cold GPU, 150 steps |
| Node + WebGPU (Dawn) | 39.8 ms | cold GPU, 150 steps |
| Chromium + WebGPU (snake-web) | ≈60 ms | measured before the Phase 3 WebGPU work |
| Python laya-mlx | ≈43 ms mean | sustained run, so thermally throttled |

- In sustained 4 × 600-step runs, Node+MLX (25.1 moves/s) and Python (23.0 moves/s)
  are equal within noise.
- Every run had 0 deaths.
- Shield interventions match Python seed for seed.

## Quantized checkpoints

`laya quantize` writes q8 (8-bit symmetric, groups of 64) and q4 (4-bit
affine, groups of 64, least-squares refit) copies of a checkpoint. On MLX and
WebGPU, `load()` keeps them quantized on the device (`quantized: "device"`,
the default): MLX runs `quantized_matmul` on the weights repacked to its
affine layout, WebGPU dequantizes inside its Linear kernels' tile loads. With
`quantized: "dequantize"` (and always on the CPU backend) they are
dequantized to f16 on the host while loading, so on the GPU the model is the
fp16 model with slightly perturbed weights. The format and the kernels are
described in [QUANTIZATION.md](QUANTIZATION.md).

The parity run is the same 16 cases / 63 questions as above, measured against
Python `result_fp16`:
`LAYA_REAL=1 LAYA_REAL_QUANT=q8,q4 LAYA_REAL_QUANT_DIR=<dir> [LAYA_REAL_QUANT_MODE=device|dequantize] npm test -w @johnhenry/laya`.
`<dir>` holds `<model>-q8` and `<model>-q4` written by `laya quantize`. The
test requires q8 to keep every choice and argmax, with max |Δp| ≤ 0.05. q4 is
reported, not asserted. It also asserts that the weights really are
quantized on the device (or not) in the chosen mode.

**Quantized on the device vs dequantized on load** (f16, same checkpoints,
2026-09-24). Every argmax and every q4 flip is identical in both modes; the
largest probability differs by at most 5e-3:

| checkpoint | backend | device: choices / argmax / max \|Δp\| | dequantize-on-load: choices / argmax / max \|Δp\| |
|---|---|---|---|
| english q8 | mlx f16 | 22/22 / 63/63 / 4.28e-2 | 22/22 / 63/63 / 4.28e-2 |
| english q8 | webgpu f16 | 22/22 / 63/63 / 4.65e-2 | 22/22 / 63/63 / 4.65e-2 |
| english q4 | mlx f16 | 21/22 / 58/63 / 0.432 | 21/22 / 58/63 / 0.432 |
| english q4 | webgpu f16 | 21/22 / 58/63 / 0.430 | 21/22 / 58/63 / 0.430 |
| multilingual q8 | mlx f16 | 22/22 / 63/63 / 4.10e-2 | 22/22 / 63/63 / 3.77e-2 |
| multilingual q8 | webgpu f16 | 22/22 / 63/63 / 4.05e-2 | 22/22 / 63/63 / 4.05e-2 |
| multilingual q4 | mlx f16 | 21/22 / 62/63 / 0.598 | 21/22 / 62/63 / 0.597 |
| multilingual q4 | webgpu f16 | 21/22 / 62/63 / 0.600 | 21/22 / 62/63 / 0.600 |

The q4 flips are the ones listed below in both modes (English: hi q2, ja q0,
ja q1, ru q1, empty_state q1; multilingual: empty_state q0). In f32 the two
modes agree to 4 decimals on every probability. In f16 the WebGPU kernels
round each dequantized weight to f16 before multiplying (as host
dequantization does); without that, the weights kept in f32 were slightly
*more* accurate per Linear but moved English `hi q2` by 6e-3, to 0.052.

**Device memory, latency, throughput** (M2 MacBook Air, fanless; f16; each
cell in a fresh process after 20 s idle; `packages/laya/bench/quantized.ts`).
Memory is MLX `memory().active` after load (peak: `memory().peak` over the
process) or WebGPU `rt.stats.liveBytes` after load (peak: live + pooled
buffers after the runs). "1 question" is the P50 of `predict` with the first
question of the first fixture case; "16 questions" is one `predict` of 16
questions (one batch of 16).

| checkpoint | backend | weights | device memory after load | peak | load | 1 question P50 | 16 questions |
|---|---|---|---:|---:|---:|---:|---:|
| english | mlx | fp16 | 804 MiB | 1413 MiB | 0.4 s | 39.1 ms | 414 ms (38.7 q/s) |
| english | mlx | q8, dequantize on load | 804 MiB | 1413 MiB | 0.8 s | 39.7 ms | 414 ms (38.7 q/s) |
| english | mlx | **q8 on device** | **441 MiB (55%)** | 1193 MiB | 0.6 s | 33.8 ms | 458 ms (34.9 q/s) |
| english | mlx | q4, dequantize on load | 804 MiB | 1413 MiB | 0.7 s | 40.2 ms | 415 ms (38.5 q/s) |
| english | mlx | **q4 on device** | **228 MiB (28%)** | 1047 MiB | 0.1 s | 34.4 ms | 466 ms (34.3 q/s) |
| english | webgpu | fp16 | 891 MiB | 999 MiB | 0.4 s | 54.6 ms | 654 ms (24.5 q/s) |
| english | webgpu | q8, dequantize on load | 891 MiB | 999 MiB | 0.7 s | 53.4 ms | 636 ms (25.2 q/s) |
| english | webgpu | **q8 on device** | **460 MiB (52%)** | 569 MiB | 0.2 s | **51.3 ms** | **641 ms (25.0 q/s)** |
| english | webgpu | q4, dequantize on load | 891 MiB | 999 MiB | 0.7 s | 53.4 ms | 636 ms (25.2 q/s) |
| english | webgpu | **q4 on device** | **251 MiB (28%)** | 360 MiB | 0.1 s | 54.7 ms | **650 ms (24.6 q/s)** |
| multilingual | mlx | fp16 | 614 MiB | 1338 MiB | 0.8 s | 15.9 ms | 151 ms (106.1 q/s) |
| multilingual | mlx | q8, dequantize on load | 614 MiB | 1338 MiB | 1.1 s | 16.1 ms | 152 ms (105.5 q/s) |
| multilingual | mlx | **q8 on device** | **338 MiB (55%)** | 1124 MiB | 1.1 s | 16.1 ms | 168 ms (95.4 q/s) |
| multilingual | mlx | q4, dequantize on load | 614 MiB | 1338 MiB | 1.2 s | 16.1 ms | 152 ms (105.5 q/s) |
| multilingual | mlx | **q4 on device** | **175 MiB (28%)** | 950 MiB | 0.6 s | 15.9 ms | 172 ms (93.2 q/s) |
| multilingual | webgpu | fp16 | 635 MiB | 713 MiB | 0.8 s | 22.8 ms | 239 ms (66.9 q/s) |
| multilingual | webgpu | q8, dequantize on load | 635 MiB | 713 MiB | 1.2 s | 22.8 ms | 239 ms (67.1 q/s) |
| multilingual | webgpu | **q8 on device** | **328 MiB (52%)** | 406 MiB | 0.6 s | **21.4 ms** | **234 ms (68.3 q/s)** |
| multilingual | webgpu | q4, dequantize on load | 635 MiB | 713 MiB | 1.2 s | 22.8 ms | 239 ms (66.9 q/s) |
| multilingual | webgpu | **q4 on device** | **179 MiB (28%)** | 257 MiB | 0.6 s | **21.9 ms** | **237 ms (67.5 q/s)** |

- **Resident weights shrink with the file:** 52–55% of fp16 for q8 and 28%
  for q4, on both backends. The MLX peak falls by 210–390 MiB; the rest of
  its peak is activations and allocator cache of the 16-row batch. WebGPU's
  peak (every buffer it holds) falls to 57% / 36% of fp16.
- **Speed:** on MLX one short question is 1.16× faster on English (33.8 vs
  39.1 ms: batch-1 is memory-bound and `quantized_matmul` reads a quarter to
  half the bytes) and unchanged on multilingual; 16-question batches are
  10–12% slower (dequantization in the GEMM is not free once it is
  compute-bound). On WebGPU quantized weights are now at least as fast as
  fp16: one question 6% faster for q8 (English 51.3 vs 54.6 ms) and 0–4%
  for q4 (English q4 54.7 vs 54.6 ms is a tie), 16-question batches 1–2%
  faster. With backend-webgpu 0.4 they were
  9–16% slower (English q8 59.4 ms / 748 ms, q4 61.7 / 762 ms; multilingual
  q8 24.4 / 274 ms, q4 25.4 / 279 ms, measured the same day with the same
  bench). The WebGPU fp16 and on-device rows were re-measured 2026-09-24
  with backend-webgpu 0.5 (`QUANT_GEMM_DEFAULT`: a matrix-"vector" kernel for small M,
  subgroup-matrix tiles that dequantize after the MMAs for larger M; the
  single question here is M = 93 tokens, the 16 questions M ≈ 1488).
  Per-Linear GFLOP/s before/after are in the
  backend-webgpu README. The dequantize-on-load and MLX rows are from the
  earlier run (those paths did not change).
- **In a browser** (Chromium in the Claude browser pane on the same M2,
  `navigator.gpu`, `shader-f16` and `subgroups` but no subgroup matrices;
  checkpoints served from `localhost` by the web-playground's `serve.ts`; a
  scratch page loading them with `load(url)`, 3 warmups, P50 of 30 single
  questions of 48–51 tokens, median of 5 batches of 16):

  | checkpoint | weights | device memory | 1 question P50 | 16 questions |
  |---|---|---:|---:|---:|
  | english | fp16 | 891 MiB | 34.6 ms | 596 ms (26.9 q/s) |
  | english | q8 on device | 460 MiB | 33.1 ms | 522 ms (30.7 q/s) |
  | english | q4 on device | 251 MiB | 35.7 ms | 545 ms (29.3 q/s) |
  | multilingual | fp16 | 635 MiB | 19.4 ms | 216 ms (74.2 q/s) |
  | multilingual | q8 on device | 328 MiB | 20.0 ms (min 15.8) | 202 ms (79.4 q/s) |
  | multilingual | q4 on device | 179 MiB | 22.0 ms (min 17.1) | 205 ms (77.9 q/s) |

  The English fp16 / q8 and multilingual q8 rows are from the final build;
  the q4 and multilingual fp16 rows from the same session a build earlier
  (same browser-side kernel choice, older unpack code). Single-question
  P50s in the browser pane vary by ±2 ms between runs, so one question is a
  tie (English q8 4% faster in the final run); 16-question batches are
  5–12% faster quantized.
  With 0.4's kernel choice the same page measured English q8 at 42.2 ms and
  1533 ms (without subgroup matrices it fell back to the direct kernel).
  Loading a checkpoint from an http(s) URL only kept the weights quantized
  on the device after a fix in `@johnhenry/laya` (the URL loader dropped the
  `quantized` option and always dequantized).
- **Load time** drops for the device path on MLX/WebGPU (no host
  dequantization): English q4 loads in 0.1 s against 0.7 s dequantizing.

| checkpoint | backend | choices | argmax | max \|Δp\| vs fp16 | worst field |
|---|---|---|---|---|---|
| english q8 | mlx f16 | 22/22 | 63/63 | 4.3e-2 | hi q2 confidence 0.638 vs 0.681 |
| english q8 | webgpu f16 | 22/22 | 63/63 | 4.7e-2 | hi q2 confidence 0.634 vs 0.681 |
| multilingual q8 | mlx f16 | 22/22 | 63/63 | 3.8e-2 | structured score confidence 0.528 vs 0.490 |
| typed-decisions q8 | mlx f16 | 22/22 | 63/63 | 2.2e-2 | mask_literals q1 score 0.890 vs 0.868 |
| english q4 | mlx f16 | 21/22 | 58/63 | 0.43 | hi q2 noul 0.248 vs 0.681 |
| english q4 | webgpu f16 | 21/22 | 58/63 | 0.43 | hi q2 noul 0.250 vs 0.681 |
| multilingual q4 | mlx f16 | 21/22 | 62/63 | 0.60 | hi q0 confidence 0.313 vs 0.910 |
| typed-decisions q4 | mlx f16 | 22/22 | 58/63 | 0.25 | ru q1 score 1.034 vs 0.783 |

**Which q4 argmaxes change** (mlx f16). "Margin" is the fp16 gap between the
top two probabilities, or 2·|noul − 0.5| for noul questions.

| checkpoint | case / question | type | fp16 → q4 | fp16 margin |
|---|---|---|---|---|
| english | hi q2 | noul | true → false | 0.36 |
| english | ja q0 | choice | other → sales | 0.05 |
| english | ja q1 | score | 1 → 0 | 0.01 |
| english | ru q1 | score | 0 → 1 | 0.16 |
| english | empty_state q1 | score | 1 → 2 | 0.05 |
| multilingual | empty_state q0 | choice | other → billing | 0.22 |
| typed-decisions | zh q1, de q1, ja q1 | score | 1 → 2 | 0.19, 0.13, 0.11 |
| typed-decisions | long q1 | score | 2 → 1 | 0.07 |
| typed-decisions | es q2 | noul | false → true | 0.03 |

Most flips are close calls, or non-English text given to the English-only
checkpoints. Two are not: English `hi q2` (margin 0.36) and multilingual
`empty_state q0` (margin 0.22). q4 genuinely changes the model.

**Sizes** of `model.safetensors`, raw and as served compressed. The
compressed columns use `gzip -9` and `brotli -q 9 -w 24`.

| checkpoint | fp16 | q8 | q4 | q8 gzip / brotli | q4 gzip / brotli |
|---|---:|---:|---:|---:|---:|
| english | 842.6 MB (gzip 777.8, br 774.5) | 434.8 MB (51.6%) | 237.5 MB (28.2%) | 413.6 / 411.8 MB | 221.0 / 219.4 MB |
| multilingual | 643.8 MB (gzip 594.2, br 591.7) | 332.2 MB (51.6%) | 181.5 MB (28.2%) | 315.9 / 314.6 MB | 168.8 / 167.4 MB |
| typed-decisions | 842.6 MB (same as english) | 434.8 MB (51.6%) | 237.5 MB (28.2%) | 413.6 / 411.8 MB | 221.0 / 219.4 MB |

- **Transfer compression adds only about 5–8%**, for fp16 and quantized files
  alike. The size win comes from the format.
- **Load time** with `quantized: "dequantize"` is 0.6–1.3 s for q8/q4 on
  MLX, against 0.4 s for fp16; the host dequantization takes the extra time.
  Kept on the device (the default on MLX/WebGPU), q4 loads faster than fp16.
- **Inference speed and GPU memory**: see the device table above.
- **Converter time:** about 10 s for q8 and 30 s for q4 on the M2, with a
  peak of about 1.1 GB.

**What was tried for q4.** Every variant below was measured on English, mlx
f16. Keeping some tensors at q8 (`--q8`) costs 3–7 percentage points of size
and does not rescue the flips:

| variant | size | choices | argmax | max \|Δp\| |
|---|---:|---|---|---|
| q4 (default) | 28.2% | 21/22 | 58/63 | 0.43 |
| q4, group 32 | 31.3% | 21/22 | 56/63 | 0.43 |
| q4, attention at q8 | 35.2% | 22/22 | 57/63 | 0.28 |
| q4, mlp.Wo at q8 | 32.4% | 20/22 | 56/63 | 0.44 |
| q4, embedding at q8 | 31.1% | 21/22 | 57/63 | 0.52 |
| q4, first and last 4 layers at q8 | 33.6% | 22/22 | 59/63 | 0.33 |
| q4, head/scorer kept fp16 (no refit) | 32.7% | 21/22 | 57/63 | 0.41 |

For q8, groups of 64 are needed. Per-row scales lose 2 argmaxes on English
(61/63, max |Δp| 0.14), because a single outlier in a row coarsens the whole
row.

**Recommendation.**
- **q8** is a safe drop-in: half the download and half the device memory,
  identical decisions on this set, and probabilities within 0.05.
- **q4** is a quarter of the download, but it flips 1–5 of 63 argmaxes per
  checkpoint and moves probabilities by up to 0.6. Use it only where a
  smaller download matters more than matching the fp16 model. Validate it on
  your own questions first.

## Reproducing

```bash
# parity (all 3 checkpoints must be in the HF cache; the fixtures record which snapshot)
LAYA_REAL=1 npm test -w @johnhenry/laya
# quantized checkpoints (write them first: laya quantize --model <repo> --bits 8|4 --out <dir>/<model>-q8|q4)
LAYA_REAL=1 LAYA_REAL_QUANT=q8,q4 LAYA_REAL_QUANT_DIR=<dir> LAYA_REAL_BACKENDS=mlx-f16 npm test -w @johnhenry/laya
# quantized: device memory, latency, throughput (fresh process per cell, 20 s cooldowns; ~12 minutes)
QDIR=<dir> node --conditions=source packages/laya/bench/quantized.ts
# quantized GEMM GFLOP/s vs fp16 on WebGPU
MODEL=both node --conditions=source packages/backend-webgpu/bench/quantized-gemm.ts
# latency grid (takes a lock on ~/gpu.lock; about 10 minutes because of cooldowns)
node --conditions=source packages/backend-webgpu/bench/grid.ts
# regenerate golden fixtures from Python
npm run fixtures
```
