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
affine, groups of 64, least-squares refit) copies of a checkpoint. `load()`
dequantizes them to f16 on the host while loading, so on the GPU the model is
the fp16 model with slightly perturbed weights. The format is described in
[QUANTIZATION.md](QUANTIZATION.md).

The parity run is the same 16 cases / 63 questions as above, measured against
Python `result_fp16`:
`LAYA_REAL=1 LAYA_REAL_QUANT=q8,q4 LAYA_REAL_QUANT_DIR=<dir> npm test -w @johnhenry/laya`.
`<dir>` holds `<model>-q8` and `<model>-q4` written by `laya quantize`. The
test requires q8 to keep every choice and argmax, with max |Δp| ≤ 0.05. q4 is
reported, not asserted.

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
- **Load time** is 0.6–1.3 s for q8/q4 on MLX, against 0.4 s for fp16. The
  host dequantization takes the extra time.
- **Inference speed and GPU memory** are the same as fp16, because the
  weights are dequantized on load.
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
- **q8** is a safe drop-in: half the download, identical decisions on this
  set, and probabilities within 0.05.
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
# latency grid (takes a lock on ~/gpu.lock; about 10 minutes because of cooldowns)
node --conditions=source packages/backend-webgpu/bench/grid.ts
# regenerate golden fixtures from Python
npm run fixtures
```
