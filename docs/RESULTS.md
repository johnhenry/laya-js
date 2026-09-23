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
| B=1 MLX (JS) | 20.0 | 22.4 | 23.0 | 39.5 | 41.7 | 80.0 | 150.6 |
| B=1 WebGPU | 15.1 | 25.1 | 39.3 | 57.4 | 73.5 | 139.7 | 278.9 |
| B=3 MLX (JS) | 25.1 | 43.5 | 59.3 | 92.2 | 108.4 | 212.8 | 436.6 |
| B=3 WebGPU | 31.5 | 69.6 | 105.7 | 145.2 | 193.3 | 385.0 | 804.9 |
| B=16 MLX (JS) | 81.7 | 159.5 | 269.6 | 408.2 | 537.2 | 1096 | 2297 |
| B=16 WebGPU | 130.4 | 264.1 | 471.6 | 696.9 | 954.9 | 1980 | 4260 |

**Python laya-mlx reference:** a full `predict()` on the same machine (L=93,
B=1, 30 runs after warmup) has a P50 of **46.2 ms**. The JS MLX `predict()`
adds about 1 ms of prompt building and result formatting on top of the 39.5 ms
forward pass. The native path is therefore at parity with Python: both call the
same MLX kernels, and graph building over FFI costs about 1.4 ms per pass.

**Why WebGPU is 1.5–1.9× slower than MLX at L ≥ 64:** at those sizes the linear
layers are about 85% of GPU time.
- WGSL cannot reach Apple's matrix units directly. Dawn's experimental subgroup
  matrices get f16 GEMMs to about 1.75 TFLOP/s, against about 3 TFLOP/s for
  MLX.
- Dawn only offers f16 accumulation for f16 inputs. That would break parity, so
  the kernel accumulates in f32.
- In browsers without the subgroup-matrix feature (and in Deno), the tiled WGSL
  kernel is used instead, at about 1.3 TFLOP/s.

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

## Reproducing

```bash
# parity (all 3 checkpoints must be in the HF cache; the fixtures record which snapshot)
LAYA_REAL=1 npm test -w @johnhenry/laya
# latency grid (takes a lock on ~/gpu.lock; about 10 minutes because of cooldowns)
node --conditions=source packages/backend-webgpu/bench/grid.ts
# regenerate golden fixtures from Python
npm run fixtures
```
