# Quantized checkpoints (laya-js format, version 1)

laya-js can load Laya checkpoints whose large weight matrices are stored as
8-bit or 4-bit integers. On MLX and WebGPU the weights **stay quantized on the
device**: the int8/int4 values and their per-group scales are uploaded as
they are stored, and the matrix multiplies dequantize inside the kernel
(MLX `quantized_matmul`; WGSL kernels that dequantize while loading each
tile). That cuts device memory as well as the download. The CPU backend (and
`load(..., { quantized: "dequantize" })`) instead **dequantizes on the host
while loading**, to float16 (float32 on the CPU backend), so the model there
is the float model with slightly perturbed weights.

## Making one

```bash
laya quantize --model aac6fef/laya-mlx --bits 8 --out ./laya-mlx-q8
laya quantize --model aac6fef/laya-mlx --bits 4 --out ./laya-mlx-q4
laya quantize --model ./my-checkpoint --bits 4 --q8 'attn\.' --out ./my-q4-mixed
```

`--model` is a Hub repo (read through the Hugging Face cache; `--offline`
uses only the cache) or a local checkpoint directory. The output directory is
a complete checkpoint:

- `model.safetensors` (quantized);
- `rl_agent_config.json`, `encoder/config.json` and `tokenizer/*`, copied
  unchanged;
- `LICENSE` and `NOTICE` copied from the source;
- a `README.md` saying the checkpoint is derived from the Apache-2.0 Laya
  checkpoint.

`mlx_config.json`, `manifest.json` and `validation.json` are not copied: they
describe the float file, and the Python laya-mlx runtime cannot read this one.

| flag | default | |
|---|---|---|
| `--bits 8\|4` | required | q8 or q4 |
| `--group-size <n\|row>` | 64 | values per scale along the input dim |
| `--no-embeddings` | off | keep the token embedding in float |
| `--exclude <regex>` | — | keep matching tensors in float (repeatable) |
| `--q8 <regex>` | — | with `--bits 4`: store matching tensors as q8 (mixed precision) |
| `--no-refine` | off | q4: skip the least-squares refit (plain min/max ranges); q8 always uses the range fit |
| `--force` | off | overwrite an existing `model.safetensors` |

Programmatic use: `quantizeCheckpoint(model, opts)` from `@johnhenry/laya-cli`,
or `quantizeSafetensors(lazyFile, opts)` from `@johnhenry/laya`, which is
browser-safe and returns the bytes.

## Loading one

Nothing changes: `load(dirOrRepoOrUrl)` reads the file's `__metadata__`, and
when it finds `laya_quant` it hands the quantized tensors to the model:

| `load` option | backend | what happens |
|---|---|---|
| `quantized: "device"` (default) | MLX, WebGPU | Linear weights and the token embedding stay int8/int4 on the device ([below](#on-the-device)) |
| `quantized: "device"` (default) | CPU (no quantized kernels) | same as `"dequantize"` |
| `quantized: "dequantize"` | any | each tensor is dequantized on the host as the model asks for it, then uploaded as float |

`agent.model.quantizedOnDevice` says which one happened. The quantized bytes
(about half or a quarter of the float file) stay in host memory until loading
finishes; when dequantizing, at most one dequantized tensor exists on the host
at a time, because the backend upload follows immediately. This works in
Node, Bun and browsers.

`load()` accepts:
- a local directory;
- a Hub repo id;
- an http(s) base URL. Node and Bun read the weights over HTTP Range requests,
  just as browsers do.

## Hosting

Any static host that serves the directory works, as long as it answers
`Range` requests for `model.safetensors` (every CDN, S3/R2/GCS, `npx serve`,
GitHub Pages; if a server ignores `Range`, the file is downloaded once in
full). Examples:
- `load("https://example.com/models/laya-mlx-q4/")`;
- in a web app, `load("/models/laya-mlx-q4/")`.

To publish on Hugging Face, upload the output directory as a new model repo,
for example with `huggingface-cli upload <you>/laya-mlx-q4 ./laya-mlx-q4`.
Then `load("<you>/laya-mlx-q4")` works through the Hub cache. The quantized
files are **not** published by this repository; publishing them is up to the
checkpoint owner.

Transfer compression barely helps: quantized weights are nearly
incompressible (see [RESULTS.md](RESULTS.md#quantized-checkpoints)). The size
win comes from the format itself.

## Format

The file is a standard safetensors file, so any safetensors reader can list
and read its tensors. Take a quantized matrix `W` with shape [rows, cols],
row-major, where cols is the input dimension of the Linear layer (or the
hidden size, for the embedding). Each row is split into groups of `g`
consecutive values (`g` = cols when `group_size` is `"row"`). Every group has
its own float16 parameters:

| scheme | `W` | `W.scales` | `W.biases` | value |
|---|---|---|---|---|
| q8 (symmetric) | `I8` [rows, cols], q ∈ [−127, 127] | `F16` [rows, cols/g] | — | w = q · scale |
| q4 (affine) | `U8` [rows, cols/2]: column 2k in the low nibble, 2k+1 in the high nibble; q ∈ [0, 15] | `F16` [rows, cols/g] | `F16` [rows, cols/g] | w = q · scale + bias |

Each quantized tensor's width follows its dtype (`I8` = q8, `U8` = q4), so a
q4 file may contain q8 tensors (`--q8`).

**`__metadata__` (all strings):**

| key | value |
|---|---|
| `laya_quant` | `"q8"` or `"q4"`: the checkpoint's scheme. Its presence is what marks the file as quantized. |
| `group_size` | `"64"` (or another integer, or `"row"`). For information only: the loader derives each tensor's group size from the shape of its `.scales`. |
| `version` | `"1"`. Loaders must refuse any other version. |
| `laya_quant_source_dtype` | the float dtype of the source tensors (`"F16"` for the published checkpoints) |
| `laya_quant_source` | the `--model` the file was made from |

**What gets quantized.** A tensor is quantized when all of these hold:
- it is a 2-D float `*.weight`;
- it has at least 64 rows;
- the group size divides its input dimension.

In practice that covers:
- the encoder's `attn.Wqkv`, `attn.Wo`, `mlp.Wi`, `mlp.Wo` and
  `embeddings.tok_embeddings`;
- the decision head's `self_attn.in_proj`, `self_attn.out_proj`, `linear1`
  and `linear2`;
- `scorer.layers.1`.

`act_head.layers.0` ([256, D+4]) is not divisible by 64, so it stays float.
So do norms, biases, `type_emb`, the one-row scorer output and `temperature`.
In the published checkpoints, the quantized tensors hold 99.9% of the bytes.

**How the parameters are fitted.** q8 uses the range fit, scale = max|w| / 127, so every value, outliers included, is within scale/2.

q4 starts from bias = min, scale = (max − min) / 15. The converter then
alternates two steps, up to 8 rounds:
1. assign q by rounding;
2. refit scale and bias by least squares.

It keeps whichever parameters give the lowest squared error, so refinement
never makes a group worse; it lowers the relative RMS weight error by about
7%. The same refit applied to q8 clips outliers. That gave a lower RMS error
but a larger worst-case probability change end to end (0.055 against 0.043
on English), so q8 does not use it.

Scale and bias are rounded to float16 *before* the q values are chosen, so
the stored parameters are exactly the ones the stored q values were fitted
to.

**Dequantization.** The loader computes `fl(q · scale + bias)`, rounded once
from double to the target dtype. For q4 it uses a 16-entry lookup table per
group. Dequantizing the whole 842 MB English checkpoint takes about 1 s in
Node on an M2.

## On the device

The tensor-backend contract (0.3) has an optional trio of ops, called
through its `compose.ts` helpers:

- `uploadQuantized(backend, hostQuantized, dtype)` → a `QuantizedTensor`;
- `quantizedLinear(backend, x, q, bias?)` = x · dequant(W)ᵀ (+ bias), f32
  accumulation, result in x's dtype;
- `quantizedEmbedding(backend, q, ids)` = rows of dequant(W), in `dtype`.

`HostQuantized` is exactly the file layout above (bytes + scales + biases,
`mode: "symmetric"` for q8 and `"affine"` for q4), so nothing is converted on
the host. A backend without the native ops (CPU) gets a default composition
that stores the integer values as floats and dequantizes on the device per
call: correct everywhere, but no memory saving, which is why `load` falls
back to host dequantization there.

**MLX** (`mlx_quantized_matmul`, `mlx_dequantize`; the same mlx-c signature
in both ABIs backend-mlx supports). Read as little-endian u32 words, the laya
bytes already are MLX's packing (value j of a row at bit (j mod 32/bits)·bits
of word ⌊j·bits/32⌋), so affine q4 uploads as-is as a uint32 array. MLX only
has affine quantization, so symmetric q8 is repacked, never dequantized:
q_u = q + 128 (an XOR of every byte with 0x80), bias = −128·scale, both
exact. The repacked weights dequantize bit-for-bit to fl32(q·scale + bias)
(checked in `backend-mlx/test/quantized.test.ts`). MLX supports groups of
32, 64 and 128 with no partial last group; other configurations fall back to
the default composition.

**WebGPU**: the packed words go into a `u32` storage buffer and the scales /
biases into f16 buffers (f32 without `shader-f16`). The kernel depends on
M = B·L (`QUANT_GEMM_DEFAULT` in backend-webgpu; tuned on an M2):

- M ≤ 48, and M ≤ 64 when N < 2048 (every M without subgroup matrices, e.g.
  in browsers without the flag): **qmv**, a memory-bound matrix-"vector"
  kernel. 8 threads walk each weight row with one `vec2<u32>` (q8) / `u32`
  (q4) load per 8 values (16 values per load for the smallest M), unpack
  them in registers and apply the group's scale (and bias) once per step;
  each thread keeps up to 8 rows of x × 2–4 weight rows of f32 partials,
  summed across the 8 threads in workgroup memory (or with subgroup
  shuffles).
- Otherwise, with subgroup matrices: BM×64 subgroup-matrix tiles (BM 64, 48
  or 32 by row padding and grid size, split-K when the grid is small). The
  B tile's raw words and scale are fetched into registers and dequantized
  when stored to workgroup memory, after the MMAs, so the loads hide behind
  them.
- The skinny, direct and tiled kernels still accept quantized B (through a
  4-value helper) for `tuneGemm` choices and other configs.

Values are unpacked without int→float conversions (a field OR 0x6400 is the
f16 1024 + field). Accumulation is f32 everywhere, as for fp16 weights.
Group sizes must be multiples of 4 (partial last groups are fine; the fast
paths need multiples of 8 or 16). The embedding is a dequantizing gather.

Numerically the on-device path matches host dequantization: with f16
activations every WebGPU kernel multiplies by exactly fl16(q·scale + bias),
the weights host dequantization uploads (qmv computes that product in f16
arithmetic, a correctly rounded multiply or fma; the tile kernels in f32,
then round), and keeps f32 weights otherwise; `backend-webgpu`'s tests check
this weight by weight, subnormals included, and that random products round
like the exact sum. MLX computes in its own order. On the parity set every
argmax and every q4 flip is identical in both modes and the probabilities
differ by at most 5e-3; in f32 they agree to 4 decimals.

**What it buys** (M2, f16; [RESULTS.md](RESULTS.md#quantized-checkpoints)):
device memory after load is 52–55% of fp16 for q8 and 28% for q4 (English on
MLX: 804 → 441 / 228 MiB). One short question is 1.16× faster on MLX
(batch 1 is memory-bound); 16-question batches are 10–16% slower there, since
dequantizing inside the GEMM costs ALU time once the multiply is
compute-bound. On WebGPU (backend-webgpu 0.5) quantized weights are as fast
as fp16 or faster: one question 6% faster for q8 and 0–4% for q4,
16-question batches 1–2% faster (0.4 was 9–16% slower). Individual Linears
still lose 3–17% at a few shapes around M = 33 and M = 93–128 (see the
backend-webgpu README).
