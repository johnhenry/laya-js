# Quantized checkpoints (laya-js format, version 1)

laya-js can load Laya checkpoints whose large weight matrices are stored as
8-bit or 4-bit integers. The weights are **dequantized on the host while
loading** (to float16, or float32 on the CPU backend). This makes the
download smaller. It does not reduce GPU memory or speed up inference: once
loaded, the model is the float model with slightly perturbed weights. No
backend changes are needed; the tensor-backend contract is untouched.

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
when it finds `laya_quant` it dequantizes each quantized tensor as the model
asks for it. The quantized bytes (about half or a quarter of the float file)
stay in memory until loading finishes. At most one dequantized tensor exists
on the host at a time, because the backend upload follows immediately. This
works in Node, Bun and browsers.

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

## Why dequantize-on-load (and what a follow-up would do)

Keeping the weights quantized on the GPU (int8/int4 matmul kernels, as MLX
`quantized_matmul` or WebGPU shaders would do) would also cut GPU memory and
could speed up the memory-bound batch-1 case. That needs a new op in the
tensor-backend contract (for example `quantizedMatmul(x, q, scales, biases,
groupSize, bits)`) and kernels in every backend. The file format above is
already shaped for this: MLX-style affine groups along the input dimension.
Only the packing would need converting at upload (MLX packs 8 nibbles into a
uint32).
