# @johnhenry/backend-mlx

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Fbackend-mlx.svg)](https://www.npmjs.com/package/@johnhenry/backend-mlx)

Native Apple Silicon backend for
[`@johnhenry/tensor-backend`](../tensor-backend). It calls Apple's
[mlx-c](https://github.com/ml-explore/mlx-c), the C API over
[MLX](https://github.com/ml-explore/mlx), through FFI. There is no C or C++
of our own. Bun uses the built-in `bun:ffi`, Deno 2 the built-in
`Deno.dlopen`, and Node 24+ the prebuilt [`koffi`](https://koffi.dev) addon. MLX runs on the Metal GPU, which is the
default, or on the CPU.

```ts
import { createMlxBackend } from "@johnhenry/backend-mlx";

const mlx = createMlxBackend(); // { device: "gpu" } by default
const x = await mlx.fromHost({ dtype: "f16", shape: [2, 3], data: new Float16Array([1, 2, 3, 4, 5, 6]) });
const y = mlx.scope(() => mlx.softmax(mlx.scale(x, 2), -1)); // lazy graph; intermediates freed
console.log(await mlx.read(y)); // evaluates on the GPU, then copies to the host
mlx.dispose(y);
mlx.dispose(x);
```

## Install

```bash
npm install @johnhenry/backend-mlx
bun add @johnhenry/backend-mlx
deno add jsr:@johnhenry/backend-mlx npm:@johnhenry/backend-mlx-darwin-arm64
```

- **macOS on Apple Silicon** (darwin/arm64) with Metal. Tested on macOS 27
  (Apple M2), Node 24.9 and Bun 1.2.17. On any other platform the
  package still installs, `createMlxBackend` throws, and its tests skip.
- **Native libraries come with the install.** npm adds the optional
  dependency [`@johnhenry/backend-mlx-darwin-arm64`](../backend-mlx-darwin-arm64)
  on darwin/arm64 only (64 MB download, 207 MB unpacked: `libmlxc.dylib`,
  Apple's `libmlx.dylib`, `libjaccl.dylib` and `mlx.metallib` from MLX
  0.32.2, the same MLX that Python laya-mlx uses). `npm install
  --omit=optional` skips it; then use one of the other sources below.
- **Node** loads the libraries through [`koffi`](https://koffi.dev), also an
  optional dependency (a prebuilt N-API addon; no compiler, no install
  script). **Bun** uses the built-in `bun:ffi`.
- **Deno 2** uses the built-in `Deno.dlopen` and needs `--allow-ffi` (plus
  `--allow-read` and `--allow-env` to locate the library). From JSR, add the
  platform package yourself (`deno add npm:@johnhenry/backend-mlx-darwin-arm64`):
  Deno finds it in `node_modules` or, without one, in its npm cache. Tested
  on Deno 2.9.7: the conformance suite (f32/f16/bf16 including the numerics
  cases, GPU and CPU devices) and `@johnhenry/laya` parity on the three
  published checkpoints (bit-identical to Python, as on Node/Bun).

`libmlxc.dylib` is resolved at runtime, first match wins:

1. `createMlxBackend({ libPath })`.
2. `$LAYA_MLXC_PATH`: the dylib, or a directory containing it
   (`$LAYA_MLXC_LIB` is a legacy alias). If it is set and the path does not
   exist, `createMlxBackend` throws instead of falling back.
3. The platform package `@johnhenry/backend-mlx-darwin-arm64` (`lib/`). Deno
   looks in `node_modules` (next to the module, then the working directory)
   and then in its npm cache.
4. A local build in this package, `prebuilds/darwin-arm64/` (skipped when
   the module was loaded over https, e.g. from JSR):
   `npm run build:mlxc -w @johnhenry/backend-mlx` compiles mlx-c (commit
   `d4afaec`, "Support MLX v0.32.2") against the `mlx` 0.32.2 Python wheel in
   about 10 s. It needs `cmake`, a macOS SDK that links (the script probes;
   override with `SDKROOT`) and `pip install mlx==0.32.2` (or
   `MLX_PY_DIR=…/site-packages/mlx`). `scripts/build-mlxc.sh` ships in the
   tarball.
5. `@nielspeter/mlx-ts-darwin-arm64` (npm; MLX 0.32.1 with the older mlx-c
   ABI).
6. Homebrew: `brew install mlx-c`.

`libCandidates()` lists every path that was tried, and `backend.info`
reports which library loaded and which ABI it has. At load time the backend
detects the two mlx-c ABIs that differ in the signatures it uses (sdpa
`force_fused`, compile-cache API, `cumsum` → `cumsum_axis` with an optional dtype).

## API

- `createMlxBackend(opts?): MlxBackend`
  - `device: "gpu" | "cpu"`. The default is `"gpu"`.
  - `libPath`: an explicit path to `libmlxc.dylib`.
  - `finalizers`. Default `true`. A `FinalizationRegistry` frees handles
    you leaked after garbage collection. It is a safety net only; use
    `dispose` and `scope` to free handles deterministically.
  - `compiledGelu`. Default `true`. On the GPU, `gelu` is an
    `mlx_compile`d shapeless kernel, like `mlx.nn.gelu`.
- The full `Backend` interface is implemented, including the optional
  `flush`, `destroy`, `geglu`, `meanPool`, `compile` and every
  general-numerics op, each one mlx-c call (`mlx_equal`, `mlx_less`, …,
  `mlx_logical_and`, `mlx_sqrt`, `mlx_rsqrt`, `mlx_power`, `mlx_negative`,
  `mlx_abs`, `mlx_tanh`, `mlx_sigmoid`, `mlx_erf`, `mlx_argmax_axis`,
  `mlx_argmin_axis`, `mlx_mean_axis`, `mlx_min_axis`, `mlx_cumsum_axis`).
  MLX's uint32 arg-reduction results are cast to i32, integer inputs of
  float-valued ops (and of `pow`) are cast to f32 first, and `cumsum` of
  bool runs on i32, as the contract specifies. Extras:
  `readSync(t)`, `memory()` (MLX active and peak bytes), `liveTensors()`
  and `info`.
- **Quantized weights** (tensor-backend 0.3's optional trio, native):
  `fromHostQuantized` keeps a laya-js q8/q4 matrix packed in MLX's own
  affine layout (uint32 words + scales + biases in the compute dtype),
  `quantizedLinear` is `mlx_quantized_matmul` (transposed), and
  `quantizedEmbedding` gathers packed rows and runs `mlx_dequantize`. The
  laya bytes read as little-endian u32 words already are MLX's packing, so
  q4 (affine) uploads as-is; symmetric q8 becomes affine by flipping each
  byte's sign bit (q + 128) with bias = −128·scale — a repack, never a
  dequantization, and bit-exact: the device weights dequantize to exactly
  fl32(q·scale + bias). Groups of 32, 64 or 128 without a partial last
  group; anything else resolves to null and runs through the default
  composition. Use it through `uploadQuantized` / `quantizedLinear` /
  `quantizedEmbedding` from `@johnhenry/tensor-backend`.
- `libCandidates()`, `resolveLib()`, `mlxPlatformSupported()` and
  `PLATFORM_PACKAGE` (the platform package's npm name).

## Behaviour

- **Lazy graph.** Every op makes one FFI call that appends an MLX graph node
  and returns immediately. Shapes and dtypes are queried from MLX lazily
  and cached. `flush(...ts)` evaluates (`mlx_eval`), and `flush()` with no
  arguments synchronizes the stream. `read` evaluates, makes the array
  row-contiguous and copies it into a new typed array. For the dtype
  mapping, see `HostData` in tensor-backend.
- **dtypes.** `supports()` is true for f32, f16, bf16, i32 and bool. Ops run
  in the input dtype, with MLX's type promotion. `softmax` uses
  `precise=true` (f32 accumulation). `scale` and `relu` use a scalar of the
  input's dtype, so f16 stays f16, as with MLX's weakly typed Python
  scalars.
- **Fused kernels.**
  - `layerNorm` uses `mlx.fast.layer_norm`.
  - `rope` uses `mlx.fast.rope` with `traditional=false`, `dims = Dh`,
    `scale = 1` and `offset = 0`.
  - `sdpa` uses `mlx.fast.scaled_dot_product_attention` with an array bool
    mask.
  - `linear` uses `addmm(b, x, wᵀ)` or `x @ wᵀ`. wᵀ is a strided view, so
    `linear` matches `mlx.nn.Linear`.
  - `gelu` uses `erf`, like `mlx.nn.gelu`.
  - `embedding` uses `take`, and `gatherRows` uses `take_along_axis`.
- **compile.** The backend wraps your function in an `mlx_closure` whose C
  function is a trampoline back into JS, then applies `mlx_compile`.
  - Tracing happens synchronously inside the first call for each input
    signature.
  - Outputs can be a tensor, an array of tensors or an object of tensors.
  - An exception thrown during tracing propagates to the caller.
  - On `device: "cpu"`, `compile` returns `fn` unchanged, because MLX's CPU
    compile path JIT-builds C++ with the host toolchain.
  - At process exit, the MLX compile cache is cleared, as Python MLX does.
    Without this, static teardown segfaults.
- **Memory.** `dispose` frees the `mlx_array` handle and is idempotent. MLX
  reference-counts buffers, so a pending graph keeps its inputs alive.
  Using a disposed tensor throws. `scope` frees everything created inside
  it except the tensors it returns (directly, or one level deep in an array
  or object). It frees everything if `fn` throws. The tests check that
  `liveTensors()` and MLX active memory stay flat across 50 iterations.
- **Errors.** mlx-c's default error handler calls `exit()`. The backend
  replaces it, so failures become JS exceptions such as `backend-mlx add:
  Shapes (2) and (3) cannot be broadcast.`, and the backend stays usable.
- **Copies.**
  - `fromHost` makes exactly one copy, from the JS buffer into an MLX
    unified-memory buffer (`mlx_array_new_data`), when it is called; the
    returned Promise is already settled, so awaiting a batch of uploads costs
    one microtask. For fp16 safetensors,
    that means one disk read into a `Buffer`, a zero-copy `Float16Array`
    view and one copy into MLX.
  - `read` makes one copy out (`memcpy` into the result array).
  - Zero-copy wrapping of JS memory (`mlx_array_new_data_managed`) is not
    used. It needs GC-pinned, page-aligned memory and a native destructor
    callback, and uploading the whole 421M English checkpoint takes only
    120–300 ms.

## Performance

**Preliminary** (Apple M2, macOS 27; the development machine, which the
binding-decision doc originally mislabelled as an M3 Max), on a machine shared with other
jobs, so expect about ±30% run-to-run noise. A separate benchmark document
will supersede these numbers. The scripts and their Python
twins are in `bench/`. The analysis is in
[`docs/mlx-binding-decision.md`](../../docs/mlx-binding-decision.md).

| | Node (koffi) | Bun (bun:ffi) | Python MLX 0.32.2 |
|---|---:|---:|---:|
| op dispatch (graph node + dispose) | 0.80 µs | 0.55 µs | 0.40 µs |
| linear f16 [16,128,1024]·[1024,1024]ᵀ | 1.50 ms | 1.51 ms | 1.48–1.54 ms |
| 8 stacked ModernBERT-large layers f16, B=1 L=32 | 5.5 ms | 7.3 ms | 9.2 ms |
| English checkpoint, 1 short question (33 tok), f16, P50 | 22–42 ms | 33–42 ms | 32–33 ms |

The rows were measured under the same GPU lock, interleaved.

Deno 2.9.7 (`Deno.dlopen`) dispatches at Node's speed: 0.67 µs per op against
0.67 µs on Node and 0.50 µs on Bun in one interleaved run of
`bench/ops.bench.ts`, and the same linear f16 time (1.52 ms).

The Laya English model (421M, `@johnhenry/laya`) runs on this backend and
is compared with the laya-mlx fp32 fixture over 63 questions:

- f32: 63/63 argmax agreement, maximum |Δlogit| 1.3e-5.
- f16: 63/63 argmax agreement, maximum |Δlogit| 4.0e-2.

Graph build for the whole model takes about 1.5 ms of JS/FFI time. The rest
is GPU time in the same libmlx that Python uses.

## Limitations

- macOS/arm64 only.
- The prebuilt bundle exists for darwin/arm64 only (there is no MLX for
  other platforms). It pins MLX 0.32.2; a different MLX needs a local build
  or `$LAYA_MLXC_PATH`.
- mlx-c is 0.x and its C signatures change. About 60 symbols are bound by
  hand and two ABI variants are detected; an unknown Homebrew mlx-c could
  misalign arguments. Prefer the platform package.
- There is no zero-copy weight upload (one copy per tensor).
- Quantized weights: MLX's `quantized_matmul` computes each group as
  scale·Σx·q + bias·Σx, so symmetric weights (run as affine with
  bias = −128·scale) lose a little to cancellation: ≈1e-6 of Σ|x·w|, or up
  to ≈1e-4 of a small output. Group sizes other than 32/64/128 and partial
  groups fall back to the composition (no memory saving).
- `compile` is the identity on `device: "cpu"`, so the CPU device also has
  no fused GELU.
- Compiled functions must be pure in their tensor arguments. Captured
  tensors become constants, as in Python MLX.
- One MLX stream per backend, which is the device's default stream. `read`
  evaluates synchronously on the calling thread, so its promise is already
  settled when it is returned.

## Family

Part of **[laya-js](https://github.com/johnhenry/laya-js#readme)**, Laya typed decisions in JavaScript on MLX, WebGPU and CPU — see its [package map](https://github.com/johnhenry/laya-js#which-package-do-i-want) and [results](https://github.com/johnhenry/laya-js#results).

- Implements [`@johnhenry/tensor-backend`](https://github.com/johnhenry/laya-js/tree/main/packages/tensor-backend); the native libraries come from [`@johnhenry/backend-mlx-darwin-arm64`](https://github.com/johnhenry/laya-js/tree/main/packages/backend-mlx-darwin-arm64).
- [`@johnhenry/laya`](https://github.com/johnhenry/laya-js/tree/main/packages/laya) selects it first under `backend: "auto"` on Apple Silicon (optional peer dependency).
- Parallel to [`@johnhenry/backend-webgpu`](https://github.com/johnhenry/laya-js/tree/main/packages/backend-webgpu) (portable GPU) and [`@johnhenry/backend-cpu`](https://github.com/johnhenry/laya-js/tree/main/packages/backend-cpu) (reference).

## License

MIT. MLX and mlx-c are MIT, © Apple Inc.; their notices ship in the
platform package (`NOTICE`, `lib/licenses/`).
