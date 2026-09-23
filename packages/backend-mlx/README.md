# @johnhenry/backend-mlx

Native Apple Silicon backend for
[`@johnhenry/tensor-backend`](../tensor-backend). It calls Apple's
[mlx-c](https://github.com/ml-explore/mlx-c), the C API over
[MLX](https://github.com/ml-explore/mlx), through FFI. There is no C or C++
of our own. Bun uses the built-in `bun:ffi`, and Node 24+ uses the prebuilt
[`koffi`](https://koffi.dev) addon. MLX runs on the Metal GPU, which is the
default, or on the CPU.

```ts
import { createMlxBackend } from "@johnhenry/backend-mlx";

const mlx = createMlxBackend(); // { device: "gpu" } by default
const x = mlx.fromHost({ dtype: "f16", shape: [2, 3], data: new Float16Array([1, 2, 3, 4, 5, 6]) });
const y = mlx.scope(() => mlx.softmax(mlx.scale(x, 2), -1)); // lazy graph; intermediates freed
console.log(await mlx.read(y)); // evaluates on the GPU, then copies to the host
mlx.dispose(y);
mlx.dispose(x);
```

## Install requirements

- **macOS on Apple Silicon** (darwin/arm64) with Metal. Tested on macOS 27
  with an M3 Max, Node 24.9 and Bun 1.2.17. On any other platform,
  `createMlxBackend` throws and this package's tests skip.
- **Node:** `koffi` 3.x must be installed. Bun needs nothing extra.
- **libmlxc.dylib**, which sits next to `libmlx.dylib` and `mlx.metallib`.
  The first path that exists wins:
  1. `createMlxBackend({ libPath })`
  2. `$LAYA_MLXC_LIB`
  3. `prebuilds/darwin-arm64/` in this package. `npm run build:mlxc -w
     @johnhenry/backend-mlx` builds this bundle in about 10 s. It compiles
     mlx-c (commit `d4afaec`, "Support MLX v0.32.2") against the `mlx`
     0.32.2 Python wheel, the same MLX that laya-mlx uses. This needs
     `cmake`, a working macOS SDK and `pip install mlx==0.32.2` (or
     `MLX_PY_DIR=…/site-packages/mlx`). The script finds an SDK that links.
     Some Command Line Tools SDKs ship `.tbd` stubs that the linker rejects;
     you can override the choice with `SDKROOT`.
  4. `@nielspeter/mlx-ts-darwin-arm64` (npm). This uses MLX 0.32.1 with the
     older mlx-c ABI.
  5. Homebrew: `brew install mlx-c`.

  `libCandidates()` lists every path that was tried, and `backend.info`
  reports which library loaded and which ABI it has. At load time the
  backend detects the two mlx-c ABIs that differ in the signatures it uses
  (sdpa `force_fused`, compile-cache API).

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
  `flush`, `destroy`, `geglu`, `meanPool` and `compile`. Extras:
  `readSync(t)`, `memory()` (MLX active and peak bytes), `liveTensors()`
  and `info`.
- `libCandidates()`, `resolveLib()` and `mlxPlatformSupported()`.

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
    unified-memory buffer (`mlx_array_new_data`). For fp16 safetensors,
    that means one disk read into a `Buffer`, a zero-copy `Float16Array`
    view and one copy into MLX.
  - `read` makes one copy out (`memcpy` into the result array).
  - Zero-copy wrapping of JS memory (`mlx_array_new_data_managed`) is not
    used. It needs GC-pinned, page-aligned memory and a native destructor
    callback, and uploading the whole 421M English checkpoint takes only
    120–300 ms.

## Performance

Measurements are from an M3 Max on macOS 27, on a machine shared with other
jobs, so expect about ±30% run-to-run noise. The scripts and their Python
twins are in `bench/`. The analysis is in
[`docs/mlx-binding-decision.md`](../../docs/mlx-binding-decision.md).

| | Node (koffi) | Bun (bun:ffi) | Python MLX 0.32.2 |
|---|---:|---:|---:|
| op dispatch (graph node + dispose) | 0.80 µs | 0.55 µs | 0.40 µs |
| linear f16 [16,128,1024]·[1024,1024]ᵀ | 1.50 ms | 1.51 ms | 1.48–1.54 ms |
| 8 stacked ModernBERT-large layers f16, B=1 L=32 | 5.5 ms | 7.3 ms | 9.2 ms |
| English checkpoint, 1 short question (33 tok), f16, P50 | 22–42 ms | 33–42 ms | 32–33 ms |

The rows were measured under the same GPU lock, interleaved.

The Laya English model (421M, `@johnhenry/laya`) runs on this backend and
is compared with the laya-mlx fp32 fixture over 63 questions:

- f32: 63/63 argmax agreement, maximum |Δlogit| 1.3e-5.
- f16: 63/63 argmax agreement, maximum |Δlogit| 4.0e-2.

Graph build for the whole model takes about 1.5 ms of JS/FFI time. The rest
is GPU time in the same libmlx that Python uses.

## Limitations

- macOS/arm64 only.
- A native library bundle is required, and this package ships none yet
  (see Install).
- There is no zero-copy weight upload (one copy per tensor).
- `compile` is the identity on `device: "cpu"`, so the CPU device also has
  no fused GELU.
- Compiled functions must be pure in their tensor arguments. Captured
  tensors become constants, as in Python MLX.
- One MLX stream per backend, which is the device's default stream. `read`
  evaluates synchronously on the calling thread, so its promise is already
  settled when it is returned.
- Deno is untested: there is no `Deno.dlopen` adapter, although mlx-ts shows
  that one works.
