# @johnhenry/backend-webgpu

WebGPU implementation of [`@johnhenry/tensor-backend`](../tensor-backend):
WGSL kernels with f16 storage and math (f32 accumulation) that run in
browsers (`navigator.gpu`), Deno (built-in WebGPU) and Node/Bun (Dawn,
via the [`webgpu`](https://www.npmjs.com/package/webgpu) package).
Tensors stay on the GPU; only `read` copies back to the host.

```ts
import { createWebGpuBackend } from "@johnhenry/backend-webgpu";

const gpu = await createWebGpuBackend();          // requests adapter + device
console.log(gpu.adapterInfo, gpu.supports("f16")); // e.g. apple / metal-3, true
const x = gpu.fromHost({ dtype: "f32", shape: [2, 3], data: new Float32Array([1, 2, 3, 4, 5, 6]) });
const y = gpu.scope(() => gpu.softmax(gpu.cast(x, "f16"), -1)); // intermediates freed
console.log(await gpu.read(y));                    // { dtype: "f16", data: Float16Array }
gpu.destroy();
```

## API

- `createWebGpuBackend(opts?): Promise<WebGpuBackend>` rejects when no
  adapter is available. Options:
  - `device?: GPUDevice`: use your device. The backend never destroys a
    device it didn't create.
  - `preferF16 = true`: request `shader-f16` and store/compute `"f16"` natively.
  - `powerPreference = "high-performance"`.
  - `maxBatch = 128`: dispatches per command buffer before an automatic submit.
  - `maxPooledBytes = 1 GiB`: idle bytes kept in the buffer pool.
  - `gemm?: GemmConfig`: GEMM tile configs (see `GEMM_DEFAULT` and `bench/gemm.ts`).
  - `profiling = false`: also request `timestamp-query` so
    `backend.rt.startProfiling()` / `await backend.rt.stopProfiling()` report
    GPU time per kernel.
- `isWebGpuAvailable(): Promise<boolean>`: use it to skip tests.
- `WebGpuBackend` implements every required op plus `geglu`, `meanPool`,
  `flush` and `destroy`. Extras:
  - `adapterInfo`: vendor, architecture, device, description, source, features, limits.
  - `sync()`: wait for submitted work.
  - `rt.stats`: live/pooled bytes, buffers created, dispatches, submits, pipelines.
  - `rt.trim()`: free pooled buffers.
- `WebGpuTensor` has `shape`, `dtype`, `storage` (a refcounted `GPUBuffer`),
  `offset` (element offset: views share storage) and `disposed`.
- `getGpu()` / `requestAdapter()` give low-level access. In Node/Bun,
  `getGpu()` calls the `webgpu` package's `create([])`, and installs its
  `globals` only if `GPUBufferUsage` is missing.

## Behaviour

**Storage per dtype**

| dtype | GPU storage | notes |
|---|---|---|
| `f32` | `f32` | |
| `f16` | `f16` when `shader-f16` is enabled, else `f32` | `supports("f16")` is true only with native f16. |
| `bf16` | `f32` | Converted on upload. Every op producing bf16 rounds its output to bf16 precision (round-to-nearest-even). `read` returns raw bf16 bits (`Uint16Array`). |
| `i32` | `i32` | |
| `bool` | `u32` (0/1) | Uses 4 bytes per element. |

**Numerics**
- Every kernel loads storage values, computes in f32 (or i32 for
  all-integer elementwise ops), then rounds once on store.
- matmul/linear, reductions, LayerNorm statistics (two-pass), softmax and
  attention all accumulate in f32.
- GELU is exact-erf GELU. erf uses a least-squares odd polynomial on
  |x| < 1 and the Numerical Recipes erfc form beyond, with max abs error
  ≈1.6e-7 in f32. `test/kernels.test.ts` checks GELU against a
  high-precision erf to 2e-6.
- `rope` is split-half (MLX `traditional=False`). cos/sin tables are cached
  per (L, D, base). Angles emulate MLX's f32 rounding; cos/sin are computed
  in f64.
- `sdpa` masks are bool, broadcast from `[B|1, H|1, Lq, Lk]`, or any lower
  rank right-aligned.

**Execution**
- Ops are synchronous and only enqueue work. All dispatches go into one
  compute pass (WebGPU orders them) and are submitted at `flush`, at `read`,
  or every `maxBatch` dispatches.
- Uniforms are packed into a 64 KiB arena that is written once per submit.
- Buffer pool with ¼-power-of-two size classes (at most 25% waste).
  `fromHost` flushes first only when it reuses a buffer that a pending,
  unsubmitted pass may still read.
- Pipeline cache keyed by (kernel, dtypes, specialization). Explicit bind
  group layouts are used, not `layout: "auto"`.
- Free views: `reshape`, same-dtype `cast`, identity-like `transpose`, and
  contiguous `slice`/`split`, such as along the leading axis. Other shape
  ops run one strided-copy kernel.
- `scope`, `dispose`: the same semantics as the CPU backend. Returned tensors
  (directly or one level deep) move to the enclosing scope. Using a
  disposed tensor throws.
- The first uncaptured device error surfaces as an exception from the next
  `read` or `sync`.

**Kernels**

| Op | Kernel |
|---|---|
| linear, M ≤ 64 (latency path) | Split-K "skinny" kernel: each weight row is read from DRAM once. |
| linear, K % 4 = 0 | Register-blocked "direct" kernel: 4×8 outputs per thread, vec4 loads straight from global memory. |
| linear otherwise; batched and broadcast matmul | 64×64×16 workgroup-memory tiled GEMM. |
| sdpa, head dim 32 or 64 | Flash attention: online softmax, register-blocked 32q×16k tiles. |
| sdpa, other head dims ≤ 256 | Generic flash kernel. |
| layerNorm, softmax (last axis) | One workgroup per row. |
| sort | Bitonic sort in workgroup memory for rows ≤ 4096; slow per-row insertion sort above that. |

Fused `geglu` and `meanPool` are single kernels. Elementwise ops are n-ary
with broadcast strides (rank ≤ 8), and 1-D launches use 2-D grids, so
sizes above 16M elements work.

## Runtime support

| Runtime | Status |
|---|---|
| Node ≥ 24 (Dawn via `webgpu@0.6`) | ✅ Conformance f32 + f16, plus kernel edge tests (`npm test`). |
| Bun 1.2 (same Dawn addon) | ✅ Same suites (`npm run test:bun`). The tests use `bun:test` under Bun, because Bun's `node:test` shim registers only the first file. |
| Deno 2.x (built-in wgpu) | ✅ Same suites (`npm run test:deno`). `navigator.gpu` is used directly and `shader-f16` is available on Apple. |
| Chrome / Edge ≥ 113 (f16 from 120) | ✅ `demo/` passes in f32 + f16 (verified in Chromium 152 on macOS). |
| Safari 26 | Expected to work (WebGPU on by default); not verified. |
| Firefox ≥ 141 | Expected to work where WebGPU is enabled; not verified. Without `shader-f16` it falls back to f32. |

Tests skip cleanly when no adapter is available.

**Browser smoke page.** Run `npm run demo:build -w @johnhenry/backend-webgpu`
(uses `bun build`), serve `packages/` (for example
`python3 -m http.server -d packages 8000`), then open
<http://localhost:8000/backend-webgpu/demo/>. The page runs every
conformance case, then a GEMM benchmark.

## Performance (Apple M2, 10-core GPU, macOS 27, Dawn/Metal)

**GEMM** (`npm run bench -w @johnhenry/backend-webgpu`)

| Case | f32 | f16 |
|---|---:|---:|
| linear [2048,1024]·[3072,1024]ᵀ | 1.20 TFLOP/s | 1.32 TFLOP/s |
| same shape, MLX `x @ w.T` | 2.32 | 3.06 |
| linear [33,1024]·[3072,1024]ᵀ (skinny) | 0.32 ms | 0.19 ms |
| batched matmul [16,128,1024]·[16,1024,128] | 0.67 TFLOP/s | 0.75 TFLOP/s |

The large Linear reaches ≈40% of the measured FMA peak (2.85 TFLOP/s). The
rest of the gap to MLX is mostly simdgroup-matrix hardware, which WGSL
can't reach yet. Chromium 152 gives the same numbers. In Deno (wgpu) the
large GEMM matches, but small dispatches cost more.

**Laya English checkpoint (ModernBERT-large, 28 layers, fp16 weights), one
question, L = 33, B = 1** (`bench/laya-parity.ts --real`)

| | WebGPU (this) | MLX on the same M2 |
|---|---:|---:|
| f16 forward, median | 25.9 ms | 23.0 ms |
| f32 forward, median | 43.1 ms | 34.6 ms |

- Parity on all 63 English fixture questions: argmax 63/63 in both dtypes.
  Max |Δlogit| vs MLX fp32 is 7e-5 (f32) and 5e-2 (f16).
- Tiny checkpoint: every stage matches within 1.2e-6 (f32).
- `bench/profile.ts` shows where GPU time goes (the Linears take about 80%)
  and the CPU encode cost (about 13 µs per dispatch, 450 dispatches per
  forward).

## Limitations

- Every op is its own dispatch. Encoding costs about 10–15 µs of CPU per
  dispatch (bind group plus pass commands), about 6 ms per ModernBERT-large
  forward. WebGPU has no reusable compute command buffers, and there is no
  graph fusion (`compile` isn't implemented).
- No subgroup or simdgroup-matrix paths yet. Kernel configs are tuned on
  Apple M2 and are untested on discrete or mobile GPUs.
- `sort` rows over 4096 elements use a slow per-row insertion sort.
  `matmul` supports batch ≤ 65535. Rank ≤ 8. `sdpa` head dim ≤ 256, with q,
  k and v having the same batch and heads (no GQA). Fully masked rows are
  undefined behaviour, as in the contract.
- `sdpa` accepts bool masks only. A float mask is cast to bool (nonzero means attend), so additive 0/−∞ masks are *not* supported.
- Bool tensors take 4 bytes per element, and bf16 takes f32 memory.
- Node/Bun need the platform's prebuilt Dawn addon (`webgpu` package:
  darwin universal, linux x64/arm64, win32 x64/arm64).
