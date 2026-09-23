# @johnhenry/backend-webgpu

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Fbackend-webgpu.svg)](https://www.npmjs.com/package/@johnhenry/backend-webgpu)

WebGPU implementation of [`@johnhenry/tensor-backend`](../tensor-backend):
WGSL kernels with f16 storage and math (f32 accumulation) that run in
browsers (`navigator.gpu`), Deno (built-in WebGPU) and Node/Bun (Dawn,
via the [`webgpu`](https://www.npmjs.com/package/webgpu) package).
Tensors stay on the GPU; only `read` copies back to the host.

## Install

```bash
npm install @johnhenry/backend-webgpu
bun add @johnhenry/backend-webgpu
deno add jsr:@johnhenry/backend-webgpu
```

Browsers use `navigator.gpu`; Deno uses its built-in WebGPU; Node ≥ 24 and Bun ≥ 1.2 use Dawn through the [`webgpu`](https://www.npmjs.com/package/webgpu) package (a dependency, loaded only when `navigator.gpu` is absent). TypeScript: the public types mention WebGPU DOM types (`GPUDevice`); `@webgpu/types` is installed with this package — add it to `compilerOptions.types` if your project has no WebGPU types yet.

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
  - `subgroupMatrix = true`: use subgroup matrices (Metal `simdgroup_matrix`)
    for Linears with M > 64 when the adapter offers
    `chromium-experimental-subgroup-matrix` with an f32 8×8×8 config and a
    fixed subgroup size of 32. In Node/Bun this creates the Dawn instance with
    the `allow_unsafe_apis` toggle (it only unlocks experimental features);
    Chrome exposes the feature with `--enable-unsafe-webgpu`. Without it the
    backend silently uses the portable kernels. `hasSubgroupMatrix` says which.
  - `sleepWhileWaiting` (default: true under Dawn, false for `navigator.gpu`):
    before awaiting a readback, sleep (`setTimeout`) for ~80% of the time the
    same amount of work took last time. Dawn-node resolves `mapAsync` by
    polling in a busy loop (≈100% of a core under Bun, ≈33% under Node);
    this cuts process CPU during a forward by ~4× at the same latency.
- `isWebGpuAvailable(): Promise<boolean>`: use it to skip tests.
- `WebGpuBackend` implements every required op plus `geglu`, `meanPool`,
  `flush` and `destroy`. Extras:
  - `adapterInfo`: vendor, architecture, device, description, source, features, limits.
  - `sync()`: wait for submitted work.
  - `rt.stats`: live/pooled bytes, buffers created, dispatches, submits, pipelines.
  - `rt.trim()`: free pooled buffers.
- `WebGpuTensor` has `shape`, `dtype`, `storage` (a refcounted `GPUBuffer`),
  `offset` (element offset: views share storage) and `disposed`.
- `getGpu({ unsafe? })` / `requestAdapter(powerPreference?, unsafe?)` give
  low-level access. In Node/Bun, `getGpu()` calls the `webgpu` package's
  `create([])` (`unsafe`: with `enable-dawn-features=allow_unsafe_apis`;
  one instance per flag set), and installs its `globals` only if
  `GPUBufferUsage` is missing.
- Bundling: the Dawn loader lives behind the package's `#dawn` import, whose
  `browser` condition maps to a stub, so browser bundles (`bun build
  --target browser`, Vite, esbuild with `platform: "browser"`) never see
  the `webgpu` specifier and need no `external` setting.

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
- Uniforms are packed into a 64 KiB arena that is written once per submit,
  bound with a dynamic offset. Bind groups are cached by (layout, buffers,
  uniform chunk); pooled buffers are reused in the same pattern every
  forward, so a steady-state ModernBERT forward creates no bind groups
  (encode cost ≈ 8 µs per dispatch in Node, was ≈ 13 µs).
- Buffer pool with ¼-power-of-two size classes (at most 25% waste).
  `fromHost` flushes first only when it reuses a buffer that a pending,
  unsubmitted pass may still read.
- Pipeline cache keyed by (kernel, dtypes, specialization). Explicit bind
  group layouts are used, not `layout: "auto"`.
- Free views: `reshape`, same-dtype `cast`, identity-like `transpose`, and
  contiguous `slice`/`split`, such as along the leading axis. Other shape
  ops run one strided-copy kernel, specialized on the rank left after
  dropping unit axes and merging axes contiguous on both sides, with 4
  elements per thread when the inner axis is unit-stride (the attention
  head transposes are rank 3–5; ≈6× faster than the generic rank-8 loop).
- `scope`, `dispose`: the same semantics as the CPU backend. Returned tensors
  (directly or one level deep) move to the enclosing scope. Using a
  disposed tensor throws.
- The first uncaptured device error surfaces as an exception from the next
  `read` or `sync`.

**Kernels**

| Op | Kernel |
|---|---|
| linear, M ≤ 64 (latency path) | Split-K "skinny" kernel: each weight row is read from DRAM once. |
| linear, M > 64, K % 4 = 0, subgroup matrices available | 32×64 tiles, 2 subgroups × 4×4 f32 8×8 fragments; K panels of 8 staged (converted to f32) through workgroup memory, software-pipelined through registers. f32 accumulation. |
| linear, K % 4 = 0 otherwise | Register-blocked "direct" kernel: 4×8 outputs per thread, vec4 loads straight from global memory. |
| linear otherwise; batched and broadcast matmul | 64×64×16 workgroup-memory tiled GEMM. |
| sdpa, head dim 32 or 64 | Flash attention: online softmax, register-blocked 32q×16k tiles. With a mask, each query block first scans its mask rows and only visits the key-tile range that has a visible key (sliding window, padding). |
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
| linear [2048,1024]·[3072,1024]ᵀ, subgroup matrices (default under Dawn) | 1.97 TFLOP/s | 1.75 TFLOP/s |
| same shape, portable kernels (`subgroupMatrix: false`, browsers without the flag) | 1.20 | 1.32 |
| same shape, MLX `x @ w.T` | 2.32 | 3.06 |
| linear [33,1024]·[3072,1024]ᵀ (skinny) | 0.32 ms | 0.19 ms |
| batched matmul [16,128,1024]·[16,1024,128] | 0.67 TFLOP/s | 0.75 TFLOP/s |

The portable kernels reach about 40% of the measured FMA peak (2.85
TFLOP/s). Subgroup matrices get f16 to ≈1.75 TFLOP/s (f16 tiles are
converted to f32 in workgroup memory; f32 needs no conversion and gets
≈2). `bench/gemm-m.ts` sweeps M for the four ModernBERT-large Linear
shapes. MLX's hand-written Metal GEMM reaches ≈3. Chromium 152 gives the same
portable-kernel numbers. In Deno (wgpu) the large GEMM matches, but small
dispatches cost more.

**Laya English checkpoint (ModernBERT-large, 28 layers), f16, median ms per
forward (upload + forward + readback)** — `bench/grid.ts`, synthetic ids,
WebGPU and MLX interleaved per cell in one Node process.

| L → | 16 | 33 | 64 | 93 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B=1 WebGPU | 15.1 | 25.1 | 39.3 | 57.4 | 73.5 | 139.7 | 278.9 |
| B=1 MLX | 20.0 | 22.4 | 23.0 | 39.5 | 41.7 | 80.0 | 150.6 |
| B=3 WebGPU | 31.5 | 69.6 | 105.7 | 145.2 | 193.3 | 385.0 | 804.9 |
| B=3 MLX | 25.1 | 43.5 | 59.3 | 92.2 | 108.4 | 212.8 | 436.6 |
| B=16 WebGPU | 130.4 | 264.1 | 471.6 | 696.9 | 954.9 | 1980.1 | 4259.9 |
| B=16 MLX | 81.7 | 159.5 | 269.6 | 408.2 | 537.2 | 1096.2 | 2296.6 |

Bun gives the same WebGPU numbers (±3%). WebGPU is 0.75–1.9× MLX's time;
for M = B·L > 64 the Linears are ≈85% of GPU time, so the ratio is
MLX's GEMM advantage (see above). Previously (portable kernels only,
uncached bind groups) B=1 L=93 took 80.5 ms, B=3 L=93 201.6 ms and B=16
L=512 18.8 s.

- Thermal note: this is a fanless MacBook Air. Sustained GPU load
  throttles to ≈35% of the cold throughput after ≈10 s and recovers within
  ≈5 s idle, so every cell idles 5 s and measures for ≤ 1 s. Long
  benchmarks (e.g. `laya bench`'s warmups + 50 runs) are measured hot.
- Parity on all 63 fixture questions of each checkpoint (English,
  multilingual, typed-decisions): argmax 63/63 in both dtypes; max |Δp| vs
  Python result_fp16 ≤ 5e-3 (f16), vs result_fp32 ≤ 1e-4 (f32).
- Tiny checkpoint: every stage matches within 1.2e-6 (f32).
- `bench/profile.ts` / `PROFILE=1 bench/grid.ts` show GPU time per kernel;
  `bench/cpu.ts` shows CPU encode cost and process CPU per forward.

## Limitations

- Every op is its own dispatch. Encoding costs about 8 µs of CPU per
  dispatch (pass commands; bind groups are cached), about 3.5 ms per
  ModernBERT-large forward, overlapped with GPU work. WebGPU has no reusable
  compute command buffers, and there is no graph fusion (`compile` isn't
  implemented).
- Subgroup-matrix GEMM needs Dawn's experimental
  `chromium-experimental-subgroup-matrix` (f32 8×8×8, subgroup size 32);
  the WGSL/Dawn path tops out at ≈1.8 TFLOP/s on M2 (MLX's Metal GEMM ≈3),
  even with fragments loaded straight from global memory. Only f16→f16
  accumulation is offered for f16 fragments, so f16 operands are converted
  to f32 in workgroup memory. Kernel configs are tuned on Apple M2 and are
  untested on discrete or mobile GPUs. No other subgroup (shuffle/reduce) paths.
- `sort` rows over 4096 elements use a slow per-row insertion sort.
  `matmul` supports batch ≤ 65535. Rank ≤ 8. `sdpa` head dim ≤ 256, with q,
  k and v having the same batch and heads (no GQA). Fully masked rows are
  undefined behaviour, as in the contract.
- `sdpa` accepts bool masks only. A float mask is cast to bool (nonzero means attend), so additive 0/−∞ masks are *not* supported.
- Bool tensors take 4 bytes per element, and bf16 takes f32 memory.
- Node/Bun need the platform's prebuilt Dawn addon (`webgpu` package:
  darwin universal, linux x64/arm64, win32 x64/arm64).

## Family

Part of **[laya-js](https://github.com/johnhenry/laya-js#readme)**, Laya typed decisions in JavaScript on MLX, WebGPU and CPU — see its [package map](https://github.com/johnhenry/laya-js#which-package-do-i-want) and [results](https://github.com/johnhenry/laya-js#results).

- Implements [`@johnhenry/tensor-backend`](https://github.com/johnhenry/laya-js/tree/main/packages/tensor-backend); [`@johnhenry/laya`](https://github.com/johnhenry/laya-js/tree/main/packages/laya) selects it under `backend: "auto"` when MLX is unavailable (optional peer dependency).
- Sibling in spirit of [`@johnhenry/math-plus-tensor-webgpu`](https://github.com/johnhenry/math-plus/tree/main/packages/tensor-webgpu) (general tensor GEMM/attention in the browser); this one is specialized to the transformer-inference op contract.

## License

Apache-2.0.
