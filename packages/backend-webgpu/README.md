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
const x = await gpu.fromHost({ dtype: "f32", shape: [2, 3], data: new Float32Array([1, 2, 3, 4, 5, 6]) });
const y = gpu.scope(() => gpu.softmax(gpu.cast(x, "f16"), -1)); // intermediates freed
console.log(await gpu.read(y));                    // { dtype: "f16", data: Float16Array }
gpu.destroy();
```

## API

- `createWebGpuBackend(opts?): Promise<WebGpuBackend>` rejects when no
  adapter is available. Options:
  - `device?: GPUDevice`: use your device. The backend never destroys a
    device it didn't create.
  - `adapter?: GPUAdapter`: with `device`, the adapter it came from. The
    subgroup-matrix check needs the adapter's `subgroupMatrixConfigs`, and
    Dawn doesn't mirror them onto `device.adapterInfo`. Without the adapter,
    a device you pass in never uses subgroup matrices.
  - `preferF16 = true`: request `shader-f16` and store/compute `"f16"` natively.
  - `powerPreference = "high-performance"`.
  - `maxBatch = 128`: dispatches per command buffer before an automatic submit.
  - `firstBatch = 24`: dispatches in the first submit after the GPU went idle
    (a completed readback), so the GPU starts while the rest is encoded.
  - `maxPooledBytes = 1 GiB`: idle bytes kept in the buffer pool.
  - `gemm?: GemmConfig`: GEMM tile configs (see `GEMM_DEFAULT`, `bench/gemm.ts`
    and `bench/linear-shapes.ts`). `GEMM_V020` is the 0.2.0 configuration.
  - `gemmTuning?: Record<string, GemmChoice>`: `tuneGemm` results to restore.
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
  - `sleepThresholdMs = 3`: sleep only when the expected wait is longer
    than this. Short readbacks (a few ms) can lose 15–60% latency to the
    sleep; raise the threshold if latency matters more than CPU.
- `isWebGpuAvailable(): Promise<boolean>`: use it to skip tests.
- `WebGpuBackend` implements every required op plus `geglu`, `meanPool`,
  `flush`, `destroy` and the quantized-weight trio of tensor-backend 0.3
  (below). Extras:
  - `adapterInfo`: vendor, architecture, device, description, source, features, limits.
  - `sync()`: wait for submitted work.
  - `tuneGemm(shapes, { dtype?, rounds? })`: measures every applicable
    Linear kernel (skinny, each `gemmConfig.sg` entry, direct) for each
    exact `{ M, N, K }` and records the fastest in `gemmTuning` (a `Map`
    shared by all backends on the same `GPUDevice`; persist it with
    `Object.fromEntries(backend.gemmTuning)`). Roughly 10–50 ms per shape.
    Without it, the built-in rules below pick the kernel.
  - `rt.stats`: live/pooled bytes, buffers created, bind groups created, dispatches, submits, pipelines.
  - `rt.trim()`: free pooled buffers.
- Extension hooks, for code that adds its own kernels to the runtime
  (math-plus's tensor-compile fusion does):
  - `elementwise(expr, xs, { outDtype?, helpers? })`: a custom n-ary
    elementwise kernel with broadcasting. `expr` is an f32 WGSL expression
    over `x0, x1, …`; `helpers` is extra WGSL (functions) placed before the
    entry point. The kernel is one dispatch, cached per expression and
    input layout. For `outDtype: "bool"`, nonzero means true.
  - `empty(shape, dtype)`: an uninitialised, scope-tracked tensor from the
    pool, for a custom kernel's output.
  - `wrapBuffer(buffer, shape, dtype, offset?)`: a view of a `GPUBuffer`
    you own. `dispose` never pools or destroys it.
  - `rt.kernel(() => source, key)` / `rt.dispatch(kernel, buffers, params,
    groups)`: compile and batch any kernel. The runtime generates the header
    from `bindings` (`array<elem>`, `read` or `read_write`) and `params` (a
    uniform struct `P`); entry point `main`. A tensor's data is
    `t.storage.buffer` from element `t.offset`.
  - Exported classes and types: `Runtime`, `Storage`, `KernelSource`,
    `BindingSpec`, `ParamSpec`, `ParamType`, `CompiledKernel` and
    `RuntimeStats`.
- **Quantized weights** (native): `fromHostQuantized` uploads a laya-js
  q8/q4 matrix unchanged — the packed bytes as a `u32` storage buffer, the
  per-group scales / biases as f16 (f32 without `shader-f16`).
  `quantizedLinear` runs the same Linear kernels as fp16 weights (skinny for
  small M, subgroup-matrix with split-K, direct, tiled): their B-operand
  loads go through one WGSL helper that unpacks 4 values from a word
  (sign-extended for symmetric), applies `fma(q, scale, bias)` in f32 and
  stages the tile, so accumulation stays f32. `quantizedEmbedding` is a
  dequantizing gather. Any group size that is a multiple of 4 works,
  including a partial last group; other group sizes resolve to null (the
  default composition). Use them through `uploadQuantized` /
  `quantizedLinear` / `quantizedEmbedding` from `@johnhenry/tensor-backend`.
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
  or every `maxBatch` dispatches (`firstBatch` for the first submit after
  the GPU went idle). A ModernBERT-large forward is ~450 dispatches in
  6 submits (4 automatic, one per `read`).
- Uniforms are packed into a 64 KiB arena that is written once per submit,
  bound with a dynamic offset. Bind groups are cached by (layout, buffers,
  uniform chunk).
- Buffer pool with ¼-power-of-two size classes (at most 25% waste). A
  size class hands out its lowest-id free buffer (not LIFO), so a
  computation that allocates and frees in the same order gets the same
  buffers every time and its bind groups are cache hits: a steady-state
  ModernBERT forward creates none (0.2.0's LIFO pool permuted buffers and
  created ~170 per forward). Encoding costs ≈ 5 µs per dispatch in Node.
- Buffer reuse is ordered: a freed buffer can be handed to the next GPU
  write immediately (the queue orders it after pending reads); `fromHost`
  copies the host data with `queue.writeBuffer` when called and returns an
  already-settled Promise, flushing first only when it reuses a buffer that
  the pending, unsubmitted pass may still read; buffers evicted from a full
  pool are destroyed only after the submit that uses them.
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
| linear, M > 64, K % 4 = 0, subgroup matrices available | `tuneGemm` choice if any, else: 64×64 tiles, 2 subgroups × 8×4 f32 8×8 fragments, when the grid has ≥ 48 workgroups (not 57–79: a poor last wave on M2) and rows pad by ≤ 10%; otherwise 32×64 tiles (2 subgroups × 4×4 fragments), split-K 2 below 48 workgroups. K panels of 8 staged (converted to f32; f16 read 8 at a time as `vec4<u32>`) through workgroup memory, software-pipelined through registers. f32 accumulation. |
| linear, K % 4 = 0 otherwise | Register-blocked "direct" kernel: 4×8 outputs per thread, vec4 loads straight from global memory. |
| linear otherwise; batched and broadcast matmul | 64×64×16 workgroup-memory tiled GEMM. |
| sdpa, head dim 32 or 64, when its workgroup memory fits (D = 64: ~20 KiB) | Flash attention: online softmax, register-blocked 32q×16k tiles. With a mask, each query block first scans its mask rows and only visits the key-tile range that has a visible key (sliding window, padding). |
| sdpa, otherwise (head dims ≤ 256) | Generic flash kernel. Its tiles halve until they fit the device's `maxComputeWorkgroupStorageSize`, so a device with the 16 KiB default works. |
| layerNorm, softmax (last axis) | One workgroup per row. |
| sort | Bitonic sort in workgroup memory for rows ≤ 4096; slow per-row insertion sort above that. |

Fused `geglu` and `meanPool` are single kernels. Elementwise ops are n-ary
with broadcast strides (rank ≤ 8), and 1-D launches use 2-D grids, so
sizes above 16M elements work.

General numerics are all native: comparisons and logical ops are n-ary
kernels writing bool; `sqrt`, `rsqrt` (`inverseSqrt`), `tanh` (argument
clamped to ±15), `sigmoid`, `neg` and `abs` are n-ary unary kernels (`neg`
/ `abs` keep i32); `pow` uses a helper with C/MLX semantics (0⁰ = 1, a
negative base with an integral exponent, WGSL's `pow` being undefined for
x < 0); `erf` is the f32 lowering of math-plus tensor-core's canonical erf
(series below 1, depth-28 continued fraction above, max abs error < 2.5e-7
over [−8, 8] in the kernel tests; `gelu` keeps its own polynomial erf).
`min` and `mean` reuse the reduction kernel (f32 accumulation), `argmax` /
`argmin` are one thread per output (first index on ties), and `cumsum` is
one thread per scanned lane (sequential along the axis, f32/i32 accumulator).

## Runtime support

| Runtime | Status |
|---|---|
| Node ≥ 24 (Dawn via `webgpu@0.6`) | ✅ Conformance f32 + f16 + bf16, plus kernel edge tests (`npm test`). |
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
| linear [2048,1024]·[3072,1024]ᵀ, subgroup matrices (default under Dawn) | 2.07 TFLOP/s | 1.95 TFLOP/s |
| same shape, 0.2.0 (32×64 tiles) | 1.97 | 1.75 |
| same shape, portable kernels (`subgroupMatrix: false`, browsers without the flag) | 1.20 | 1.32 |
| same shape, MLX `x @ w.T` | 2.32 | 3.06 |
| linear [33,1024]·[3072,1024]ᵀ (skinny) | 0.32 ms | 0.19 ms |
| batched matmul [16,128,1024]·[16,1024,128] | 0.67 TFLOP/s | 0.75 TFLOP/s |

The portable kernels reach about 40% of the measured FMA peak (2.85
TFLOP/s). Subgroup matrices get f16 to ≈1.95 TFLOP/s for large M (f16
tiles are converted to f32 in workgroup memory). MLX's hand-written Metal
GEMM reaches ≈3. `bench/linear-shapes.ts` reports GFLOP/s per Laya Linear
shape and M, against main's backend and MLX in the same process;
`bench/gemm-m.ts` sweeps M per config.

GFLOP/s, f16, English Linears (hidden 1024, Wi 5248, intermediate 2624),
0.2.0 → now (MLX `x @ w.T`); best of 3 interleaved rounds:

| M = B·L | qkv 3072×1024 | o 1024×1024 | wi 5248×1024 | mlp-o 1024×2624 |
|---:|---:|---:|---:|---:|
| 33 | 885 → 956 (519) | 802 → 810 (310) | 1255 → 1260 (725) | 1056 → 1096 (565) |
| 93 | 1339 → 1425 (1117) | 1116 → 1154 (680) | 1493 → 1576 (1426) | 1206 → 1227 (1152) |
| 128 | 1487 → 1604 (1757) | 1334 → 1358 (838) | 1579 → 1744 (2021) | 1394 → 1409 (1573) |
| 256 | 1549 → 1724 (2070) | 1460 → 1488 (1241) | 1644 → 1829 (2495) | 1488 → 1546 (1798) |
| 512 | 1710 → 1879 (2215) | 1585 → 1679 (2047) | 1667 → 1856 (2792) | 1641 → 1746 (2412) |
| 1488 | 1696 → 1837 (2828) | 1637 → 1783 (2454) | 1730 → 1900 (2882) | 1693 → 1858 (2638) |
| 4096 | 1722 → 1956 (3010) | 1654 → 1856 (2835) | 1695 → 1903 (3065) | 1706 → 1929 (3003) |

MLX's small-M numbers include one FFI evaluation per Linear, so they
understate it there. Multilingual shapes (768 / 2304 / 1152) move the same way
(e.g. M=4096: 1.67–1.72 → 1.86–1.95 TFLOP/s; M=93 o/mlp-o: 0.90/0.94 → 1.18/1.26 with split-K). Chromium 152 gives the same
portable-kernel numbers. In Deno (wgpu) the large GEMM matches, but small
dispatches cost more.

**Laya English checkpoint (ModernBERT-large, 28 layers), f16, median ms per
forward (upload + forward + readback)** — `bench/grid.ts`, synthetic ids,
WebGPU and MLX interleaved per cell in one Node process.

| L → | 16 | 33 | 64 | 93 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B=1 WebGPU 0.2.0 | 15.4 | 25.7 | 39.4 | 59.0 | 77.0 | 149.3 | 288.2 |
| B=1 WebGPU | **14.6** | 24.1 | 37.9 | 53.8 | 67.2 | 130.2 | 252.2 |
| B=1 MLX | 20.0 | 23.6 | 23.6 | 41.7 | 39.0 | 80.4 | 154.4 |
| B=3 WebGPU 0.2.0 | 32.1 | 71.1 | 106.0 | 148.3 | 199.6 | 398.7 | 836.0 |
| B=3 WebGPU | 30.8 | 67.0 | 90.6 | 141.7 | 175.1 | 354.1 | 727.3 |
| B=3 MLX | 25.4 | 43.7 | 62.1 | 95.5 | 107.4 | 220.9 | 442.6 |
| B=16 WebGPU 0.2.0 | 129.7 | 279.0 | 488.7 | 723.6 | 992.2 | 2067.0 | 4440.0 |
| B=16 WebGPU | 119.6 | 252.5 | 435.7 | 653.4 | 886.3 | 1820.5 | 3842.9 |
| B=16 MLX | 79.7 | 159.4 | 271.7 | 413.2 | 545.9 | 1117.9 | 2367.3 |

All three measured in one Node process, interleaved per cell
(`BACKEND=webgpu-main,webgpu,mlx`, with main's `src/` copied to `.base/`).
WebGPU is 4–15% faster than 0.2.0 and 0.73–1.7× MLX's time. For
M = B·L > 64 the Linears are ≈85% of GPU time (B=16 L=256: 1545 of
1827 ms, ≈1.85 TFLOP/s); that is most of the gap to MLX (see Limitations).
Attention is ≈9% (169 ms at B=16 L=256). Previously (portable kernels only,
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

- Every op is its own dispatch. Encoding costs about 5 µs of CPU per
  dispatch (pass commands; bind groups are cached), about 2.4 ms per
  ModernBERT-large forward, overlapped with GPU work. WebGPU has no reusable
  compute command buffers, and there is no graph fusion (`compile` isn't
  implemented).
- Subgroup-matrix GEMM needs Dawn's experimental
  `chromium-experimental-subgroup-matrix` (f32 8×8×8, subgroup size 32);
  the WGSL/Dawn path tops out at ≈1.95 TFLOP/s on M2 (MLX's Metal GEMM ≈3).
  Measured ceiling (`subgroupMatrixMultiplyAccumulate` in a loop, M2): 3.2
  TFLOP/s with fragments held in registers, but 2.0–2.45 TFLOP/s when each
  MMA's fragments are loaded from workgroup memory, as a GEMM must; the
  kernel reaches ≈80% of that. Only f16→f16 accumulation is offered for f16
  fragments (parity needs f32), so f16 operands are converted to f32 in
  workgroup memory, doubling its traffic. Split-K, double buffering and
  larger per-subgroup blocks did not help on M2 (occupancy/registers). Kernel configs are tuned on Apple M2 and are
  untested on discrete or mobile GPUs. No other subgroup (shuffle/reduce) paths.
- `sort` rows over 4096 elements use a slow per-row insertion sort.
  `cumsum` scans each lane sequentially, so it is slow for very long axes.
  NaN handling in comparisons, `min`/`argmax` etc. follows the driver
  (WGSL may assume no NaNs).
  The generic `sdpa` kernel (head dims other than 32/64, or a device too
  small for the fast one) walks every key tile, masked or not.
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

MIT.
