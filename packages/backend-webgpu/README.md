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
  - `tuneGemm(shapes, { dtype?, rounds?, quantized? })`: measures every applicable
    Linear kernel (skinny, each `gemmConfig.sg` entry, direct) for each
    exact `{ M, N, K }` and records the fastest in `gemmTuning` (a `Map`
    shared by all backends on the same `GPUDevice`; persist it with
    `Object.fromEntries(backend.gemmTuning)`). Roughly 10–50 ms per shape.
    With `quantized: { bits, groupSize?, mode? }` it tunes `quantizedLinear`
    instead (keys prefixed `q`; qmv is a candidate too) and records a choice
    only when it beats the built-in one. Without it, the built-in rules below
    pick the kernel.
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
  `quantizedLinear` picks its kernel from `gemmConfig.quant`
  (`QUANT_GEMM_DEFAULT`; `QUANT_GEMM_NAVIGATOR` under `navigator.gpu`; see
  the kernel table below). Every kernel reads each weight once per tile,
  unpacks it in registers (f16-exponent trick, no int→float conversions),
  applies the group's scale (and bias) and accumulates in f32; with f16
  activations each weight is first rounded to f16, giving exactly the
  weights host dequantization uploads, fl16(q·scale + bias).
  `quantizedEmbedding` is a dequantizing gather. Any group size that is a
  multiple of 4 works, including a partial last group; other group sizes
  resolve to null (the default composition). Use them through
  `uploadQuantized` / `quantizedLinear` / `quantizedEmbedding` from
  `@johnhenry/tensor-backend`. `tuneGemm(shapes, { quantized: { bits } })`
  measures the alternatives for quantized weights and records one only
  when it beats the built-in choice.
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
| quantizedLinear, M ≤ 48 (and M ≤ 64 when N < 2048; any M without subgroup matrices) | "qmv" (memory-bound matrix-vector): 8 threads walk each weight row, one `vec2<u32>`/`u32` word load per 8 values (16 values and a `vec4`/`vec2` load for M ≤ 4, q4 M ≤ 8), unpacked and scaled in registers; each thread keeps up to 8 rows of x × 2–4 weight rows of f32 partials, summed across the 8 threads in workgroup memory (subgroup shuffles for the 16-value configs when `subgroups` has a fixed size). Larger M is split into balanced blocks of ≤ 8 rows. |
| quantizedLinear otherwise, subgroup matrices available | Subgroup-matrix tiles of BM×64 (BM 64, 48 or 32: 64 when rows pad ≤ 10% and the grid has ≥ 64 workgroups, else the tallest low-padding tile with ≥ 72 workgroups, else split-K up to 4). B is fetched raw (words + scale) into registers and dequantized when stored to workgroup memory, after the MMAs, so the loads hide behind them; A too. |
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

**Quantized Linear** (`bench/quantized-gemm.ts`; f16 activations, groups of
64, q8 symmetric / q4 affine, default kernel choice per M; 5 interleaved
rounds of ≈2·10¹⁰ flop, best round; 2026-09-24). GFLOP/s, "before" is 0.4.0
(main's `src/` via `SRC=`, run just before), fp16 kernels are unchanged
(the fp16 column is from the "after" run); ratios are to the fp16 of the
same run:

| Linear [N×K] | M | fp16 | q8 before | q8 after | q4 before | q4 after |
|---|---:|---:|---:|---:|---:|---:|
| english qkv [3072×1024] | 1 | 98 | 75 (0.78×) | **238 (2.42×)** | 83 (0.86×) | **396 (4.02×)** |
| english qkv [3072×1024] | 3 | 289 | 226 (0.78×) | **626 (2.17×)** | 250 (0.86×) | **737 (2.55×)** |
| english qkv [3072×1024] | 8 | 719 | 553 (0.79×) | **1077 (1.50×)** | 611 (0.88×) | **1068 (1.49×)** |
| english qkv [3072×1024] | 16 | 1229 | 826 (0.67×) | **1253 (1.02×)** | 877 (0.71×) | **1174 (0.96×)** |
| english qkv [3072×1024] | 33 | 1297 | 673 (0.52×) | **1248 (0.96×)** | 660 (0.51×) | **1197 (0.92×)** |
| english qkv [3072×1024] | 64 | 1465 | 1254 (0.83×) | **1565 (1.07×)** | 1212 (0.81×) | **1524 (1.04×)** |
| english qkv [3072×1024] | 93 | 1454 | 1345 (0.91×) | **1387 (0.95×)** | 1311 (0.89×) | **1355 (0.93×)** |
| english qkv [3072×1024] | 128 | 1640 | 1412 (0.85×) | **1658 (1.01×)** | 1411 (0.85×) | **1633 (1.00×)** |
| english qkv [3072×1024] | 256 | 1731 | 1527 (0.87×) | **1796 (1.04×)** | 1489 (0.84×) | **1770 (1.02×)** |
| english qkv [3072×1024] | 1024 | 1910 | 1683 (0.87×) | **1985 (1.04×)** | 1650 (0.85×) | **1942 (1.02×)** |
| english o [1024×1024] | 1 | 90 | 58 (0.51×) | **204 (2.26×)** | 66 (0.58×) | **202 (2.24×)** |
| english o [1024×1024] | 3 | 322 | 179 (0.54×) | **484 (1.50×)** | 204 (0.62×) | **479 (1.49×)** |
| english o [1024×1024] | 8 | 659 | 427 (0.63×) | **741 (1.12×)** | 479 (0.70×) | **784 (1.19×)** |
| english o [1024×1024] | 16 | 886 | 666 (0.73×) | **1013 (1.14×)** | 711 (0.78×) | **1015 (1.15×)** |
| english o [1024×1024] | 33 | 1004 | 618 (0.62×) | **1077 (1.07×)** | 608 (0.60×) | **1036 (1.03×)** |
| english o [1024×1024] | 64 | 1190 | 1159 (0.97×) | **1327 (1.11×)** | 1128 (0.94×) | **1232 (1.04×)** |
| english o [1024×1024] | 93 | 1181 | 1087 (0.92×) | **1421 (1.20×)** | 1056 (0.90×) | **1371 (1.16×)** |
| english o [1024×1024] | 128 | 1378 | 1243 (0.91×) | **1326 (0.96×)** | 1222 (0.90×) | **1332 (0.97×)** |
| english o [1024×1024] | 256 | 1488 | 1384 (0.92×) | **1691 (1.14×)** | 1333 (0.89×) | **1674 (1.13×)** |
| english o [1024×1024] | 1024 | 1845 | 1594 (0.87×) | **1846 (1.00×)** | 1566 (0.85×) | **1867 (1.01×)** |
| english wi [5248×1024] | 1 | 77 | 74 (0.91×) | **228 (2.96×)** | 81 (0.99×) | **436 (5.65×)** |
| english wi [5248×1024] | 3 | 230 | 221 (0.91×) | **653 (2.83×)** | 241 (0.99×) | **786 (3.41×)** |
| english wi [5248×1024] | 8 | 601 | 540 (0.89×) | **1183 (1.97×)** | 597 (0.99×) | **1135 (1.89×)** |
| english wi [5248×1024] | 16 | 1090 | 815 (0.74×) | **1330 (1.22×)** | 873 (0.79×) | **1278 (1.17×)** |
| english wi [5248×1024] | 33 | 1275 | 746 (0.59×) | **1268 (0.99×)** | 723 (0.57×) | **1221 (0.96×)** |
| english wi [5248×1024] | 64 | 1493 | 1382 (0.90×) | **1526 (1.02×)** | 1357 (0.88×) | **1513 (1.01×)** |
| english wi [5248×1024] | 93 | 1594 | 1449 (0.91×) | **1688 (1.06×)** | 1429 (0.90×) | **1656 (1.04×)** |
| english wi [5248×1024] | 128 | 1717 | 1515 (0.86×) | **1788 (1.04×)** | 1483 (0.84×) | **1771 (1.03×)** |
| english wi [5248×1024] | 256 | 1856 | 1608 (0.87×) | **1917 (1.03×)** | 1572 (0.85×) | **1887 (1.02×)** |
| english wi [5248×1024] | 1024 | 1955 | 1682 (0.87×) | **2001 (1.02×)** | 1651 (0.85×) | **1992 (1.02×)** |
| english mlp-o [1024×2624] | 1 | 90 | 61 (0.68×) | **229 (2.55×)** | 48 (0.53×) | **317 (3.53×)** |
| english mlp-o [1024×2624] | 3 | 264 | 184 (0.68×) | **560 (2.12×)** | 145 (0.53×) | **558 (2.12×)** |
| english mlp-o [1024×2624] | 8 | 652 | 452 (0.70×) | **852 (1.31×)** | 374 (0.58×) | **874 (1.34×)** |
| english mlp-o [1024×2624] | 16 | 1029 | 703 (0.69×) | **1156 (1.12×)** | 684 (0.67×) | **1142 (1.11×)** |
| english mlp-o [1024×2624] | 33 | 1114 | 650 (0.59×) | **1197 (1.07×)** | 609 (0.55×) | **1142 (1.02×)** |
| english mlp-o [1024×2624] | 64 | 1291 | 1212 (0.94×) | **1500 (1.16×)** | 1142 (0.89×) | **1366 (1.06×)** |
| english mlp-o [1024×2624] | 93 | 1233 | 1103 (0.89×) | **1466 (1.19×)** | 1031 (0.84×) | **1226 (0.99×)** |
| english mlp-o [1024×2624] | 128 | 1411 | 1255 (0.88×) | **1417 (1.00×)** | 1173 (0.82×) | **1171 (0.83×)** |
| english mlp-o [1024×2624] | 256 | 1542 | 1385 (0.90×) | **1765 (1.14×)** | 1227 (0.80×) | **1726 (1.12×)** |
| english mlp-o [1024×2624] | 1024 | 1934 | 1619 (0.84×) | **1965 (1.02×)** | 1550 (0.81×) | **1910 (0.99×)** |
| multilingual qkv [2304×768] | 1 | 90 | 68 (0.75×) | **336 (3.71×)** | 80 (0.88×) | **319 (3.52×)** |
| multilingual qkv [2304×768] | 3 | 269 | 202 (0.75×) | **611 (2.27×)** | 239 (0.89×) | **614 (2.28×)** |
| multilingual qkv [2304×768] | 8 | 669 | 497 (0.74×) | **977 (1.46×)** | 551 (0.82×) | **931 (1.39×)** |
| multilingual qkv [2304×768] | 16 | 1055 | 764 (0.72×) | **1147 (1.09×)** | 822 (0.78×) | **1052 (1.00×)** |
| multilingual qkv [2304×768] | 33 | 1149 | 643 (0.56×) | **1161 (1.01×)** | 627 (0.55×) | **1116 (0.97×)** |
| multilingual qkv [2304×768] | 64 | 1352 | 1207 (0.91×) | **1337 (0.99×)** | 1190 (0.90×) | **1334 (0.99×)** |
| multilingual qkv [2304×768] | 93 | 1267 | 1183 (0.90×) | **1487 (1.17×)** | 1133 (0.86×) | **1450 (1.14×)** |
| multilingual qkv [2304×768] | 128 | 1491 | 1345 (0.91×) | **1371 (0.92×)** | 1301 (0.88×) | **1336 (0.90×)** |
| multilingual qkv [2304×768] | 256 | 1660 | 1398 (0.84×) | **1658 (1.00×)** | 1379 (0.83×) | **1641 (0.99×)** |
| multilingual qkv [2304×768] | 1024 | 1848 | 1586 (0.86×) | **1912 (1.03×)** | 1552 (0.84×) | **1877 (1.02×)** |
| multilingual o [768×768] | 1 | 55 | 44 (0.54×) | **116 (2.12×)** | 50 (0.62×) | **132 (2.42×)** |
| multilingual o [768×768] | 3 | 185 | 130 (0.54×) | **299 (1.61×)** | 151 (0.63×) | **291 (1.57×)** |
| multilingual o [768×768] | 8 | 470 | 315 (0.69×) | **528 (1.12×)** | 347 (0.76×) | **633 (1.35×)** |
| multilingual o [768×768] | 16 | 653 | 482 (0.74×) | **906 (1.39×)** | 526 (0.81×) | **873 (1.34×)** |
| multilingual o [768×768] | 33 | 723 | 505 (0.70×) | **997 (1.38×)** | 488 (0.68×) | **959 (1.33×)** |
| multilingual o [768×768] | 64 | 1073 | 959 (0.89×) | **1193 (1.11×)** | 932 (0.86×) | **1113 (1.04×)** |
| multilingual o [768×768] | 93 | 1212 | 1119 (0.92×) | **1163 (0.96×)** | 1094 (0.90×) | **1153 (0.95×)** |
| multilingual o [768×768] | 128 | 1184 | 1069 (0.90×) | **1400 (1.18×)** | 1038 (0.87×) | **1400 (1.18×)** |
| multilingual o [768×768] | 256 | 1509 | 1322 (0.86×) | **1558 (1.03×)** | 1299 (0.85×) | **1516 (1.00×)** |
| multilingual o [768×768] | 1024 | 1675 | 1426 (0.84×) | **1764 (1.05×)** | 1428 (0.84×) | **1770 (1.06×)** |
| multilingual wi [2304×768] | 1 | 90 | 67 (0.74×) | **336 (3.72×)** | 80 (0.88×) | **321 (3.56×)** |
| multilingual wi [2304×768] | 3 | 274 | 201 (0.75×) | **609 (2.22×)** | 238 (0.89×) | **592 (2.16×)** |
| multilingual wi [2304×768] | 8 | 666 | 495 (0.73×) | **984 (1.48×)** | 548 (0.81×) | **943 (1.42×)** |
| multilingual wi [2304×768] | 16 | 1051 | 764 (0.72×) | **1146 (1.09×)** | 821 (0.78×) | **1048 (1.00×)** |
| multilingual wi [2304×768] | 33 | 1146 | 641 (0.56×) | **1155 (1.01×)** | 621 (0.54×) | **1111 (0.97×)** |
| multilingual wi [2304×768] | 64 | 1327 | 1196 (0.91×) | **1338 (1.01×)** | 1179 (0.90×) | **1315 (0.99×)** |
| multilingual wi [2304×768] | 93 | 1302 | 1186 (0.92×) | **1470 (1.13×)** | 1132 (0.88×) | **1452 (1.12×)** |
| multilingual wi [2304×768] | 128 | 1470 | 1332 (0.91×) | **1355 (0.92×)** | 1293 (0.88×) | **1343 (0.91×)** |
| multilingual wi [2304×768] | 256 | 1648 | 1413 (0.85×) | **1668 (1.01×)** | 1375 (0.82×) | **1637 (0.99×)** |
| multilingual wi [2304×768] | 1024 | 1847 | 1596 (0.86×) | **1912 (1.04×)** | 1565 (0.84×) | **1886 (1.02×)** |
| multilingual mlp-o [768×1152] | 1 | 58 | 44 (0.51×) | **162 (2.81×)** | 51 (0.58×) | **141 (2.44×)** |
| multilingual mlp-o [768×1152] | 3 | 252 | 134 (0.54×) | **418 (1.66×)** | 153 (0.62×) | **420 (1.67×)** |
| multilingual mlp-o [768×1152] | 8 | 484 | 326 (0.66×) | **607 (1.26×)** | 358 (0.73×) | **733 (1.52×)** |
| multilingual mlp-o [768×1152] | 16 | 670 | 507 (0.76×) | **971 (1.45×)** | 538 (0.80×) | **948 (1.41×)** |
| multilingual mlp-o [768×1152] | 33 | 776 | 529 (0.68×) | **1083 (1.40×)** | 506 (0.65×) | **1035 (1.33×)** |
| multilingual mlp-o [768×1152] | 64 | 1141 | 1013 (0.88×) | **1271 (1.11×)** | 964 (0.84×) | **1193 (1.05×)** |
| multilingual mlp-o [768×1152] | 93 | 1285 | 1173 (0.91×) | **1222 (0.95×)** | 1118 (0.87×) | **1184 (0.92×)** |
| multilingual mlp-o [768×1152] | 128 | 1223 | 1092 (0.89×) | **1466 (1.20×)** | 1033 (0.85×) | **1397 (1.14×)** |
| multilingual mlp-o [768×1152] | 256 | 1569 | 1370 (0.86×) | **1553 (0.99×)** | 1324 (0.83×) | **1502 (0.96×)** |
| multilingual mlp-o [768×1152] | 1024 | 1750 | 1478 (0.84×) | **1828 (1.04×)** | 1447 (0.82×) | **1797 (1.03×)** |

- M ≤ 8: 1.1–5.7× fp16 (median 2.1×). The fp16 skinny kernel is
  latency-bound there; qmv reads half / a quarter of the bytes with many
  more threads in flight.
- 16 ≤ M ≤ 64: median 1.06×, 0.92–1.45×. The weakest cells are the large-N
  shapes around M = 33 (English qkv/wi, multilingual qkv/wi: 0.92–0.97×):
  each 8-row block re-dequantizes the weights, and the fp16 skinny kernel
  is at its best there.
- M ≥ 93: median 1.02×; 0.92–1.20× at M = 93 and 0.99–1.06× at M = 1024.
  Mid-size cells where the simple tile rule picks badly stay at 0.83–0.97×
  (English mlp-o q4 and multilingual qkv/wi at M = 128, English qkv and
  multilingual o/mlp-o at M = 93).
- Before: 0.51–0.99× for M ≤ 64 and 0.80–0.92× above.

The model-level result is what the kernel choice was tuned for: see
docs/RESULTS.md (English q8 on WebGPU: 1 question 51.6 vs 54.7 ms,
16 questions 643 vs 657 ms, quiet machine).

**Laya English checkpoint (ModernBERT-large, 28 layers), f16, median ms per
forward (upload + forward + readback)** — `bench/grid.ts`, synthetic ids,
WebGPU and MLX interleaved per cell in one Node process.

| L → | 16 | 33 | 64 | 93 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B=1 WebGPU 0.2.0 | 16.6 | 25.5 | 40.2 | 58.3 | 74.9 | 140.3 | 286.5 |
| B=1 WebGPU | **14.6** | 24.3 | 37.9 | 54.5 | 67.6 | 128.0 | 252.6 |
| B=1 MLX | 20.4 | 23.8 | 23.6 | 40.8 | 44.4 | 81.8 | 148.3 |
| B=3 WebGPU 0.2.0 | 32.3 | 71.5 | 106.4 | 147.7 | 198.9 | 397.7 | 834.7 |
| B=3 WebGPU | 30.9 | 67.1 | 90.6 | 141.2 | 175.2 | 352.3 | 730.1 |
| B=3 MLX | 25.8 | 46.2 | 61.4 | 96.4 | 106.4 | 212.8 | 452.0 |
| B=16 WebGPU 0.2.0 | 131.1 | 270.3 | 489.9 | 720.4 | 992.2 | 2074.2 | 4473.6 |
| B=16 WebGPU | 120.6 | 250.9 | 437.2 | 653.6 | 879.7 | 1824.8 | 3853.9 |
| B=16 MLX | 80.8 | 165.7 | 280.4 | 413.2 | 556.0 | 1131.9 | 2354.0 |

Quiet machine, 2026-09-24 (backend-webgpu 0.5.0). All three measured in
one Node process per batch size, interleaved per cell
(`BACKEND=webgpu-main,webgpu,mlx BS=<B>`, with 0.2.0's `src/` copied to
`.base/`).
WebGPU is 4–15% faster than 0.2.0 and 0.72–1.7× MLX's time. For
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

- `sleepWhileWaiting` (the default on Node/Bun) keys its expected wait by the
  number of dispatches since the last readback, not by shape. So after a
  large batch, a smaller batch with the same graph sleeps for most of the
  large batch's GPU time. The estimate shrinks by 20% per read, so it takes
  about 15 calls to recover. Measured on the English checkpoint: B=3 L=16
  took 594, 476, 374, … 136 ms after B=3 L=512, against 31 ms steady state.
  Until this is fixed, pass `sleepWhileWaiting: false` when shapes vary a
  lot between calls. Browsers (`navigator.gpu`) are not affected.
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
  untested on discrete or mobile GPUs. The only other subgroup path is qmv's shuffle reduction (quantized weights).
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
- Quantized Linears match or beat fp16 for most shapes and M, but not all:
  0.92–0.97× on the large-N Laya shapes around M = 33 and 0.83–0.97× in a
  few cells at M = 93–128 (table above). The kernel choice
  (`QUANT_GEMM_DEFAULT`, `QUANT_GEMM_NAVIGATOR`) is tuned on one Apple M2
  (Dawn in Node, and Chromium without subgroup matrices); other GPUs may
  want other configs (`tuneGemm({ quantized })`). There is no integer-dot
  path. Group sizes must be multiples of 4 (8 for the fast paths, 16 for
  qmv's 16-value steps). The subgroup-shuffle reduction assumes a fixed
  subgroup size and subgroups made of consecutive invocations (as the
  subgroup-matrix kernel does).
- Node/Bun need the platform's prebuilt Dawn addon (`webgpu` package:
  darwin universal, linux x64/arm64, win32 x64/arm64).

## Family

Part of **[laya-js](https://github.com/johnhenry/laya-js#readme)**, Laya typed decisions in JavaScript on MLX, WebGPU and CPU — see its [package map](https://github.com/johnhenry/laya-js#which-package-do-i-want) and [results](https://github.com/johnhenry/laya-js#results).

- Implements [`@johnhenry/tensor-backend`](https://github.com/johnhenry/laya-js/tree/main/packages/tensor-backend); [`@johnhenry/laya`](https://github.com/johnhenry/laya-js/tree/main/packages/laya) selects it under `backend: "auto"` when MLX is unavailable (optional peer dependency).
- Sibling in spirit of [`@johnhenry/math-plus-tensor-webgpu`](https://github.com/johnhenry/math-plus/tree/main/packages/tensor-webgpu) (general tensor GEMM/attention in the browser); this one is specialized to the transformer-inference op contract.

## License

MIT.
