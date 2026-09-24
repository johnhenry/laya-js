---
"@johnhenry/backend-webgpu": patch
---

Runtime hooks for math-plus's WebGPU convergence (johnhenry/math-plus#146, RFC 0001 §12 Q6 path (a), under which math-plus-tensor-webgpu becomes a facade over this package). All changes are additive. This is a patch so that `^0.3.0` peer ranges keep matching.

- `elementwise(expr, xs, { outDtype?, helpers? })`: a custom n-ary elementwise kernel with broadcasting over `x0, x1, …`, cached per expression. math-plus compiles its tensor-compile IR to it.
- `empty(shape, dtype)`: a scope-tracked output tensor from the pool, for custom kernels.
- `wrapBuffer(buffer, shape, dtype, offset?)`: a view of a caller-owned `GPUBuffer`, which is never pooled or destroyed (`Storage.external`).
- `Runtime`, `Storage` and the kernel types (`KernelSource`, `BindingSpec`, `ParamSpec`, `ParamType`, `CompiledKernel`, `RuntimeStats`) are exported, and `rt.kernel` / `rt.dispatch` are documented.
- `createWebGpuBackend({ device, adapter })`: subgroup-matrix detection now works for a device you pass in. Dawn doesn't mirror `subgroupMatrixConfigs` onto `device.adapterInfo`.
- `sleepThresholdMs` option (default 3, unchanged): short readbacks can lose 15–60% latency to the pre-read sleep.
- Fix: `sdpa` now respects `maxComputeWorkgroupStorageSize`. The fast kernel (~20 KiB at head dim 64) runs only where it fits, and the generic kernel's tiles halve until they fit. On a device with the 16 KiB WebGPU default, both previously failed validation for most head dims: 48, 64 and 128 at ~20–26 KiB.
