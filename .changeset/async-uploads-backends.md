---
"@johnhenry/backend-cpu": minor
"@johnhenry/backend-mlx": minor
"@johnhenry/backend-webgpu": minor
---

**Breaking: `fromHost` returns a Promise**, per `@johnhenry/tensor-backend` 0.2. Each backend still copies the host data when `fromHost` is called (CPU: a typed-array copy; MLX: `mlx_array_new_data`; WebGPU: `queue.writeBuffer`), and the Promise is already settled, so a batch of uploads awaited together costs one microtask. The benches and the WebGPU demo await their uploads.

Migration: `const x = await backend.fromHost(h)` (or `Promise.all` over several).
