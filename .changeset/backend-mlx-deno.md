---
"@johnhenry/backend-mlx": minor
---

Deno 2 support: a `Deno.dlopen` loader next to the `bun:ffi` and koffi ones (same ~60 mlx-c signatures, both mlx-c ABIs; handles cross as `usize`, callbacks are `Deno.UnsafeCallback`s). `backend.info.runtime` can now be `"deno"`. Library resolution keeps its order; under Deno the platform package is also found from the working directory's `node_modules` or Deno's npm cache (`deno add npm:@johnhenry/backend-mlx-darwin-arm64`), and loading the module over https (JSR) no longer throws in `createRequire`/`fileURLToPath`. The conformance suite (f32/f16/bf16 incl. numerics, GPU and CPU) and `@johnhenry/laya` parity on all three checkpoints pass under Deno 2.9.7 (`npm run test:deno`). Now also published to JSR as `jsr:@johnhenry/backend-mlx`.
