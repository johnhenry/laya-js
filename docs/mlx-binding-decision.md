# MLX binding decision (Workstream D, phase D1)

**Decision:** we wrote our own thin binding to **mlx-c**, about 250 lines
in `packages/backend-mlx/src/ffi.ts`. Bun calls it through `bun:ffi` and
Node calls it through `koffi`. The binding loads a `libmlxc.dylib`
resolved at runtime. We do not build on `@nielspeter/mlx-ts`. We do reuse
its prebuilt native bundle (`@nielspeter/mlx-ts-darwin-arm64`) as a
fallback library source.

The preferred library is one we build ourselves:
`scripts/build-mlxc.sh` compiles mlx-c against the `mlx` 0.32.2 Python
wheel in about 10 s, which matches laya-mlx exactly.

Machine: M3 Max, macOS 27 (Darwin 27.0), Node 24.9.0, Bun 1.2.17,
Python MLX 0.32.2. Date: 2026-09-22.

## Candidates

### (a) `@nielspeter/mlx-ts` 0.5.0

- A TypeScript SDK over mlx-c. It runs on Bun, Deno and Node (Node through
  koffi) and ships a model zoo covering LLMs, Whisper, Stable Diffusion and
  TTS.
- Its native bundle, `@nielspeter/mlx-ts-darwin-arm64`, contains libmlx,
  libjaccl and mlx.metallib from Apple's `mlx-metal` 0.32.1 wheel. It also
  contains a `libmlxc` built from mlx-c *before* the 0.32.2 bindings
  (#123). That build has no `mlx_compile_cache_*` symbols and uses the
  9-argument sdpa signature.
- The prebuilt koffi (`@koromix/koffi-darwin-arm64`) works without install
  scripts.

### (b) Our own binding

- Handles are one-pointer structs passed by value. On arm64 that is
  ABI-identical to passing a pointer, so every handle crosses the boundary
  as a JS number.
- Out-params point into one long-lived scratch `ArrayBuffer`. Its address
  is computed once, and an off-heap backing store never moves.
- Errors are recorded by our own `mlx_set_error_handler`.
- `compile` goes through `mlx_closure_new_func_payload` with a single C
  trampoline, dispatched by payload id.
- libmlxc comes from one of the sources in the next section.

## Where libmlxc comes from

| Source | MLX | mlx-c ABI | Status |
|---|---|---|---|
| `scripts/build-mlxc.sh` → `prebuilds/darwin-arm64/` | **0.32.2** (Python wheel binaries) | ≥ 0.6 (sdpa has `force_fused`) | Builds in about 9 s. It only compiles mlx-c's C++ wrapper. MLX and its Metal kernels are Apple's wheel binaries. |
| `@nielspeter/mlx-ts-darwin-arm64` 0.1.0 | 0.32.1 | < 0.6 | Works as it comes. Conformance is also green with it. |
| Homebrew `mlx-c` | varies | varies | Not installed here. It is supported through ABI detection. |
| Full mlx-c + MLX source build (FetchContent) | 0.32.2 | ≥ 0.6 | Not needed. It would rebuild every Metal kernel. |

Build notes:

- The Command Line Tools **MacOSX27.0 SDK's `.tbd` stubs are rejected by
  the linker** ("unknown architecture"). This is the same breakage that
  kills MLX's CPU JIT here.
- The MacOSX26.5 SDK links. The script probes SDKs until one links.
- Metal GPU is the default and is unaffected.
- We never depend on MLX's CPU JIT. `compile` on `device: "cpu"` is the
  identity.

The backend detects the ABI at load time from whether
`mlx_compile_cache_new` exists. Both ABIs must be handled because they
differ in:

- `mlx_fast_scaled_dot_product_attention`, which gained `bool force_fused`;
- `mlx_detail_compile_clear_cache`, which is `(void)` in the old ABI and
  takes a `mlx_compile_cache` in the new one.

Loading an mlx-c ≥ 0.6 library through mlx-ts's hard-coded 9-argument sdpa
binding would put the stream handle into the `force_fused` slot.

## Findings

| | (a) mlx-ts 0.5.0 | (b) own binding |
|---|---|---|
| Runs on Node 24 / Bun 1.2 | yes / yes | yes / yes |
| Import + first backend | 621 ms (Node), 158 ms (Bun). It loads the whole SDK. | 60 ms (Node), 68 ms (Bun) |
| Op dispatch, build + free, per op | 1.0–1.2 µs (Node), 0.67 µs (Bun) | **0.80 µs (Node), 0.55 µs (Bun)**, including the FinalizationRegistry and scope bookkeeping (Python: 0.40 µs) |
| Chained 10k adds, build + eval, per op | 3.0 µs (Node), 2.9 µs (Bun) | 5.8 µs (Node), 2.5 µs (Bun) (Python: 1.8 µs). Node's cost is the dispose of 10k handles in `scope`. |
| fp16 / bf16 upload | No: `fromF32`/`fromI32`/`fromU32` only | Yes: f32, f16 (Float16Array), bf16 (raw bits), i32, bool |
| `fast.rope` | yes (MX.rope) | yes, `traditional=false` |
| `fast.sdpa` with **bool array mask** | **no** (causal or none only) | yes |
| `fast.layer_norm`, erf | yes | yes |
| `compile` | not exposed | yes, `mlx_compile` over a JS trampoline |
| Lifetime model | its own `tidy()` arena plus FinalizationRegistry | our contract's `scope`/`dispose` plus FinalizationRegistry |
| Contract fit | Needs an adapter over `MX`, still needs raw `m.*` calls for the missing ops, and has its own arena | Implements `Backend` directly |
| mlx-c ≥ 0.6 / MLX 0.32.2 | ABI mismatch (sdpa) | detected at load time |
| Conformance, f32 + f16 | not attempted | **green on Node and Bun, GPU and CPU devices, with both libraries** |

GPU work runs in the same `libmlx`, so kernel time is identical for any
binding. Only dispatch overhead and API coverage can differ.

## Performance vs Python MLX

All runs use `bench/ops.bench.ts` and `bench/python_ref.py`, interleaved
under `~/gpu.lock`. The machine was shared with other agents (load average
4–7), so expect about ±30% noise. Numbers are medians.

| Benchmark | Node | Bun | Python |
|---|---:|---:|---:|
| linear f16 [16,128,1024]·[1024,1024]ᵀ | 1.50 ms | 1.51–1.87 ms | 1.48–1.54 ms |
| sdpa f16 [16,16,128,64], bool mask | 0.90–0.96 ms | 0.75 ms | 0.68 ms |
| 1 ModernBERT-large layer f16, B=16 L=128 | 17.9–18.9 ms | 17.9–19.2 ms | 18.3–19.9 ms |
| 1 layer, B=1 L=32 (bimodal: GPU clocks) | 1.1–2.0 ms | 1.2–2.0 ms | 1.0–2.0 ms |
| 8 stacked layers, B=1 L=32 | 5.5 ms | 7.3 ms | 9.2 ms |
| graph build for 1 layer (JS + FFI) | 0.03 ms | 0.04 ms | 0.02 ms |

On the real English checkpoint (ModernBERT-large 421M plus heads), run
through Workstream C's `@johnhenry/laya` model on this backend
(`bench/real-english.ts`):

- Accuracy against the laya-mlx fp32 fixture, 63 questions:
  - **f32:** argmax 63/63, maximum |Δlogit| **1.28e-5**, maximum relative
    |Δact| 1.6e-6.
  - **f16:** argmax 63/63, maximum |Δlogit| **4.0e-2**, maximum relative
    |Δact| 2.3e-3.
- Latency for one short question (33 tokens, B=1), P50. The Python
  column comes from `bench/python_real_short.py`, run in the same lock
  window with `Agent`, compile off, which is the released default.

  | dtype | Node | Bun | Python |
  |---|---:|---:|---:|
  | f16 | 22–42 ms | 33–42 ms | 26.9–33 ms |
  | f32 | 46–57 ms | 50–63 ms | 50–66 ms |

- Profile of one f16 forward pass in Node: graph build about 1.4 ms, eval
  about 20.5 ms, read 0.2 ms. The best Node f16 run (22.2 ms) beat Python
  in the same window (32 ms). In the other window Node was slower (42 ms
  vs 33 ms).
- The README's **13.4 ms** could not be reproduced today, **even by Python**
  (21.8–33 ms f16 across runs), because the GPU was contended. The gap is
  environmental, not the binding's. We should re-measure on a quiet
  machine.

## Risks

1. **Native distribution.** No npm package of ours ships a libmlxc yet.
   - Consumers today need our build script (Python wheel plus cmake), the
     third-party `@nielspeter/mlx-ts-darwin-arm64` (MLX 0.32.1, about
     200 MB unpacked, MIT, not ours), or Homebrew.
   - Recommendation: publish `@johnhenry/backend-mlx-darwin-arm64` from
     `build-mlxc.sh` in CI on a macOS arm64 runner, with `os`/`cpu`
     fields, as an `optionalDependency`.
   - `mlx.metallib` alone is 174–182 MB. Check npm size limits; it may need
     a download-on-first-use fallback.
   - Apple's MIT notice must ship with it.
2. **ABI drift.** mlx-c is 0.x and its C signatures change: #123 changed
   sdpa and the compile-cache API. We bind about 60 symbols by hand and
   detect only the known variant. A mismatched Homebrew mlx-c could cause
   silent argument misalignment. Mitigations:
   - pin the bundled build;
   - check the conformance suite in CI for each library source;
   - consider reading `mlx_version()` and refusing unknown majors.
3. **FFI quirks found.**
   - Bun's `ptr` arguments turn small non-pointer integers into 0. The
     closure payload id is therefore passed as `usize`.
   - Bun returns `usize` as a BigInt.
   - Clearing MLX's compile cache is required at exit; otherwise the
     process exits with a segfault (139).
   - Callbacks must never throw through C frames. The trampoline catches
     errors and re-raises them after `mlx_closure_apply` returns.
4. **CPU JIT.** MLX CPU `compile` needs a C++ toolchain at runtime, and this
   machine's CLT SDK is broken for linking. `device: "cpu"` therefore
   disables compile. It is correct but slower.
5. **Lifetime.** FinalizationRegistry only runs after GC. A long-lived
   service that skips `scope`/`dispose` can grow unboundedly between GCs,
   which is the same issue mlx-ts documents. The tests check that
   `scope`/`dispose` keep active memory flat.
6. **Deno** has no adapter yet. `Deno.dlopen` would be about 60 lines, and
   mlx-ts proves it works.
