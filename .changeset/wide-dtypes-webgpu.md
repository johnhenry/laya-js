---
"@johnhenry/backend-webgpu": minor
---

`supports()` now explicitly enumerates every dtype instead of defaulting unknowns to `true`: `u32` is newly supported (a real core WGSL type) with correct `add`/`sub`/`cumsum`/comparison/copy behavior across the full u32 range; `i8 u8 i16 u16 i64 u64 f64` are explicitly `false` and permanently unsupported (real WGSL spec limits, not a gap — see README "Limitations"). Along the way, fixed two real bugs the u32 addition surfaced: `st()` was collapsing any u32-storage value to a 0/1 boolean (correct for the pre-existing bool-packed-as-u32 case, wrong once u32 became a real integer dtype — `Kind` now has a `bool` flag to distinguish the two), and `cumsum`/`copyKernel`'s compute-type selection unconditionally widened any non-i32 integer input to f32, silently discarding u32 precision. `sort`/`argsort` on u32 is flagged as not yet verified correct above 2³¹ (see README) rather than silently claimed correct.
