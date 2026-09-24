---
"@johnhenry/backend-webgpu": patch
---

`sleepWhileWaiting`: the readback-wait estimate is now keyed by dispatch count **and** total workgroups (i.e. per shape). Before, a small input after a large one with the same op graph slept for most of the large input's GPU time (measured 594 ms instead of 31 ms, ~15 calls to recover) on Node/Bun. New `rt.waitEstimates` exposes the estimates for diagnostics.
