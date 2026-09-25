import { QUANTIZED_OPS } from "@johnhenry/tensor-backend";
import { OP_CASE_FILES, loadOpCases, runConformance, withoutOptionalOps, type OpCase, type TestApi } from "@johnhenry/tensor-backend/conformance";
import { createWebGpuBackend, isWebGpuAvailable, type WebGpuBackend } from "../src/index.ts";
import { harness, isBun } from "./harness.ts";

// @ts-ignore -- bun types are not installed
const h = harness(isBun ? await import("bun:test") : null);


const t: TestApi = { describe: h.describe, it: h.it };
const available = await isWebGpuAvailable();
const created: WebGpuBackend[] = [];
const make = (preferF16: boolean) => async () => {
  const b = await createWebGpuBackend({ preferF16 });
  created.push(b);
  return b;
};

/**
 * i8/u8/i16/u16/u64/i64/f64 are permanently unsupported here -- real WGSL
 * spec limits, not a gap (see README "Limitations"). u32 IS supported
 * (a real core WGSL type) and stays in. The outer per-run f32/f16/bf16
 * dtype loop only checks `supports()` for the run dtype itself, not each
 * individual case's own fixed dtype, so these need filtering out here the
 * same way backend-mlx excludes f64 from its GPU-device conformance run.
 */
const UNSUPPORTED = new Set(["u8", "i8", "u16", "i16", "u64", "i64", "f64"]);
const isUnsupportedCase = (c: OpCase): boolean => c.outputs.some((o) => UNSUPPORTED.has(o.dtype)) || c.inputs.some((i) => i && UNSUPPORTED.has(i.dtype));
const supportedCases = () => loadOpCases().then((cases) => cases.filter((c) => !isUnsupportedCase(c)));

if (!available) {
  h.skip("webgpu conformance", "no WebGPU adapter in this runtime");
} else {
  h.describe("webgpu (f16 enabled when shader-f16 is available)", () => {
    runConformance(make(true), supportedCases(), t);
  });
  h.describe("webgpu (f32 only, preferF16=false)", () => {
    runConformance(make(false), supportedCases(), t);
  });
  h.describe("webgpu quantized ops hidden (default composition)", () => {
    runConformance(async () => withoutOptionalOps(await make(true)(), QUANTIZED_OPS), loadOpCases(OP_CASE_FILES[2]!), t);
  });
  h.after(() => {
    for (const b of created) b.destroy();
  });
}
