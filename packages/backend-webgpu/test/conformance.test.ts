import { loadOpCases, runConformance, type TestApi } from "@johnhenry/tensor-backend/conformance";
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

if (!available) {
  h.skip("webgpu conformance", "no WebGPU adapter in this runtime");
} else {
  h.describe("webgpu (f16 enabled when shader-f16 is available)", () => {
    runConformance(make(true), loadOpCases(), t);
  });
  h.describe("webgpu (f32 only, preferF16=false)", () => {
    runConformance(make(false), loadOpCases(), t);
  });
  h.after(() => {
    for (const b of created) b.destroy();
  });
}
