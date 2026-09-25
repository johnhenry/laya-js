import * as nodeTest from "node:test";
// Bun 1.2's node:test shim only registers tests from the first file of a run,
// and bun:test binds describe/it per importing file, so import it here.
// @ts-ignore -- bun types are not installed
const bunTest: unknown = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const { describe, it } = (bunTest ?? nodeTest) as Pick<typeof nodeTest, "describe" | "it">;
import { NUMERICS_OPS, QUANTIZED_OPS } from "@johnhenry/tensor-backend";
import { OP_CASE_FILES, loadOpCases, runConformance, withoutOptionalOps, type OpCase, type TestApi } from "@johnhenry/tensor-backend/conformance";
import { createMlxBackend } from "../src/index.ts";
import { skipReason } from "./env.ts";

const api = { describe, it: it as unknown as TestApi["it"] };

/**
 * f64 is CPU-only on MLX (no Apple GPU has double-precision hardware --
 * confirmed against MLX's own docs and ml-explore/mlx#799), so f64 cases
 * are real, permanent failures on the default (GPU) backend, not a gap:
 * excluded here, covered instead by the dedicated `device: cpu` run below,
 * which already includes them (no filtering there).
 */
const isF64Case = (c: OpCase): boolean => c.outputs.some((o) => o.dtype === "f64") || c.inputs.some((i) => i?.dtype === "f64");
const nonF64Cases = () => loadOpCases().then((cases) => cases.filter((c) => !isF64Case(c)));

if (skipReason) {
  describe("tensor-backend conformance (mlx)", () => it.skip(`skipped: ${skipReason}`, () => {}));
} else {
  // GPU (default) and CPU devices; f32 + f16 + bf16 per device (the harness does all three).
  runConformance(() => createMlxBackend(), nonF64Cases(), api);
  describe("device: cpu", () => {
    runConformance(() => createMlxBackend({ device: "cpu" }), loadOpCases(), api);
  });
  // compose.ts default compositions in f32/f16/bf16 (cumsum has none, so it stays native).
  describe("default compositions (numerics ops hidden)", () => {
    runConformance(() => withoutOptionalOps(createMlxBackend(), NUMERICS_OPS.filter((o) => o !== "cumsum")), nonF64Cases(), api);
  });
  // quantized weights through the default composition (dequantize on the device)
  describe("default compositions (quantized ops hidden)", () => {
    runConformance(() => withoutOptionalOps(createMlxBackend(), QUANTIZED_OPS), loadOpCases(OP_CASE_FILES[2]!), api);
  });
}
