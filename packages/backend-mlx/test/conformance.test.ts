import * as nodeTest from "node:test";
// Bun 1.2's node:test shim only registers tests from the first file of a run,
// and bun:test binds describe/it per importing file, so import it here.
// @ts-ignore -- bun types are not installed
const bunTest: unknown = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const { describe, it } = (bunTest ?? nodeTest) as Pick<typeof nodeTest, "describe" | "it">;
import { NUMERICS_OPS } from "@johnhenry/tensor-backend";
import { loadOpCases, runConformance, withoutOptionalOps, type TestApi } from "@johnhenry/tensor-backend/conformance";
import { createMlxBackend } from "../src/index.ts";
import { skipReason } from "./env.ts";

const api = { describe, it: it as unknown as TestApi["it"] };
if (skipReason) {
  describe("tensor-backend conformance (mlx)", () => it.skip(`skipped: ${skipReason}`, () => {}));
} else {
  // GPU (default) and CPU devices; f32 + f16 + bf16 per device (the harness does all three).
  runConformance(() => createMlxBackend(), loadOpCases(), api);
  describe("device: cpu", () => {
    runConformance(() => createMlxBackend({ device: "cpu" }), loadOpCases(), api);
  });
  // compose.ts default compositions in f32/f16/bf16 (cumsum has none, so it stays native).
  describe("default compositions (numerics ops hidden)", () => {
    runConformance(() => withoutOptionalOps(createMlxBackend(), NUMERICS_OPS.filter((o) => o !== "cumsum")), loadOpCases(), api);
  });
}
