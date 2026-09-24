import * as nodeTest from "node:test";
// Bun 1.2's node:test shim only registers tests from the first file of a run,
// so use bun:test natively under Bun (same describe/it/test API).
// @ts-ignore -- bun types are not installed
const bunTest: unknown = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const { describe, it } = (bunTest ?? nodeTest) as Pick<typeof nodeTest, "describe" | "it">;
import { loadOpCases, runConformance, withoutOptionalOps, type TestApi } from "@johnhenry/tensor-backend/conformance";
import { NUMERICS_OPS } from "@johnhenry/tensor-backend";
import { createCpuBackend } from "../src/index.ts";

// Cast: node:test's `it` wants `() => void | Promise<void>` while TestApi passes `() => unknown`.
const api = { describe, it: it as unknown as TestApi["it"] };
runConformance(() => createCpuBackend(), loadOpCases(), api);
// The compose.ts default compositions of the optional numerics ops (cumsum stays native: it has none).
describe("default compositions (numerics ops hidden)", () => {
  runConformance(() => withoutOptionalOps(createCpuBackend(), NUMERICS_OPS.filter((o) => o !== "cumsum")), loadOpCases(), api);
});
