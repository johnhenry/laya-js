import * as nodeTest from "node:test";
// Bun 1.2's node:test shim only registers tests from the first file of a run,
// so use bun:test natively under Bun (same describe/it/test API).
// @ts-ignore -- bun types are not installed
const bunTest: unknown = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const { describe, it } = (bunTest ?? nodeTest) as Pick<typeof nodeTest, "describe" | "it">;
import { loadOpCases, runConformance, type TestApi } from "@johnhenry/tensor-backend/conformance";
import { createCpuBackend } from "../src/index.ts";

// Cast: node:test's `it` wants `() => void | Promise<void>` while TestApi passes `() => unknown`.
runConformance(() => createCpuBackend(), loadOpCases(), { describe, it: it as unknown as TestApi["it"] });
