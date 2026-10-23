import * as nodeTest from "node:test";
// Bun 1.2's node:test shim only registers tests from the first file of a run.
// @ts-ignore -- bun types are not installed
const bunTest: unknown = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const { it: test } = (bunTest ?? nodeTest) as Pick<typeof nodeTest, "it">;
import assert from "node:assert/strict";
import { guardState, moderationState, routerState, triageState } from "../src/index.ts";

test("state builders use the field each question set names", () => {
  assert.deepEqual(guardState("hi"), { prompt: "hi" });
  assert.deepEqual(moderationState("hi"), { post: "hi" });
  assert.deepEqual(routerState("hi"), { request: "hi" });
  assert.deepEqual(triageState("hi"), { message: "hi" });
});
