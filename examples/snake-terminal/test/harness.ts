/**
 * `test()` that runs under both `node --test` and `bun test` (same pattern as
 * packages/laya-core/test/harness.ts: under Bun each file passes its own bun:test).
 */
import * as nodeTest from "node:test";

export type TestFn = (name: string, fn: () => unknown, opts?: { skip?: string | false | null; timeout?: number }) => void;

export function makeTest(bunTest: unknown): TestFn {
  const bun = bunTest as {
    test: ((n: string, f: () => unknown, timeout?: number) => void) & { skip: (n: string, f: () => unknown) => void };
  } | null;
  return (name, fn, opts = {}) => {
    if (bun) opts.skip ? bun.test.skip(`${name} (skipped: ${opts.skip})`, fn) : bun.test(name, fn, opts.timeout ?? 600_000);
    else nodeTest.test(name, { skip: opts.skip || false, timeout: opts.timeout ?? 600_000 }, fn as () => void);
  };
}

export interface Frame {
  type: "frame";
  game: import("../src/core/game.ts").GameSnapshot;
  decision: import("../src/core/policy.ts").Decision;
}

export async function readFrames(): Promise<{ meta: Record<string, unknown>; frames: Frame[] }> {
  const { readFile } = await import("node:fs/promises");
  const text = await readFile(new URL("./fixtures/snake-showcase.head.jsonl", import.meta.url), "utf8");
  const lines = text.trim().split("\n").map((l) => JSON.parse(l));
  return { meta: lines[0], frames: lines.filter((l) => l.type === "frame") };
}

export interface PromptCase {
  tick: number;
  state: string;
  questions: Record<string, unknown>;
  items: { ids: number[]; markers: number[]; qtype: number }[];
  probabilities: Record<string, number>;
  risk_noul: number;
  food_noul: number;
}

/** Python `Agent.prepare` ids + fp16 predictions for the first recorded frames (scripts/dump-snake-prompts.py). */
export async function readPromptCases(): Promise<PromptCase[]> {
  const { readFile } = await import("node:fs/promises");
  return JSON.parse(await readFile(new URL("./fixtures/snake-prompts.json", import.meta.url), "utf8")).cases;
}
