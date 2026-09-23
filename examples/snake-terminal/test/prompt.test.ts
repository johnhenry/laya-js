/**
 * Prompt regression against Python's recorded run (snake-showcase.jsonl):
 * rebuilding each frame's prompt and tokenizing it with the checkpoint's
 * tokenizer (laya-core `prepare`) gives exactly Python's `input_tokens`.
 * Skips when the multilingual checkpoint is not in the local HF cache.
 */
import assert from "node:assert/strict";
import { join } from "node:path";
import { prepare } from "@johnhenry/laya-core";
import { loadTokenizerFromDir } from "@johnhenry/laya-core/node";
import { snapshot } from "@johnhenry/hf-cache";
import { DEFAULT_MODEL, SnakeGame, buildPrompt } from "../src/core/index.ts";
import { makeTest, readFrames, readPromptCases } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const test = makeTest(bun);

let dir: string | undefined;
try {
  dir = (await snapshot(DEFAULT_MODEL, { offline: true })).dir;
} catch {}

test("JS state + questions equal Python's exactly on the first 40 recorded frames", async () => {
  const { frames } = await readFrames();
  const cases = await readPromptCases();
  cases.forEach((c, i) => {
    const p = buildPrompt(SnakeGame.fromSnapshot(frames[i]!.game), "compact");
    assert.equal(frames[i]!.game.ticks, c.tick);
    assert.equal(p.state, c.state);
    assert.equal(JSON.stringify(p.questions), JSON.stringify(c.questions)); // key order too
  });
});

test(
  "prepared ids/markers/qtype equal Python Agent.prepare on the first 40 recorded frames",
  async () => {
    const tok = await loadTokenizerFromDir(join(dir!, "tokenizer"));
    const { frames } = await readFrames();
    const cases = await readPromptCases();
    cases.forEach((c, i) => {
      const p = buildPrompt(SnakeGame.fromSnapshot(frames[i]!.game), "compact");
      const { items } = prepare(tok, p.state, p.questions as never);
      assert.deepEqual(items, c.items, `tick ${c.tick}`);
    });
  },
  { skip: dir ? false : `${DEFAULT_MODEL} not in the local Hugging Face cache` },
);

test(
  "compact prompts tokenize to Python's recorded input_tokens on every recorded frame",
  async () => {
    const tok = await loadTokenizerFromDir(join(dir!, "tokenizer"));
    const { frames } = await readFrames();
    for (const f of frames) {
      const p = buildPrompt(SnakeGame.fromSnapshot(f.game), "compact");
      const { items, internal } = prepare(tok, p.state, p.questions as never);
      assert.deepEqual(internal.map((q) => q.id), ["move", "risk", "food"]);
      assert.deepEqual(items.map((i) => i.qtype), [0, 2, 2]);
      assert.equal(items[0]!.markers.length, 4);
      const tokens = items.reduce((n, it) => n + it.ids.length, 0);
      assert.equal(tokens, f.decision.input_tokens, `tick ${f.game.ticks}`);
    }
  },
  { skip: dir ? false : `${DEFAULT_MODEL} not in the local Hugging Face cache` },
);
