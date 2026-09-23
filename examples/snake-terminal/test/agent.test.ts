/**
 * Real-model regression: `@johnhenry/laya` predictions for the snake prompts
 * of the first 20 recorded frames are within 0.02 of Python's fp16 values
 * (move probabilities + both noul answers), and the shielded decision agrees.
 *
 * Backends: LAYA_SNAKE_BACKENDS=mlx,webgpu (default "mlx"). Skips cleanly
 * without the checkpoint in the local HF cache, without Apple silicon (mlx)
 * or without a WebGPU adapter.
 */
import assert from "node:assert/strict";
import { snapshot } from "@johnhenry/hf-cache";
import { DEFAULT_MODEL, SnakeGame, buildPrompt, decisionFrom } from "../src/core/index.ts";
import { makeTest, readFrames, readPromptCases } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const test = makeTest(bun);

let cached = false;
try {
  await snapshot(DEFAULT_MODEL, { offline: true });
  cached = true;
} catch {}
const laya = (await import("@johnhenry/laya")) as Record<string, unknown>;
const hasLoad = typeof laya.load === "function";

async function webgpuAvailable(): Promise<boolean> {
  try {
    const { isWebGpuAvailable } = await import("@johnhenry/backend-webgpu");
    return await isWebGpuAvailable();
  } catch {
    return false;
  }
}

const backends = (process.env.LAYA_SNAKE_BACKENDS ?? "mlx").split(",").filter(Boolean);
for (const backend of backends) {
  const hw =
    backend === "mlx"
      ? process.platform === "darwin" && process.arch === "arm64"
      : backend === "webgpu"
        ? await webgpuAvailable()
        : true;
  const skip = !cached
    ? `${DEFAULT_MODEL} not in the local HF cache`
    : !hasLoad
      ? "@johnhenry/laya does not export load() yet"
      : !hw
        ? `${backend} unavailable on this machine`
        : false;
  test(
    `${backend} f16 predictions match Python fp16 within 0.02 on 20 recorded frames`,
    async () => {
      const { loadSnakeAgent } = await import("../src/load-agent.ts");
      const { agent } = await loadSnakeAgent(DEFAULT_MODEL, { backend: backend as "mlx", dtype: "f16", offline: true });
      try {
        const { frames } = await readFrames();
        const cases = (await readPromptCases()).slice(0, 20);
        let worst = 0;
        for (const [i, c] of cases.entries()) {
          const p = buildPrompt(SnakeGame.fromSnapshot(frames[i]!.game), "compact");
          const out = await agent.predict(p.state, p.questions);
          const got = out.answers.move!.probabilities!;
          for (const [k, v] of Object.entries(c.probabilities)) worst = Math.max(worst, Math.abs(got[k]! - v));
          worst = Math.max(worst, Math.abs(out.answers.risk!.noul! - c.risk_noul), Math.abs(out.answers.food!.noul! - c.food_noul));
          assert.equal(out.usage.input_tokens, frames[i]!.decision.input_tokens);
          assert.equal(decisionFrom(out, p, true).executed, frames[i]!.decision.executed, `tick ${c.tick}`);
        }
        console.log(`[${backend}] max |Δp| vs Python fp16 over ${cases.length} frames: ${worst.toFixed(4)}`);
        assert.ok(worst <= 0.02, `max |Δp| ${worst}`);
      } finally {
        await agent.dispose?.();
      }
    },
    { skip },
  );
}
