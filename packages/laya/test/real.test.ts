/**
 * Opt-in DecisionModel parity on the three published checkpoints (fp16 weights → f32 CPU):
 *   LAYA_REAL_CPU_FULL=1 [LAYA_REAL_MODELS=english,multilingual,typed-decisions] npm test -w @johnhenry/laya
 * Needs the local HF snapshots recorded in the fixtures (`model_dir`). Slow:
 * 7-19 minutes per checkpoint on one Apple M2 core (see README).
 */
import * as nodeTest from "node:test";
// Bun 1.2's node:test shim only registers tests from the first file of a run.
// @ts-ignore -- bun types are not installed
const bunTest: unknown = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const { test } = (bunTest ?? nodeTest) as Pick<typeof nodeTest, "test">;
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { createCpuBackend } from "@johnhenry/backend-cpu";
import { MODELS, loadJson, loadReal, type ModelName, type RealFixture } from "@johnhenry/laya-fixtures";
import { readSafetensors } from "@johnhenry/math-plus-safetensors";
import { safetensorsWeights } from "@johnhenry/modernbert";
import type { AgentConfig, PreparedItem } from "@johnhenry/laya-core";
import { loadDecisionModel } from "../src/model.ts";
import { collateItems, errStats } from "./helpers.ts";

const env = (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env ?? {};
const enabled = env.LAYA_REAL_CPU_FULL === "1";
const selected = (env.LAYA_REAL_MODELS ?? MODELS.join(",")).split(",");
/** Bun's default per-test timeout is 5 s; node:test has none. */
const slow = (name: string, fn: () => Promise<void>) =>
  bunTest ? (test as unknown as (n: string, f: () => Promise<void>, ms: number) => void)(name, fn, 4 * 3600_000) : test(name, fn);

/** Groups item indices of one case into batches of similar length (≤ 8 rows, ≤ 15% padding). */
function batches(items: readonly PreparedItem[]): number[][] {
  const order = items.map((_, i) => i).sort((a, b) => items[a]!.ids.length - items[b]!.ids.length);
  const out: number[][] = [];
  for (const i of order) {
    const cur = out[out.length - 1];
    if (cur && cur.length < 8 && items[i]!.ids.length <= 1.15 * items[cur[0]!]!.ids.length) cur.push(i);
    else out.push([i]);
  }
  return out;
}

const argmax = (xs: ArrayLike<number>, n: number) => {
  let best = 0;
  for (let i = 1; i < n; i++) if (xs[i]! > xs[best]!) best = i;
  return best;
};

for (const m of MODELS) {
  const name = `real ${m}: 63 questions, CPU f32 vs MLX fp32 fixture (logits ≤ 1e-3, identical argmax)`;
  let fixture: RealFixture | undefined;
  if (enabled && selected.includes(m)) fixture = await loadReal(m as ModelName);
  if (!fixture || !existsSync(`${fixture.model_dir}/model.safetensors`)) {
    test.skip(`${name} [${fixture ? `checkpoint not in the HF cache: ${fixture.repo}` : "opt-in: LAYA_REAL_CPU_FULL=1 (7-19 min per checkpoint)"}]`, () => {});
    continue;
  }
  const fx = fixture;
  slow(name, async () => {
    const t0 = performance.now();
    const encoderConfig = JSON.parse(await readFile(`${fx.model_dir}/encoder/config.json`, "utf8")) as Record<string, unknown>;
    const tok = await loadJson<{ special: Record<string, [string, number]> }>("tables", `tokenizer-${m}.json`);
    const padId = tok.special.pad_token![1];
    const backend = createCpuBackend();
    const file = readSafetensors(await readFile(`${fx.model_dir}/model.safetensors`));
    const model = loadDecisionModel(backend, { encoderConfig, agentConfig: fx.config as AgentConfig, weights: safetensorsWeights(file) });
    const loadS = (performance.now() - t0) / 1000;

    let logitAbs = 0, actAbs = 0, actRel = 0, agree = 0, total = 0, tokens = 0;
    const lines: string[] = [];
    const tAll = performance.now();
    for (const c of fx.cases) {
      const tc = performance.now();
      let caseLogit = 0;
      for (const idx of batches(c.items)) {
        const chunk = idx.map((i) => c.items[i]!);
        const batch = collateItems(chunk, padId);
        const out = await model.forward(batch);
        idx.forEach((i, r) => {
          const want = c.outputs[i]!, k = want.logits.length;
          const got = out.logits.subarray(r * batch.markerCount, r * batch.markerCount + k);
          const gotAct = out.act.subarray(r * out.nAct, (r + 1) * out.nAct);
          const sl = errStats(got, want.logits, 0, 0), sa = errStats(gotAct, want.act, 0, 0);
          logitAbs = Math.max(logitAbs, sl.maxAbs);
          caseLogit = Math.max(caseLogit, sl.maxAbs);
          actAbs = Math.max(actAbs, sa.maxAbs);
          actRel = Math.max(actRel, sa.maxRel);
          if (argmax(got, k) === argmax(want.logits, k)) agree++;
          total++;
          tokens += c.items[i]!.ids.length;
        });
      }
      const secs = (performance.now() - tc) / 1000;
      lines.push(`${c.case.padEnd(24)} ${String(c.items.length).padStart(2)} q  ${secs.toFixed(1).padStart(6)} s  max|Δlogit| ${caseLogit.toExponential(2)}`);
    }
    const runS = (performance.now() - tAll) / 1000;
    console.log(
      `[${m}] load ${loadS.toFixed(1)} s, forward ${runS.toFixed(1)} s for ${total} questions / ${tokens} tokens ` +
        `(${(runS / fx.cases.length).toFixed(1)} s per case)\n  argmax ${agree}/${total}, max|Δlogit| ${logitAbs.toExponential(3)}, ` +
        `max|Δact| ${actAbs.toExponential(3)} (rel ${actRel.toExponential(2)})\n  ` + lines.join("\n  "),
    );
    model.dispose();
    assert.equal(total, 63);
    assert.equal(agree, total, "argmax agreement");
    assert.ok(logitAbs <= 1e-3, `max logit error ${logitAbs}`);
    assert.ok(actRel <= 1e-4 || actAbs <= 1e-3, `max act error ${actAbs} (rel ${actRel})`);
  });
}

test.skip("real checkpoints: mean-pooled `embeddings` fixture — needs a JS tokenizer (texts are stored without ids); see README", () => {});
