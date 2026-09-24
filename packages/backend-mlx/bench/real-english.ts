/**
 * Real English checkpoint (ModernBERT-large 421M + decision heads) on
 * backend-mlx, vs the laya-mlx fp32 fixture. Bench-only: imports the
 * Workstream C model (@johnhenry/laya src) without depending on it.
 *   node --conditions=source bench/real-english.ts [f32|f16|bf16 ...]
 */
import { readFile } from "node:fs/promises";
import { loadJson, loadReal } from "@johnhenry/laya-fixtures";
import { readSafetensors } from "@johnhenry/math-plus-safetensors";
import { safetensorsWeights } from "@johnhenry/modernbert";
import type { AgentConfig, PreparedItem } from "@johnhenry/laya-core";
import { loadDecisionModel } from "../../laya/src/model.ts";
import { collateItems, errStats } from "../../laya/test/helpers.ts";
import { createMlxBackend } from "../src/index.ts";

const argv = (globalThis as { process?: { argv: string[] } }).process!.argv.slice(2);
const dtypes = (argv.length ? argv : ["f32", "f16"]) as ("f32" | "f16" | "bf16")[];
const fx = await loadReal("english");
const encoderConfig = JSON.parse(await readFile(`${fx.model_dir}/encoder/config.json`, "utf8")) as Record<string, unknown>;
const tok = await loadJson<{ special: Record<string, [string, number]> }>("tables", "tokenizer-english.json");
const padId = tok.special.pad_token![1];
const tRead = performance.now();
const file = readSafetensors(await readFile(`${fx.model_dir}/model.safetensors`));
console.log(`safetensors read ${(performance.now() - tRead).toFixed(0)} ms`);
const argmax = (xs: ArrayLike<number>, n: number) => {
  let best = 0;
  for (let i = 1; i < n; i++) if (xs[i]! > xs[best]!) best = i;
  return best;
};
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[xs.length >> 1]!;

const b = createMlxBackend();
console.log(`runtime ${b.info.runtime} | ${b.info.mlxcAbi}`);
for (const dtype of dtypes) {
  const t0 = performance.now();
  const model = await loadDecisionModel(b, { encoderConfig, agentConfig: fx.config as AgentConfig, weights: safetensorsWeights(file), dtype });
  b.flush(); // weights resident
  const loadMs = performance.now() - t0;
  // single short question latency (B=1, the shortest item), measured first
  const items = fx.cases.flatMap((c) => c.items as PreparedItem[]);
  const short = items.reduce((a, c) => (c.ids.length < a.ids.length ? c : a));
  const batch = collateItems([short], padId);
  for (let i = 0; i < 5; i++) await model.forward(batch);
  const ts: number[] = [];
  for (let i = 0; i < 30; i++) {
    const s = performance.now();
    await model.forward(batch);
    ts.push(performance.now() - s);
  }
  let logitAbs = 0, actRel = 0, agree = 0, total = 0;
  const tAll = performance.now();
  for (const c of fx.cases) {
    for (let i = 0; i < c.items.length; i++) {
      const out = await model.forward(collateItems([c.items[i]! as PreparedItem], padId));
      const want = c.outputs[i]!, k = want.logits.length;
      const got = out.logits.subarray(0, k);
      logitAbs = Math.max(logitAbs, errStats(got, want.logits, 0, 0).maxAbs);
      actRel = Math.max(actRel, errStats(out.act, want.act, 0, 0).maxRel);
      if (argmax(got, k) === argmax(want.logits, k)) agree++;
      total++;
    }
  }
  const allMs = performance.now() - tAll;
  console.log(
    `[${dtype}] load+upload ${loadMs.toFixed(0)} ms | ${total} questions in ${allMs.toFixed(0)} ms | argmax ${agree}/${total} | ` +
      `max|Δlogit| ${logitAbs.toExponential(3)} | max rel Δact ${actRel.toExponential(2)} | ` +
      `short question (${short.ids.length} tok) P50 ${median(ts).toFixed(2)} ms, min ${Math.min(...ts).toFixed(2)} ms`,
  );
  model.dispose();
  console.log(`  live tensors after dispose: ${b.liveTensors()}, MLX active ${(b.memory().active / 2 ** 20).toFixed(0)} MiB`);
}
