/**
 * Latency + GPU-time profile of one short English question (B=1) through the
 * Laya DecisionModel in f16. Workspace-only (imports @johnhenry/laya's model by path).
 *   node --conditions=source bench/profile.ts
 */
import { readFile } from "node:fs/promises";
import { readSafetensors } from "@johnhenry/math-plus-safetensors";
import { safetensorsWeights } from "@johnhenry/modernbert";
import { loadJson, loadReal } from "@johnhenry/laya-fixtures";
import type { AgentConfig } from "@johnhenry/laya-core";
import { loadDecisionModel } from "../../laya/src/model.ts";
import { collateItems } from "../../laya/test/helpers.ts";
import { createWebGpuBackend } from "../src/index.ts";
const b = await createWebGpuBackend({ profiling: true, ...(process.env.BATCH ? { maxBatch: Number(process.env.BATCH) } : {}) });
const fx = await loadReal("english");
const encoderConfig = JSON.parse(await readFile(`${fx.model_dir}/encoder/config.json`, "utf8"));
const tok = await loadJson<{ special: Record<string, [string, number]> }>("tables", "tokenizer-english.json");
const file = readSafetensors(await readFile(`${fx.model_dir}/model.safetensors`));
const model = loadDecisionModel(b, { encoderConfig, agentConfig: fx.config as AgentConfig, weights: safetensorsWeights(file), dtype: "f16" });
const items = fx.cases.flatMap((c) => c.items);
const short = items.reduce((a, c) => (c.ids.length < a.ids.length ? c : a));
const batch = collateItems([short], tok.special.pad_token![1]);
for (let i = 0; i < 5; i++) await model.forward(batch);
await b.sync();
let enc = 0, gpu = 0, tot = 0; const n = 20; const d0 = b.rt.stats.dispatches, s0 = b.rt.stats.submits;
for (let i = 0; i < n; i++) {
  const t0 = performance.now();
  const { logits, act } = model.forwardTensors(batch);
  const t1 = performance.now();
  b.flush();
  await b.sync();
  const t2 = performance.now();
  await b.read(logits); await b.read(act); b.dispose(logits); b.dispose(act);
  enc += t1 - t0; gpu += t2 - t1; tot += performance.now() - t0;
}
console.log(`L=${short.ids.length}: encode ${(enc / n).toFixed(1)} ms, gpu wait after encode ${(gpu / n).toFixed(1)} ms, total ${(tot / n).toFixed(1)} ms; ${(b.rt.stats.dispatches - d0) / n} dispatches, ${(b.rt.stats.submits - s0) / n} submits per forward`);
b.rt.startProfiling();
for (let i = 0; i < 5; i++) await model.forward(batch);
const prof = await b.rt.stopProfiling();
const totalMs = prof.reduce((a, p) => a + p.ms, 0) / 5;
console.log(`GPU time per forward (profiled, per-dispatch passes): ${totalMs.toFixed(2)} ms`);
for (const p of prof.slice(0, 15)) console.log(`  ${(p.ms / 5).toFixed(3).padStart(8)} ms  ${String(p.count / 5).padStart(4)}×  ${p.kernel}`);
model.dispose(); b.destroy();
