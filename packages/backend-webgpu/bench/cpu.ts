/**
 * CPU-side cost of encoding one Laya English forward (f16) on WebGPU:
 * time spent in forwardTensors (enqueue + auto-submits) vs the whole
 * forward, dispatches and submits per forward.
 *   [L=93] [B=1] node --conditions=source bench/cpu.ts   (or bun)
 */
import { readFile } from "node:fs/promises";
import { readSafetensors } from "@johnhenry/math-plus-safetensors";
import { safetensorsWeights } from "@johnhenry/modernbert";
import { loadReal } from "@johnhenry/laya-fixtures";
import type { AgentConfig, Batch } from "@johnhenry/laya-core";
import { loadDecisionModel } from "../../laya/src/model.ts";
import { createWebGpuBackend } from "../src/index.ts";

const L = Number(process.env.L ?? 93), B = Number(process.env.B ?? 1);
const b = await createWebGpuBackend(process.env.SLEEP ? { sleepWhileWaiting: process.env.SLEEP === "1" } : {});
const fx = await loadReal("english");
const encoderConfig = JSON.parse(await readFile(`${fx.model_dir}/encoder/config.json`, "utf8"));
const file = readSafetensors(await readFile(`${fx.model_dir}/model.safetensors`));
const model = loadDecisionModel(b, { encoderConfig, agentConfig: fx.config as AgentConfig, weights: safetensorsWeights(file), dtype: "f16" });
const M = 4;
const batch: Batch = {
  size: B, length: L, markerCount: M, inputIds: Int32Array.from({ length: B * L }, (_, i) => 1000 + ((i * 7919) % 30000)),
  attentionMask: new Uint8Array(B * L).fill(1), markerPos: Int32Array.from({ length: B * M }, (_, i) => 1 + (i % M) * 5),
  markerMask: new Uint8Array(B * M).fill(1), qtype: new Int32Array(B),
};
for (let i = 0; i < 3; i++) await model.forward(batch);
const n = Number(process.env.N ?? 20);
let enc = 0, tot = 0, wait = 0;
const d0 = b.rt.stats.dispatches, s0 = b.rt.stats.submits;
const c0 = process.cpuUsage();
for (let i = 0; i < n; i++) {
  const t0 = performance.now();
  const r = model.forwardTensors(batch);
  const t1 = performance.now();
  await model.readOutputs(r);
  const t2 = performance.now();
  enc += t1 - t0; wait += t2 - t1; tot += t2 - t0;
}
const disp = (b.rt.stats.dispatches - d0) / n;
const cu = process.cpuUsage(c0);
console.log(`process CPU: ${((cu.user + cu.system) / 1000 / n).toFixed(1)} ms per forward (${(((cu.user + cu.system) / 1000 / tot) * 100).toFixed(0)}% of wall)`);
console.log(`${(globalThis as { Bun?: unknown }).Bun ? "bun" : "node"} L=${L} B=${B}: encode ${(enc / n).toFixed(2)} ms (${((1000 * enc) / n / disp).toFixed(1)} µs/dispatch), read-wait ${(wait / n).toFixed(2)} ms, total ${(tot / n).toFixed(2)} ms; ${disp} dispatches, ${(b.rt.stats.submits - s0) / n} submits`);
model.dispose();
b.destroy();
