/**
 * Regression bench for `sleepWhileWaiting` (Node/Bun): are the first forwards
 * at a small shape right after a large one as fast as steady state? Before
 * 0.5.1 the readback-wait estimate was keyed by dispatch count only, so B=3
 * L=16 after B=3 L=512 took 594 ms instead of 31 ms and needed ~15 calls to
 * recover. Laya English DecisionModel, synthetic token ids (as bench/grid.ts).
 *
 *   [DTYPE=f16] [SMALL=3x16] [LARGE=3x512] [N=5] [TOL=0.2] [COOL=5] [BASE=1] \
 *     node --conditions=source bench/sleep-after-large.ts
 *
 * BASE=1 runs main's unmodified backend from ../.base/src instead (see
 * grid.ts), for a before/after. Exits 1 if any of the first N small forwards
 * after the large ones exceeds steady state by more than TOL. Take ~/gpu.lock
 * around it; COOL seconds of idle precede each phase (fanless M2).
 */
import { readFile } from "node:fs/promises";
import { readSafetensors } from "@johnhenry/math-plus-safetensors";
import { safetensorsWeights } from "@johnhenry/modernbert";
import { loadReal } from "@johnhenry/laya-fixtures";
import type { AgentConfig, Batch } from "@johnhenry/laya-core";
import type { Backend } from "@johnhenry/tensor-backend";
import { loadDecisionModel } from "../../laya/src/model.ts";

const env = process.env;
const dtype = (env.DTYPE ?? "f16") as "f16" | "f32";
const [sB, sL] = (env.SMALL ?? "3x16").split("x").map(Number) as [number, number];
const [lB, lL] = (env.LARGE ?? "3x512").split("x").map(Number) as [number, number];
const n = Number(env.N ?? 5);
const tol = Number(env.TOL ?? 0.2);
const cool = Number(env.COOL ?? 5);

const fx = await loadReal("english");
const encoderConfig = JSON.parse(await readFile(`${fx.model_dir}/encoder/config.json`, "utf8"));
const file = readSafetensors(await readFile(`${fx.model_dir}/model.safetensors`));
const w = env.BASE === "1" ? await import(new URL("../.base/src/index.ts", import.meta.url).href) : await import("../src/index.ts");
const backend = (await w.createWebGpuBackend()) as unknown as Backend;
const model = await loadDecisionModel(backend, { encoderConfig, agentConfig: fx.config as AgentConfig, weights: safetensorsWeights(file), dtype });

function batchOf(B: number, L: number): Batch {
  const M = 4;
  const ids = new Int32Array(B * L);
  let s = 12345;
  for (let i = 0; i < ids.length; i++) ids[i] = 1000 + ((s = (s * 1103515245 + 12345) >>> 0) % 30000);
  const pos = new Int32Array(B * M);
  for (let r = 0; r < B; r++) for (let m = 0; m < M; m++) pos[r * M + m] = 1 + Math.floor(((m + 1) * (L - 2)) / (M + 1));
  return {
    size: B, length: L, markerCount: M, inputIds: ids,
    attentionMask: new Uint8Array(B * L).fill(1),
    markerPos: pos, markerMask: new Uint8Array(B * M).fill(1),
    qtype: new Int32Array(B),
  };
}
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const time = async (b: Batch) => { const t = performance.now(); await model.forward(b); return performance.now() - t; };
const small = batchOf(sB, sL);
const large = batchOf(lB, lL);

// Steady state at the small shape, never having seen the large one.
await sleep(cool * 1000);
for (let i = 0; i < 3; i++) await time(small);
const ss: number[] = [];
for (let i = 0; i < 9; i++) ss.push(await time(small));
ss.sort((a, b) => a - b);
const steady = ss[4]!;

// Large shape (also records its wait estimate), then small immediately.
await sleep(cool * 1000);
for (let i = 0; i < 3; i++) await time(large);
const after: number[] = [];
for (let i = 0; i < n; i++) after.push(await time(small));

const worst = Math.max(...after);
const ok = worst <= steady * (1 + tol);
const runtime = (globalThis as { Bun?: unknown }).Bun ? "bun" : "node";
console.log(
  `${env.BASE === "1" ? "base" : "src"}/${runtime} ${dtype}: B=${sB} L=${sL} steady ${steady.toFixed(1)} ms; ` +
  `after B=${lB} L=${lL}: ${after.map((v) => v.toFixed(1)).join(", ")} ms; ` +
  `worst ${(worst / steady).toFixed(2)}x steady -> ${ok ? "PASS" : "FAIL"} (tol ${Math.round(tol * 100)}%)`,
);
model.dispose();
(backend as { destroy?: () => void }).destroy?.();
process.exitCode = ok ? 0 : 1;
