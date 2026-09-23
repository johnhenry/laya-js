/**
 * Laya English DecisionModel latency over an (L, B) grid, synthetic token ids.
 * Workspace-only (imports @johnhenry/laya's model by path).
 *
 *   [BACKEND=webgpu,mlx] [DTYPE=f16] [LS=16,33,64,93,128,256,512] [BS=1,3,16] [COOL=5] \
 *     node --conditions=source bench/grid.ts          (or: bun --conditions=source bench/grid.ts)
 *
 * Thermal note: the reference machine (MacBook Air M2, fanless) throttles the
 * GPU to ~35% after ~10 s of sustained load and recovers within ~5 s idle. So
 * each cell idles COOL seconds, then runs 1 warmup and timed forwards
 * (upload + forward + read of logits/act) for at most ~1 s (3..30 runs), and
 * backends are interleaved per cell in one process. Prints the median (ms)
 * as Markdown tables plus JSON. PROFILE=1 (webgpu) also prints the GPU time
 * per kernel for each cell.
 */
import { readFile } from "node:fs/promises";
import { readSafetensors } from "@johnhenry/math-plus-safetensors";
import { safetensorsWeights } from "@johnhenry/modernbert";
import { loadReal } from "@johnhenry/laya-fixtures";
import type { AgentConfig, Batch } from "@johnhenry/laya-core";
import type { Backend } from "@johnhenry/tensor-backend";
import { loadDecisionModel } from "../../laya/src/model.ts";

const env = process.env;
const which = (env.BACKEND ?? "webgpu").split(",");
const dtype = (env.DTYPE ?? "f16") as "f16" | "f32";
const Ls = (env.LS ?? "16,33,64,93,128,256,512").split(",").map(Number);
const Bs = (env.BS ?? "1,3,16").split(",").map(Number);
const profile = env.PROFILE === "1";
const cool = Number(env.COOL ?? 5);

const fx = await loadReal("english");
const encoderConfig = JSON.parse(await readFile(`${fx.model_dir}/encoder/config.json`, "utf8"));
const file = readSafetensors(await readFile(`${fx.model_dir}/model.safetensors`));
const setups: { name: string; backend: Backend; model: ReturnType<typeof loadDecisionModel> }[] = [];
for (const name of which) {
  let backend: Backend;
  if (name === "mlx") {
    const m = await import("@johnhenry/backend-mlx");
    backend = m.createMlxBackend() as unknown as Backend;
  } else {
    const w = await import("../src/index.ts");
    backend = (await w.createWebGpuBackend({ profiling: profile })) as unknown as Backend;
  }
  const model = loadDecisionModel(backend, { encoderConfig, agentConfig: fx.config as AgentConfig, weights: safetensorsWeights(file), dtype });
  setups.push({ name, backend, model });
}

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

const results: Record<string, Record<string, number>> = {};
const runtime = (globalThis as { Bun?: unknown }).Bun ? "bun" : "node";
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
for (const B of Bs) {
  for (const L of Ls) {
    const batch = batchOf(B, L);
    for (const { name, backend, model } of setups) {
      if (cool) await sleep(cool * 1000);
      await model.forward(batch);
      const ts: number[] = [];
      const t0 = performance.now();
      while (ts.length < 30 && (ts.length < 3 || performance.now() - t0 < 1000)) {
        const a = performance.now();
        await model.forward(batch);
        ts.push(performance.now() - a);
      }
      ts.sort((a, b) => a - b);
      const med = ts[Math.floor(ts.length / 2)]!;
      (results[name] ??= {})[`${B}x${L}`] = med;
      console.error(`${name}/${runtime} ${dtype} B=${B} L=${L}: median ${med.toFixed(1)} ms (min ${ts[0]!.toFixed(1)}, n=${ts.length})`);
      if (profile && name === "webgpu") {
        const rt = (backend as unknown as { rt: { startProfiling(): void; stopProfiling(): Promise<{ kernel: string; ms: number; count: number }[]> } }).rt;
        rt.startProfiling();
        await model.forward(batch);
        const prof = await rt.stopProfiling();
        const tot = prof.reduce((a, p) => a + p.ms, 0);
        console.error(`  GPU (profiled) ${tot.toFixed(2)} ms`);
        for (const p of prof.slice(0, 8)) console.error(`    ${p.ms.toFixed(3).padStart(8)} ms ${String(p.count).padStart(4)}×  ${p.kernel}`);
      }
    }
  }
}
for (const { name } of setups) {
  console.log(`| ${name}/${runtime} ${dtype} | ${Ls.map((L) => `L=${L}`).join(" | ")} |`);
  console.log(`|---|${Ls.map(() => "---:").join("|")}|`);
  for (const B of Bs) console.log(`| B=${B} | ${Ls.map((L) => results[name]![`${B}x${L}`]!.toFixed(1)).join(" | ")} |`);
}
console.log(JSON.stringify({ runtime, dtype, results }));
for (const { backend, model } of setups) {
  model.dispose();
  (backend as { destroy?: () => void }).destroy?.();
}
