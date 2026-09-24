/**
 * Quantized checkpoints on the device vs dequantize-on-load vs fp16:
 * device memory, single-question latency and 16-question throughput, per
 * checkpoint × variant × backend (f16). Workspace-only bench.
 *
 *   QDIR=<dir with <model>-q8 / <model>-q4 from `laya quantize`> \
 *   [MODELS=english,multilingual] [BACKENDS=mlx,webgpu] [VARIANTS=fp16,q8-dequantize,q8-device,q4-dequantize,q4-device] \
 *   [COOL=20] node --conditions=source packages/laya/bench/quantized.ts
 *
 * Every cell runs in a fresh child process (clean allocator statistics:
 * MLX `memory()` active after load and peak over the process; WebGPU
 * `rt.stats.liveBytes` after load, and live + pooled bytes after the runs,
 * i.e. every buffer the backend holds). Thermal note: the reference machine
 * (MacBook Air M2) is fanless and throttles after ~10 s of sustained GPU
 * load, so each cell starts after COOL seconds idle, the timed loops are
 * capped at ~1.5 s each, and the throughput loop starts after a 5 s pause.
 * Take ~/gpu.lock around the run (AGENTS.md rule 6).
 */
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { MODELS, REPOS, loadReal, type ModelName } from "@johnhenry/laya-fixtures";
import type { Questions } from "@johnhenry/laya-core";
import { load } from "../src/index.ts";

const env = process.env;
const sleep = (s: number) => new Promise((r) => setTimeout(r, s * 1000));
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[xs.length >> 1]!;
const MiB = (b: number) => b / 2 ** 20;

interface Cell {
  model: ModelName;
  variant: string;
  backend: "mlx" | "webgpu";
}
interface Result extends Cell {
  loadMs: number;
  quantizedOnDevice: boolean;
  /** MLX active / WebGPU liveBytes right after load (weights resident). */
  memLoad: number;
  /** MLX peak over the process / WebGPU live + pooled after the runs. */
  memPeak: number;
  p50: number;
  runs1: number;
  batchMs: number;
  qps: number;
  runs16: number;
}

async function runCell(c: Cell): Promise<Result> {
  const fx = await loadReal(c.model);
  const [q, mode] = c.variant === "fp16" ? ["", ""] : c.variant.split("-");
  const path = q ? `${env.QDIR}/${c.model}-${q}` : REPOS[c.model];
  const t0 = performance.now();
  const agent = await load(path, { offline: true, backend: c.backend, dtype: "f16", warn: () => {}, ...(mode ? { quantized: mode as "device" | "dequantize" } : {}) });
  const b = agent.backend as unknown as { flush?(): void; sync?(): Promise<void>; memory?(): { active: number; peak: number }; rt?: { stats: { liveBytes: number; pooledBytes: number } } };
  b.flush?.();
  await b.sync?.();
  const loadMs = performance.now() - t0;
  const memLoad = b.memory ? b.memory().active : b.rt!.stats.liveBytes;
  const cs = fx.cases[0]!;
  const qs = Object.entries(cs.questions as Record<string, unknown>);
  const one = Object.fromEntries(qs.slice(0, 1)) as unknown as Questions;
  const sixteen = Object.fromEntries(Array.from({ length: 16 }, (_, i) => [`q${i}`, qs[i % qs.length]![1]])) as unknown as Questions;
  // single question: 3 warmups, then up to 30 runs or ~1.5 s
  for (let i = 0; i < 3; i++) await agent.predict(cs.state as never, one);
  const ts: number[] = [];
  const tEnd = performance.now() + 1500;
  while (ts.length < 30 && (ts.length < 5 || performance.now() < tEnd)) {
    const s = performance.now();
    await agent.predict(cs.state as never, one);
    ts.push(performance.now() - s);
  }
  await sleep(5);
  // 16 questions in one call (one batch of 16): 1 warmup, then up to 10 runs or ~1.5 s
  await agent.predict(cs.state as never, sixteen);
  const tb: number[] = [];
  const tEnd2 = performance.now() + 1500;
  while (tb.length < 10 && (tb.length < 3 || performance.now() < tEnd2)) {
    const s = performance.now();
    await agent.predict(cs.state as never, sixteen);
    tb.push(performance.now() - s);
  }
  const memPeak = b.memory ? b.memory().peak : b.rt!.stats.liveBytes + b.rt!.stats.pooledBytes;
  const quantizedOnDevice = agent.model.quantizedOnDevice;
  agent.dispose();
  const batchMs = median(tb);
  return { ...c, loadMs, quantizedOnDevice, memLoad, memPeak, p50: median(ts), runs1: ts.length, batchMs, qps: 16000 / batchMs, runs16: tb.length };
}

const cellArg = process.argv.indexOf("--cell");
if (cellArg >= 0) {
  const [model, variant, backend] = process.argv[cellArg + 1]!.split(",") as [ModelName, string, "mlx" | "webgpu"];
  const r = await runCell({ model, variant, backend });
  console.log("RESULT " + JSON.stringify(r));
  process.exit(0);
}

const models = (env.MODELS ?? "english,multilingual").split(",") as ModelName[];
for (const m of models) if (!MODELS.includes(m)) throw new Error(`unknown model ${m}`);
const backends = (env.BACKENDS ?? "mlx,webgpu").split(",") as ("mlx" | "webgpu")[];
const variants = (env.VARIANTS ?? "fp16,q8-dequantize,q8-device,q4-dequantize,q4-device").split(",");
if (variants.some((v) => v !== "fp16") && !env.QDIR) throw new Error("QDIR=<dir with <model>-q8 / <model>-q4> is required");
const cool = Number(env.COOL ?? 20);
const self = fileURLToPath(import.meta.url);
const results: Result[] = [];
for (const model of models)
  for (const backend of backends)
    for (const variant of variants) {
      await sleep(cool);
      const p = spawnSync(process.execPath, [...process.execArgv, self, "--cell", `${model},${variant},${backend}`], { encoding: "utf8", env });
      const line = p.stdout.split("\n").find((l) => l.startsWith("RESULT "));
      if (!line) {
        console.error(`${model} ${variant} ${backend} failed:\n${p.stderr}`);
        continue;
      }
      const r = JSON.parse(line.slice(7)) as Result;
      results.push(r);
      console.error(`${model} ${backend} ${variant}: load ${r.loadMs.toFixed(0)} ms, mem ${MiB(r.memLoad).toFixed(0)}/${MiB(r.memPeak).toFixed(0)} MiB, P50 ${r.p50.toFixed(1)} ms, ${r.qps.toFixed(1)} q/s`);
    }

console.log("\n| checkpoint | backend | weights | on device | device memory after load | peak | load | 1 question P50 | 16 questions |");
console.log("|---|---|---|---|---:|---:|---:|---:|---:|");
for (const r of results) {
  const fp = results.find((x) => x.model === r.model && x.backend === r.backend && x.variant === "fp16");
  const rel = fp && r !== fp ? ` (${((100 * r.memLoad) / fp.memLoad).toFixed(0)}%)` : "";
  console.log(
    `| ${r.model} | ${r.backend} f16 | ${r.variant} | ${r.quantizedOnDevice ? "quantized" : "float"} | ${MiB(r.memLoad).toFixed(0)} MiB${rel} | ${MiB(r.memPeak).toFixed(0)} MiB | ` +
      `${(r.loadMs / 1000).toFixed(1)} s | ${r.p50.toFixed(1)} ms | ${r.batchMs.toFixed(0)} ms (${r.qps.toFixed(1)} q/s) |`,
  );
}
console.log("\n" + JSON.stringify(results));
