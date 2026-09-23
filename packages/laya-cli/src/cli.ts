/**
 * `laya predict` / `laya bench` (Node >= 24 or Bun). Port of the laya-mlx CLI
 * (`laya-mlx predict`) plus a benchmark that mirrors laya-mlx
 * `benchmarks/worker.py`'s end-to-end timings.
 */
import { existsSync, readFileSync } from "node:fs";
import { cpus, platform, arch, totalmem } from "node:os";
import { parseArgs } from "node:util";
import { loads } from "@johnhenry/pyjson";
import { load, type LayaAgent, type LoadOptions } from "@johnhenry/laya";
import { MLX_MODELS, Router } from "@johnhenry/laya-router";
import { dumpsIndent, pythonFloats } from "./format.ts";

export const VERSION = "0.0.0";
const DEFAULT_MODEL = "aac6fef/laya-mlx";

const USAGE = `usage: laya <command> [options]

commands:
  predict   answer questions about a state, print the result as JSON
  bench     P50/P95 latency of one short question and 50-question throughput

predict options:
  --model <id|path>        Hub repo or local checkpoint (default ${DEFAULT_MODEL})
  --state <json|@file|text> the state: JSON when it parses, @file to read a file, else plain text
  --state-file <file>      JSON state file (laya-mlx compatible)
  --questions <json|@file|file>  question definitions (JSON)
  --route                  pick the checkpoint with @johnhenry/laya-router (MLX exports);
                           --model then names a checkpoint (english|multilingual|typed-decisions or alias)
  --lang <code>, --task <name>  routing hints (with --route)

common options:
  --backend auto|mlx|webgpu|cpu   (default auto: mlx, then webgpu, then cpu)
  --dtype f16|f32          (default f16; cpu always computes in f32)
  --device gpu|cpu         MLX device (default gpu)
  --subfolder <dir>  --revision <rev>  --batch-size <n>  --offline

bench options:
  --iterations <n> (default 50)  --warmup <n> (default 5)  --batch-size <n> (default 64)  --json
`;

const OPTIONS = {
  model: { type: "string" },
  state: { type: "string" },
  "state-file": { type: "string" },
  questions: { type: "string" },
  route: { type: "boolean" },
  lang: { type: "string" },
  task: { type: "string" },
  backend: { type: "string" },
  dtype: { type: "string" },
  device: { type: "string" },
  subfolder: { type: "string" },
  revision: { type: "string" },
  "batch-size": { type: "string" },
  offline: { type: "boolean" },
  iterations: { type: "string" },
  warmup: { type: "string" },
  json: { type: "boolean" },
  help: { type: "boolean", short: "h" },
  version: { type: "boolean" },
} as const;

type Args = { [K in keyof typeof OPTIONS]?: (typeof OPTIONS)[K]["type"] extends "boolean" ? boolean : string };

export class UsageError extends Error {}

function readSource(value: string, what: string, { plainText = false } = {}): unknown {
  if (value.startsWith("@")) return loads(readFileSync(value.slice(1), "utf8"));
  try {
    return loads(value);
  } catch {
    if (!plainText && existsSync(value)) return loads(readFileSync(value, "utf8")); // laya-mlx: --questions <path>
    if (plainText) return value;
    throw new UsageError(`${what} is neither JSON, @file nor an existing file: ${value.slice(0, 80)}`);
  }
}

const int = (v: string | undefined, name: string, dflt: number): number => {
  if (v === undefined) return dflt;
  const n = Number(v);
  if (!Number.isInteger(n) || n < 1) throw new UsageError(`--${name} must be a positive integer`);
  return n;
};

function loadOptions(a: Args, batchDefault: number): LoadOptions {
  const dtype = a.dtype === undefined ? undefined : ({ f16: "f16", float16: "f16", f32: "f32", float32: "f32" } as const)[a.dtype as "f16"];
  if (a.dtype !== undefined && !dtype) throw new UsageError("--dtype must be f16 or f32");
  if (a.backend !== undefined && !["auto", "mlx", "webgpu", "cpu"].includes(a.backend)) throw new UsageError("--backend must be auto, mlx, webgpu or cpu");
  if (a.device !== undefined && !["gpu", "cpu", "metal"].includes(a.device)) throw new UsageError("--device must be gpu or cpu");
  return {
    ...(a.backend ? { backend: a.backend as LoadOptions["backend"] } : {}),
    ...(dtype ? { dtype } : {}),
    ...(a.device ? { device: a.device as "gpu" } : {}),
    ...(a.subfolder ? { subfolder: a.subfolder } : {}),
    ...(a.revision ? { revision: a.revision } : {}),
    ...(a.offline ? { offline: true } : {}),
    batchSize: int(a["batch-size"], "batch-size", batchDefault),
    warn: (m) => process.stderr.write(m + "\n"),
  };
}

async function predict(a: Args, out: (s: string) => void): Promise<void> {
  if ((a.state === undefined) === (a["state-file"] === undefined)) throw new UsageError("predict needs exactly one of --state or --state-file");
  if (a.questions === undefined) throw new UsageError("predict needs --questions");
  const state = a["state-file"] !== undefined ? loads(readFileSync(a["state-file"], "utf8")) : readSource(a.state!, "--state", { plainText: true });
  const questions = readSource(a.questions, "--questions") as Record<string, unknown>;
  const opts = loadOptions(a, 16);
  let result: unknown;
  if (a.route) {
    const router = new Router({ mlxRepos: true, loadOptions: opts as Record<string, unknown> });
    try {
      result = await router.predict(state, questions, { model: a.model ?? null, lang: a.lang ?? null, task: a.task ?? null });
    } finally {
      router.unload();
    }
  } else {
    const agent = await load(a.model ?? DEFAULT_MODEL, opts);
    try {
      result = await agent.predict(state as never, questions as never);
    } finally {
      agent.dispose();
    }
  }
  out(dumpsIndent(pythonFloats(result as { answers: Record<string, unknown> })) + "\n");
}

// ------------------------------------------------------------------ bench
/** laya-mlx examples/state.json and examples/questions.json (the benchmark workload). */
const BENCH_STATE = {
  from: "user@example.com",
  subject: "Duplicate charge on invoice #4411",
  body: "We were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
};
const BENCH_QUESTIONS = [
  {
    type: "choice",
    instructions: "Which department should handle this email?",
    criteria: { billing: "invoices, payments, refunds", technical: "bugs, outages, system errors", sales: "pricing, new contracts", other: "everything else" },
  },
  { type: "score", instructions: "How urgent is this request?", criteria: ["not urgent", "soon", "critical deadline or blocking issue"] },
  { type: "noul", instructions: "Does the customer ask for money back?" },
];
/** `benchmarks.common.workload(count)`: q0..q{n-1} cycling through the three example questions. */
export function workload(count: number): { state: typeof BENCH_STATE; questions: Record<string, unknown> } {
  return { state: BENCH_STATE, questions: Object.fromEntries(Array.from({ length: count }, (_, i) => [`q${i}`, BENCH_QUESTIONS[i % 3]])) };
}

export interface Timing {
  p50_ms: number;
  p95_ms: number;
  mean_ms: number;
  min_ms: number;
  max_ms: number;
}
/** numpy.median / numpy.percentile(…, 95) (linear interpolation). */
export function timing(samples: number[]): Timing {
  const s = [...samples].sort((x, y) => x - y);
  const pct = (q: number) => {
    const pos = (q / 100) * (s.length - 1);
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return s[lo]! + (s[hi]! - s[lo]!) * (pos - lo);
  };
  return { p50_ms: pct(50), p95_ms: pct(95), mean_ms: s.reduce((x, y) => x + y, 0) / s.length, min_ms: s[0]!, max_ms: s[s.length - 1]! };
}

async function measure(fn: () => Promise<unknown>, warmup: number, iterations: number): Promise<Timing> {
  for (let i = 0; i < warmup; i++) await fn();
  const samples: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    await fn(); // predict resolves after the outputs are read back: GPU work is complete
    samples.push(performance.now() - t0);
  }
  return timing(samples);
}

function backendInfo(agent: LayaAgent): Record<string, unknown> {
  const b = agent.backend as unknown as { name: string; device?: string; info?: unknown; adapterInfo?: Record<string, unknown>; memory?: () => { peak: number } };
  const info: Record<string, unknown> = { backend: b.name, dtype: agent.dtype };
  if (typeof b.device === "string") info.device = b.device;
  if (b.info) info.mlx = b.info;
  if (b.adapterInfo) {
    const { limits: _limits, features: _features, ...rest } = b.adapterInfo;
    info.adapter = rest;
  }
  return info;
}

async function bench(a: Args, out: (s: string) => void): Promise<void> {
  const iterations = int(a.iterations, "iterations", 50);
  const warmup = int(a.warmup, "warmup", 5);
  const opts = loadOptions(a, 64);
  const model = a.model ?? DEFAULT_MODEL;
  const t0 = performance.now();
  const agent = await load(model, opts);
  const loadS = (performance.now() - t0) / 1000;
  try {
    const env = {
      runtime: (globalThis as { Bun?: { version: string } }).Bun ? `bun ${(globalThis as unknown as { Bun: { version: string } }).Bun.version}` : `node ${process.version}`,
      platform: `${platform()}/${arch()}`,
      cpu: cpus()[0]?.model ?? "unknown",
      memory_gib: Math.round(totalmem() / 2 ** 30),
    };
    const short = workload(1);
    const many = workload(50);
    const shortResult = await agent.predict(short.state, short.questions as never);
    const shortT = await measure(() => agent.predict(short.state, short.questions as never), warmup, iterations);
    const manyT = await measure(() => agent.predict(many.state, many.questions as never), warmup, iterations);
    const peak = (agent.backend as unknown as { memory?: () => { peak: number } }).memory?.().peak;
    const report = {
      model,
      ...backendInfo(agent),
      batch_size: agent.batchSize,
      environment: env,
      load_seconds: loadS,
      short: { questions: 1, input_tokens: shortResult.usage.input_tokens, ...shortT },
      fifty: { questions: 50, ...manyT, questions_per_second: (50 * 1000) / manyT.mean_ms },
      ...(peak !== undefined ? { mlx_peak_mib: peak / 2 ** 20 } : {}),
    };
    if (a.json) {
      out(JSON.stringify(report, null, 2) + "\n");
      return;
    }
    const f = (x: number) => x.toFixed(2);
    out(
      [
        `model      ${model} (${agent.dtype}, batch_size ${agent.batchSize})`,
        `backend    ${JSON.stringify(backendInfo(agent))}`,
        `host       ${env.runtime} on ${env.platform}, ${env.cpu}, ${env.memory_gib} GiB`,
        `load       ${loadS.toFixed(2)} s`,
        `1 short question (${shortResult.usage.input_tokens} tok): P50 ${f(shortT.p50_ms)} ms, P95 ${f(shortT.p95_ms)} ms, min ${f(shortT.min_ms)} ms (${iterations} runs)`,
        `50 questions: mean ${f(manyT.mean_ms)} ms → ${report.fifty.questions_per_second.toFixed(1)} q/s (P50 ${f(manyT.p50_ms)} ms)`,
        ...(peak !== undefined ? [`MLX peak allocation ${(peak / 2 ** 20).toFixed(1)} MiB`] : []),
      ].join("\n") + "\n",
    );
  } finally {
    agent.dispose();
  }
}

/** Runs the CLI; returns the exit code. */
export async function main(argv: string[], io: { out?: (s: string) => void; err?: (s: string) => void } = {}): Promise<number> {
  const out = io.out ?? ((s: string) => void process.stdout.write(s));
  const err = io.err ?? ((s: string) => void process.stderr.write(s));
  try {
    const { values, positionals } = parseArgs({ args: argv, options: OPTIONS, allowPositionals: true, strict: true });
    const a = values as Args;
    if (a.version) {
      out(VERSION + "\n");
      return 0;
    }
    const [command, ...rest] = positionals;
    if (a.help || !command) {
      (a.help ? out : err)(USAGE);
      return a.help ? 0 : 2;
    }
    if (rest.length) throw new UsageError(`unexpected arguments: ${rest.join(" ")}`);
    if (command === "predict") await predict(a, out);
    else if (command === "bench") await bench(a, out);
    else throw new UsageError(`unknown command ${command}`);
    return 0;
  } catch (e) {
    if (e instanceof UsageError || (e as { code?: string }).code?.startsWith("ERR_PARSE_ARGS")) {
      err(`laya: ${(e as Error).message}\n\n${USAGE}`);
      return 2;
    }
    err(`laya: ${(e as Error).message}\n`);
    return 1;
  }
}

export { MLX_MODELS };
