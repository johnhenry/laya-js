/**
 * Opt-in end-to-end parity on the three published checkpoints:
 *   LAYA_REAL=1 [LAYA_REAL_MODELS=english,...] [LAYA_REAL_BACKENDS=mlx-f32,webgpu-f16,...] npm test -w @johnhenry/laya
 * `load(repo, { offline: true })` from the local HF cache, then `predict` on
 * every fixture case (16 cases, 63 questions per checkpoint):
 *   f32 vs Python result_fp32: every probability / confidence / score / noul /
 *       act_probability within 1e-4, identical choices;
 *   f16 vs Python result_fp16: within 0.02, identical choices.
 * `embed` is compared with the Python `embed_fn_from_agent` vectors (f32: 1e-3 rel).
 * CPU (≈60 s per case): LAYA_REAL_CPU=1 adds one small English case on cpu f32.
 * Take ~/gpu.lock around this run (AGENTS.md rule 6).
 */
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
import { env, makeTest } from "./harness.ts";
const test = makeTest(bun);
import assert from "node:assert/strict";
import { MODELS, REPOS, loadReal, type ModelName, type RealFixture } from "@johnhenry/laya-fixtures";
import { decodeTensor } from "@johnhenry/tensor-backend/conformance";
import type { PredictResult } from "@johnhenry/laya-core";
import { createBackend, load } from "../src/index.ts";

const enabled = env.LAYA_REAL === "1";
const models = (env.LAYA_REAL_MODELS ?? MODELS.join(",")).split(",") as ModelName[];
const CONFIGS = ["mlx-f32", "mlx-f16", "webgpu-f32", "webgpu-f16"] as const;
const configs = (env.LAYA_REAL_BACKENDS ?? CONFIGS.join(",")).split(",");
const HOURS = 3600_000;

interface Cmp {
  questions: number;
  choiceAgree: number;
  choiceTotal: number;
  exact: number;
  /** argmax of the answer: choice label, score argmax level, noul side (p >= 0.5) */
  argmaxAgree: number;
  maxAbs: number;
  worst: string;
}

function argmaxOf(a: any): string {
  if (a.type === "noul") return a.noul >= 0.5 ? "true" : "false";
  const p = Object.entries(a.probabilities as Record<string, number>);
  return p.reduce((best, cur) => (cur[1] > best[1] ? cur : best))[0];
}

/** Compares every numeric answer field; counts choice agreement and exact (deep-equal) answers. */
function compare(got: PredictResult, want: PredictResult, into: Cmp, where: string): void {
  assert.deepEqual(Object.keys(got.answers), Object.keys(want.answers), where);
  assert.deepEqual(got.usage, want.usage, `${where} usage`);
  for (const [qid, w] of Object.entries(want.answers) as [string, any][]) {
    const g = got.answers[qid] as any;
    into.questions++;
    if (JSON.stringify(g) === JSON.stringify(w)) into.exact++;
    if (argmaxOf(g) === argmaxOf(w)) into.argmaxAgree++;
    if (w.choice !== undefined) {
      into.choiceTotal++;
      if (g.choice === w.choice) into.choiceAgree++;
    }
    const pairs: [string, number, number][] = [
      ["confidence", g.confidence, w.confidence],
      ["act_probability", g.action.act_probability, w.action.act_probability],
    ];
    for (const key of ["noul", "score"]) if (w[key] !== undefined) pairs.push([key, g[key], w[key]]);
    for (const [l, p] of Object.entries(w.probabilities ?? {}) as [string, number][]) pairs.push([`p[${l}]`, g.probabilities?.[l], p]);
    for (const [name, a, b] of pairs) {
      const d = Math.abs(a - b);
      if (!(d <= into.maxAbs)) {
        into.maxAbs = Number.isFinite(d) ? d : Infinity;
        into.worst = `${where} ${qid}.${name}: ${a} vs ${b}`;
      }
    }
  }
}

const matrix: string[] = [];
const fixtures = new Map<ModelName, RealFixture>();
if (enabled) for (const m of models) fixtures.set(m, await loadReal(m));

async function unavailable(name: "mlx" | "webgpu"): Promise<string | false> {
  try {
    (await createBackend(name)).destroy?.();
    return false;
  } catch (e) {
    return `${name} unavailable: ${(e as Error).message.split("\n")[0]}`;
  }
}
const skipOf: Record<string, string | false> = enabled ? { mlx: await unavailable("mlx"), webgpu: await unavailable("webgpu") } : {};

for (const m of models) {
  for (const cfg of configs) {
    const [backend, dtype] = cfg.split("-") as ["mlx" | "webgpu", "f32" | "f16"];
    const tol = dtype === "f32" ? 1e-4 : 0.02;
    const name = `real ${m} on ${backend} ${dtype}: 63 questions vs Python result_${dtype === "f32" ? "fp32" : "fp16"} (≤ ${tol}, identical choices)`;
    test(name, async () => {
      const fx = fixtures.get(m)!;
      const t0 = performance.now();
      const agent = await load(REPOS[m], { offline: true, backend, dtype, warn: () => {} });
      const loadMs = performance.now() - t0;
      const cmp: Cmp = { questions: 0, choiceAgree: 0, choiceTotal: 0, exact: 0, argmaxAgree: 0, maxAbs: 0, worst: "" };
      const t1 = performance.now();
      try {
        for (const c of fx.cases) {
          const got = await agent.predict(c.state as any, c.questions as any);
          compare(got, dtype === "f32" ? c.result_fp32 : c.result_fp16, cmp, c.case);
        }
        let embRel = NaN;
        if (dtype === "f32") {
          const want = decodeTensor(fx.embeddings.vectors);
          const H = want.shape[1]!;
          const got = await agent.embed(fx.embeddings.texts);
          embRel = 0;
          got.forEach((v, r) => {
            let num = 0, den = 0;
            for (let i = 0; i < H; i++) {
              num = Math.max(num, Math.abs(v[i]! - (want.data as Float32Array)[r * H + i]!));
              den = Math.max(den, Math.abs((want.data as Float32Array)[r * H + i]!));
            }
            embRel = Math.max(embRel, num / den);
          });
        }
        const runMs = performance.now() - t1;
        const line =
          `| ${m} | ${backend} ${dtype} | ${cmp.choiceAgree}/${cmp.choiceTotal} | ${cmp.argmaxAgree}/${cmp.questions} | ${cmp.exact}/${cmp.questions} | ` +
          `${cmp.maxAbs.toExponential(2)} | ${Number.isNaN(embRel) ? "—" : embRel.toExponential(2)} | ${(loadMs / 1000).toFixed(1)} s | ${(runMs / 1000).toFixed(1)} s |`;
        matrix.push(line);
        console.log(line + (cmp.worst ? `\n    worst: ${cmp.worst}` : ""));
        assert.equal(cmp.questions, 63);
        assert.equal(cmp.choiceAgree, cmp.choiceTotal, "choice agreement");
        if (dtype === "f32") assert.equal(cmp.argmaxAgree, cmp.questions, "argmax agreement");
        assert.ok(cmp.maxAbs <= tol, `max |Δ| ${cmp.maxAbs} > ${tol} (${cmp.worst})`);
        if (dtype === "f32") assert.ok(embRel <= 1e-3, `embedding max rel error ${embRel}`);
      } finally {
        agent.dispose();
      }
    }, { skip: !enabled ? "opt-in: LAYA_REAL=1" : skipOf[backend] || false, timeout: 2 * HOURS });
  }
}

test("real english on cpu f32: one small case vs Python result_fp32 (LAYA_REAL_CPU=1)", async () => {
  const fx = fixtures.get("english") ?? (await loadReal("english"));
  const c = [...fx.cases].sort((a, b) => a.items.reduce((n, i) => n + i.ids.length, 0) - b.items.reduce((n, i) => n + i.ids.length, 0))[0]!;
  const agent = await load(REPOS.english, { offline: true, backend: "cpu", warn: () => {} });
  const cmp: Cmp = { questions: 0, choiceAgree: 0, choiceTotal: 0, exact: 0, argmaxAgree: 0, maxAbs: 0, worst: "" };
  const t0 = performance.now();
  try {
    compare(await agent.predict(c.state as any, c.questions as any), c.result_fp32, cmp, c.case);
  } finally {
    agent.dispose();
  }
  const line = `| english (case ${c.case}) | cpu f32 | ${cmp.choiceAgree}/${cmp.choiceTotal} | ${cmp.argmaxAgree}/${cmp.questions} | ${cmp.exact}/${cmp.questions} | ${cmp.maxAbs.toExponential(2)} | — | — | ${((performance.now() - t0) / 1000).toFixed(1)} s |`;
  matrix.push(line);
  console.log(line);
  assert.equal(cmp.choiceAgree, cmp.choiceTotal);
  assert.equal(cmp.argmaxAgree, cmp.questions);
  assert.ok(cmp.maxAbs <= 1e-4, cmp.worst);
}, { skip: env.LAYA_REAL_CPU === "1" ? false : "opt-in: LAYA_REAL_CPU=1", timeout: 2 * HOURS });

if (enabled) {
  test("parity matrix", () => {
    console.log(
      "\n| checkpoint | backend | choices | argmax | exact answers | max |Δ| | embed rel | load | 16 cases |\n|---|---|---|---|---|---|---|---|---|\n" + matrix.join("\n"),
    );
  });
}
