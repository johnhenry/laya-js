/**
 * End-to-end agent tests on the tiny fixture checkpoint (default suite; the
 * CPU backend always runs, MLX and WebGPU when available). Ports the
 * Agent-level cases of laya-mlx tests/test_runtime.py and test_router.py.
 */
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
import { makeTest } from "./harness.ts";
const test = makeTest(bun);
import assert from "node:assert/strict";
import { cp, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fixturePath, loadJson } from "@johnhenry/laya-fixtures";
import { loadTokenizerFromDir } from "@johnhenry/laya-core/node";
import { readSafetensors, writeSafetensors } from "@johnhenry/math-plus-safetensors";
import { safetensorsWeights } from "@johnhenry/modernbert";
import { createCpuBackend } from "@johnhenry/backend-cpu";
import type { PredictResult, Questions } from "@johnhenry/laya-core";
import { createAgent, createBackend, load, predictShortlist, type LayaAgent } from "../src/index.ts";

const TINY = fixturePath("tiny");
const fx = await loadJson<{ state: any; questions: Questions; items: unknown; result: PredictResult }>("tiny", "predict.json");
const QUESTIONS: Questions = {
  topic: { type: "choice", instructions: "Choose", criteria: ["a", "b", "c"] },
  level: { type: "score", instructions: "Level", criteria: ["low", "high"] },
  yes: { type: "noul", instructions: "Is this true?" },
};
const quiet = { warn: () => {} };

async function available(name: "mlx" | "webgpu"): Promise<string | false> {
  try {
    (await createBackend(name)).destroy?.();
    return false;
  } catch (e) {
    return `${name} unavailable: ${(e as Error).message.split("\n")[0]}`;
  }
}
const skipMlx = await available("mlx");
const skipWebgpu = await available("webgpu");

/** Probabilities / noul / score within `tol`, identical choices and everything else. */
function assertClose(got: PredictResult, want: PredictResult, tol: number) {
  assert.deepEqual(Object.keys(got.answers), Object.keys(want.answers));
  assert.deepEqual(got.usage, want.usage);
  for (const [qid, w] of Object.entries(want.answers)) {
    const g = got.answers[qid]!;
    assert.equal(g.type, w.type, qid);
    if (w.choice !== undefined) assert.equal(g.choice, w.choice, `${qid} choice`);
    for (const key of ["confidence", "noul", "score"] as const) {
      if (w[key] !== undefined) assert.ok(Math.abs(g[key]! - w[key]!) <= tol, `${qid}.${key} ${g[key]} vs ${w[key]}`);
    }
    assert.ok(Math.abs(g.action.act_probability - w.action.act_probability) <= tol, `${qid} act`);
    if (w.probabilities) {
      assert.deepEqual(Object.keys(g.probabilities!), Object.keys(w.probabilities));
      for (const [l, p] of Object.entries(w.probabilities)) assert.ok(Math.abs(g.probabilities![l]! - p) <= tol, `${qid}[${l}]`);
    }
  }
}

test("tiny checkpoint: load() + predict() deep-equals Python's result (cpu)", async () => {
  const agent = await load(TINY, { backend: "cpu", batchSize: 2, ...quiet });
  assert.equal(agent.dtype, "f32"); // cpu always computes in f32
  assert.equal(agent.backend.name, "cpu");
  assert.deepEqual(agent.prepare(fx.state, fx.questions).items, fx.items);
  assert.deepEqual(await agent.predict(fx.state, fx.questions), fx.result);
  assert.deepEqual(await agent.systemOne(fx.state, fx.questions), fx.result);
  agent.dispose();
  agent.dispose(); // idempotent
  await assert.rejects(agent.predict(fx.state, fx.questions), /disposed/);
});

for (const [name, skip] of [["mlx", skipMlx], ["webgpu", skipWebgpu]] as const) {
  for (const dtype of ["f32", "f16"] as const) {
    const tol = dtype === "f32" ? 1e-4 : 0.02;
    test(`tiny checkpoint: predict() on ${name} ${dtype} within ${tol} of Python`, async () => {
      const agent = await load(TINY, { backend: name, dtype, batchSize: 2, ...quiet });
      assert.equal(agent.dtype, dtype);
      try {
        assertClose(await agent.predict(fx.state, fx.questions), fx.result, tol);
      } finally {
        agent.dispose();
      }
    }, { skip });
  }
}

test("chunking does not change results; empty requests are answered", async () => {
  const together = await load(TINY, { backend: "cpu", batchSize: 16, ...quiet });
  const separate = await load(TINY, { backend: "cpu", batchSize: 1, ...quiet });
  const a = await together.predict({ text: "hello" }, QUESTIONS);
  assert.deepEqual(await separate.predict({ text: "hello" }, QUESTIONS), a);
  assert.deepEqual(Object.keys(a.answers), Object.keys(QUESTIONS));
  assert.equal(a.usage.output_tokens, 0);
  assert.ok(a.answers.yes!.noul! >= 0 && a.answers.yes!.noul! <= 1);
  assert.ok(a.answers.level!.score! >= 0 && a.answers.level!.score! <= 1);
  const empty = await together.predict("", {});
  assert.deepEqual(empty.answers, {});
  assert.equal(empty.usage.input_tokens, 0);
  together.dispose();
  separate.dispose();
});

test("single option and a state longer than max_len", async () => {
  const agent = await load(TINY, { backend: "cpu", ...quiet });
  const r = await agent.predict("hello ".repeat(1000), { one: { type: "choice", instructions: "choose", criteria: ["only"] } });
  assert.deepEqual(r.answers.one!.probabilities, { only: 1 });
  assert.equal(r.usage.input_tokens, agent.config.max_len);
  agent.dispose();
});

test("padToMultiple + cachePrompts + compile preserve predictions", async () => {
  const backend = skipMlx ? "cpu" : "mlx";
  const original = await load(TINY, { backend, dtype: "f32", batchSize: 2, ...quiet });
  const optimized = await load(TINY, { backend, dtype: "f32", batchSize: 2, compile: true, padToMultiple: 16, cachePrompts: true, ...quiet });
  for (const state of ["hello", "hello ".repeat(70), "hello hello", ""]) {
    for (let rep = 0; rep < 2; rep++) assertClose(await optimized.predict(state, QUESTIONS), await original.predict(state, QUESTIONS), 2e-4);
  }
  original.dispose();
  optimized.dispose();
});

test("cached prefixes preserve inputs under mutation and truncation", async () => {
  const original = await load(TINY, { backend: "cpu", ...quiet });
  const cached = await load(TINY, { backend: "cpu", cachePrompts: true, ...quiet });
  const qs = structuredClone(QUESTIONS) as Record<string, any>;
  for (const state of ["", "[MASK] hello", "hello ".repeat(1000), { text: "你好", flag: false }]) {
    for (const count of [2, 5, 12]) {
      qs.topic.criteria = Object.fromEntries(Array.from({ length: count }, (_, i) => [String(i), { value: i }]));
      assert.deepEqual(cached.prepare(state as any, qs), original.prepare(state as any, qs));
    }
  }
  assert.deepEqual(cached.prepare("", {}), { items: [], internal: [] });
  original.dispose();
  cached.dispose();
});

test("load() validation mirrors Agent.__init__ / resolve_model", async () => {
  await assert.rejects(load(TINY, { batchSize: 0 }), /batch_size must be a positive integer/);
  await assert.rejects(load(TINY, { padToMultiple: 0 }), /pad_to_multiple must be a positive integer or None/);
  await assert.rejects(load(TINY, { dtype: "f64" as never }), /dtype must be one of/);
  await assert.rejects(load(TINY, { device: "tpu" as never }), /MLX device must be/);
  await assert.rejects(load(join(tmpdir(), "laya-missing-dir-xyz")), /Local model directory does not exist/);
  await assert.rejects(load("./laya-missing-dir-xyz"), /Local model directory does not exist/);
  await assert.rejects(load("test/model", { subfolder: "../outside" }), /subfolder must be a relative path/);
  await assert.rejects(load(TINY, { subfolder: "nope", backend: "cpu" }), /Not a complete Laya checkpoint: .*nope\/model\.safetensors is missing/);
});

async function tinyCopy(edit?: (cfg: Record<string, any>) => void): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "laya-tiny-"));
  await cp(TINY, dir, { recursive: true });
  if (edit) {
    const cfg = JSON.parse(await readFile(join(dir, "rl_agent_config.json"), "utf8"));
    edit(cfg);
    await writeFile(join(dir, "rl_agent_config.json"), JSON.stringify(cfg));
  }
  return dir;
}

test("config bounds, temperatures and clamping warnings", async () => {
  const bad = [
    [(c: any) => delete c.head_layers, /must specify encoder and head_layers/],
    [(c: any) => (c.head_max_len = 4), /4 < head_max_len < max_len <= max_position_embeddings/],
    [(c: any) => (c.max_len = 4096), /4 < head_max_len < max_len <= max_position_embeddings/],
    [(c: any) => (c.temperature = [1, 0, 1]), /Calibration temperatures must be finite and positive/],
    [(c: any) => (c.temperature_by_options = { "choice:2": -1 }), /Calibration temperatures must be finite and positive/],
  ] as const;
  for (const [edit, msg] of bad) {
    const dir = await tinyCopy(edit);
    await assert.rejects(load(dir, { backend: "cpu", ...quiet }), msg);
    await rm(dir, { recursive: true });
  }
  const dir = await tinyCopy((c) => (c.temperature_by_options["choice:11+"] = 0.10058280825614929));
  const warnings: string[] = [];
  const agent = await load(dir, { backend: "cpu", warn: (m) => warnings.push(m) });
  assert.deepEqual(warnings, [
    "laya: this checkpoint ships temperatures outside [0.5, 5] which would distort confidence; clamping choice:11+=0.1006. " +
      "Treat confidence from the affected buckets as uncalibrated.",
  ]);
  assert.equal(agent.temperatureByOptions["choice:11+"], 0.5);
  assert.equal(agent.temperatureByOptionsRaw["choice:11+"], 0.10058280825614929);
  assert.equal(agent.temperatureByOptions["choice:2"], 1.7); // legitimate value untouched
  assert.deepEqual(agent.temperature, [1.3, 1.1, 2.0]);
  agent.dispose();
  await rm(dir, { recursive: true });
});

test("unexpected checkpoint parameters fail loudly (strict loading)", async () => {
  const dir = await tinyCopy();
  const file = readSafetensors(await readFile(join(dir, "model.safetensors")));
  const tensors: Record<string, { dtype: string; shape: number[]; data: ArrayBufferView }> = {};
  for (const n of file.names()) tensors[n] = { dtype: file.info(n).dtype, shape: [...file.info(n).shape], data: file.view(n) as ArrayBufferView };
  tensors["head.extra.weight"] = { dtype: "F32", shape: [2], data: new Float32Array([1, 2]) };
  await writeFile(join(dir, "model.safetensors"), writeSafetensors(tensors as never));
  await assert.rejects(load(dir, { backend: "cpu", ...quiet }), /Received parameters not in model: head\.extra\.weight/);
  await rm(dir, { recursive: true });
});

test("createAgent() builds an agent from parts (no I/O) and forward() runs one batch", async () => {
  const tokenizer = await loadTokenizerFromDir(join(TINY, "tokenizer"));
  const file = readSafetensors(await readFile(join(TINY, "model.safetensors")));
  const backend = createCpuBackend();
  const agent: LayaAgent = createAgent({
    backend,
    encoderConfig: await loadJson("tiny", "encoder", "config.json"),
    agentConfig: await loadJson("tiny", "rl_agent_config.json"),
    weights: safetensorsWeights(file),
    tokenizer,
    batchSize: 2,
    ...quiet,
  });
  assert.equal(agent.modelId, "<local>");
  assert.deepEqual(await agent.predict(fx.state, fx.questions), fx.result);
  const { items } = agent.prepare(fx.state, fx.questions);
  const { collate } = await import("@johnhenry/laya-core");
  const out = await agent.forward(collate(items, tokenizer.padTokenId));
  assert.equal(out.nAct, 2);
  assert.equal(out.act.length, items.length * 2);
  agent.dispose();
});

test("embed(): mean pool over real tokens only; empty text is the zero vector", async () => {
  const agent = await load(TINY, { backend: "cpu", ...quiet });
  const H = agent.encoderConfig.hiddenSize;
  const [a, b, c] = await agent.embed(["hello", "hello hello w1 w2", ""]);
  assert.equal(a!.length, H);
  assert.ok([...a!, ...b!].every(Number.isFinite));
  assert.ok(c!.every((x) => x === 0));
  const [alone] = await agent.embed(["hello"]);
  for (let i = 0; i < H; i++) assert.ok(Math.abs(alone![i]! - a![i]!) <= 1e-5, "padding leaked into the mean");
  const [again] = await agent.embed(["hello"], { batchSize: 1, maxLength: 8 });
  assert.deepEqual(again, alone);
  await assert.rejects(agent.embed(["x"], { maxLength: 0 }), /max_length must be a positive integer/);
  assert.deepEqual(await agent.embed([]), []);
  agent.dispose();
});

test("predictShortlist end to end with the agent's own encoder", async () => {
  const agent = await load(TINY, { backend: "cpu", ...quiet });
  const out = await predictShortlist(agent, "hello", {
    intent: { type: "choice", instructions: "Which?", criteria: ["alpha", "beta", "gamma", "delta"] },
  }, { k: 2 });
  assert.equal(out.shortlist.intent!.n, 4);
  assert.equal(out.shortlist.intent!.labels.length, 2);
  assert.ok(out.shortlist.intent!.labels.includes(out.answers.intent!.choice!));
  assert.deepEqual(Object.keys(out.answers.intent!.probabilities!), out.shortlist.intent!.labels);
  agent.dispose();
});
