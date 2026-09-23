/** Port of laya-mlx tests/test_shortlist.py (mock embed_fn and predict; no model). */
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
import { makeTest } from "./harness.ts";
const test = makeTest(bun);
import assert from "node:assert/strict";
import { renderOptions, toInternal, type Json } from "@johnhenry/laya-core";
import { predictShortlist, shortlistChoice, type EmbedFn } from "../src/index.ts";

class TableEmbed {
  calls: string[][] = [];
  readonly vectors: Record<string, number[]>;
  constructor(vectors: Record<string, number[]>) {
    this.vectors = vectors;
  }
  fn: EmbedFn = (texts) => {
    this.calls.push([...texts]);
    const missing = texts.filter((t) => !(t in this.vectors));
    if (missing.length) throw new Error(`unexpected texts ${JSON.stringify(missing)}`);
    return texts.map((t) => this.vectors[t]!);
  };
}
const boom: EmbedFn = () => {
  throw new Error("embed_fn should not run when k >= n");
};

const answers = (questions: Record<string, any>) => {
  const out: Record<string, unknown> = {};
  for (const [qid, q] of Object.entries(questions)) {
    if (q && typeof q === "object" && q.type === "choice") {
      const keys = Array.isArray(q.criteria) ? q.criteria : Object.keys(q.criteria);
      out[qid] = { type: "choice", choice: keys[0] };
    }
  }
  return out;
};
class Recorder {
  calls: [unknown, Record<string, any>, unknown[]][] = [];
  systemOneCalls = 0;
  predict(state: unknown, questions: Record<string, any>, ...rest: unknown[]) {
    this.calls.push([state, questions, rest]);
    return { model: "fake", answers: answers(questions) };
  }
  systemOne(state: unknown, questions: Record<string, any>, ...rest: unknown[]) {
    this.systemOneCalls++;
    this.calls.push([state, questions, rest]);
    return { model: "fake", answers: answers(questions) };
  }
}

const CRITERIA = { alpha: null, beta: "", gamma: "mid", delta: "same" };
const OPTION_TEXTS = { alpha: [1, 0], beta: [0, 1], "gamma: mid": [0.6, 0.8], "delta: same": [1, 0] };
const embedFor = (query: string, opts: Record<string, number[]>) => new TableEmbed({ [query]: [1, 0], ...opts });

test("top-k keeps cosine ties in input order", async () => {
  const e = embedFor("pay me", OPTION_TEXTS);
  assert.deepEqual(await shortlistChoice("pay me", CRITERIA, e.fn, 2), ["alpha", "delta"]);
  assert.equal(e.calls.length, 1);
  assert.equal(e.calls[0]![0], "pay me");
  assert.deepEqual(e.calls[0]!.slice(1), renderOptions({ t: "choice", crit: CRITERIA }));
  assert.deepEqual(await shortlistChoice("pay me", CRITERIA, e.fn, 1), ["alpha"]);
  assert.deepEqual(await shortlistChoice("pay me", CRITERIA, e.fn, 3), ["alpha", "delta", "gamma"]);
});

test("a zero query keeps the original order", async () => {
  const z = new TableEmbed({ "pay me": [0, 0], alpha: [1, 0], beta: [0, 1], "gamma: mid": [0.6, 0.8], "delta: same": [3, 4] });
  assert.deepEqual(await shortlistChoice("pay me", CRITERIA, z.fn, 2), ["alpha", "beta"]);
});

test("a NaN vector sorts behind a finite match", async () => {
  const n = new TableEmbed({ "pay me": [1, 0], alpha: [NaN, NaN], beta: [1, 0] });
  assert.deepEqual(await shortlistChoice("pay me", { alpha: null, beta: null }, n.fn, 1), ["beta"]);
});

test("list criteria and instructions change the query", async () => {
  const e = new TableEmbed({ "Classify\npay me": [0, 1], alpha: [1, 0], beta: [0, 1], gamma: [0, 0.2] });
  assert.deepEqual(await shortlistChoice("pay me", ["alpha", "beta", "gamma"], e.fn, 2, { instructions: "Classify" }), ["beta", "gamma"]);
  assert.equal(e.calls[0]![0], "Classify\npay me");
  assert.deepEqual(e.calls[0]!.slice(1), ["alpha", "beta", "gamma"]);
});

test("a dict state is serialized", async () => {
  const e = new TableEmbed({ 'Classify\n{"text": "hi"}': [1, 0], alpha: [1, 0], beta: [0, 1] });
  assert.deepEqual(await shortlistChoice({ text: "hi" }, ["alpha", "beta"], e.fn, 1, { instructions: "Classify" }), ["alpha"]);
  assert.equal(e.calls[0]![0], 'Classify\n{"text": "hi"}');
});

test("zero and false stay in the option text", async () => {
  const rich: Record<string, Json> = { zero: 0, no: false, bare: null, named: { desc: "payments" } };
  const rendered = renderOptions({ t: "choice", crit: rich });
  const e = new TableEmbed({ "pay me": [1, 0], ...Object.fromEntries(rendered.map((t) => [t, [1, 0]])) });
  await shortlistChoice("pay me", rich, e.fn, 1);
  assert.deepEqual(e.calls[0]!.slice(1), rendered);
});

test("k >= n passes every label through without embedding", async () => {
  assert.deepEqual(await shortlistChoice("pay me", CRITERIA, boom, 4), Object.keys(CRITERIA));
  assert.deepEqual(await shortlistChoice("pay me", CRITERIA, boom, 20), Object.keys(CRITERIA));
});

const SENTINEL = { desc: "payments" };
const FULL = { billing: SENTINEL, tech: "bugs", sales: null, other: "misc" };
const FULL_VECTORS = {
  "Which desk?\nI was charged twice": [1, 0],
  'billing: {"desc": "payments"}': [0, 1],
  "tech: bugs": [1, 0],
  sales: [0.2, 0.2],
  "other: misc": [0, 1],
};
const SCORE_Q = { type: "score", instructions: "How urgent?", criteria: ["low", "mid", "high", "now"] };
const NOUL_Q = { type: "noul", instructions: "Is a refund requested?" };

test("predictShortlist reduces choice questions and preserves the rest", async () => {
  const agent = new Recorder();
  const questions = {
    intent: { type: "choice", instructions: "Which desk?", criteria: FULL },
    urgency: SCORE_Q,
    refund: NOUL_Q,
    note: "leave me alone",
  };
  const state = "I was charged twice";
  const result = await predictShortlist(agent, state, questions, { embedFn: new TableEmbed(FULL_VECTORS).fn, k: 2, predictOptions: { model: "english" } });
  assert.equal(agent.calls.length, 1);
  assert.equal(agent.systemOneCalls, 0);
  const [gotState, got, rest] = agent.calls[0]!;
  assert.equal(gotState, state);
  assert.deepEqual(rest, [{ model: "english" }]);
  assert.deepEqual(Object.keys(got.intent.criteria), ["tech", "sales"]);
  assert.equal(got.intent.criteria.tech, "bugs");
  assert.equal(got.urgency, SCORE_Q);
  assert.equal(got.refund, NOUL_Q);
  assert.equal(got.note, questions.note);
  assert.equal(questions.intent.criteria, FULL); // caller's questions untouched
  assert.equal(FULL.billing, SENTINEL);
  assert.deepEqual(Object.keys(toInternal(got.intent).crit as object), ["tech", "sales"]);
  assert.equal((result.answers as any).intent.choice, "tech");
  const meta = result.shortlist.intent!;
  assert.deepEqual(meta.labels, ["tech", "sales"]);
  assert.ok(meta.scores![0]! > meta.scores![1]! && meta.scores![1]! > 0);
  assert.deepEqual([meta.k, meta.n], [2, 4]);
  assert.equal(meta.passthrough, false);
  assert.ok(!("urgency" in result.shortlist));
});

test("the shortlist key is attached to a copy", async () => {
  const held: { result?: Record<string, unknown> } = {};
  const holding = { predict: () => (held.result = { model: "fake", answers: {} }) };
  const out = await predictShortlist(holding, "pay me", { intent: { type: "choice", criteria: CRITERIA } }, { embedFn: boom, k: 4 });
  assert.ok("shortlist" in out);
  assert.ok(!("shortlist" in held.result!));
});

test("pass-through reaches predict unchanged", async () => {
  const agent = new Recorder();
  const q = { type: "choice", instructions: "Which desk?", criteria: FULL };
  const out = await predictShortlist(agent, "I was charged twice", { intent: q }, { embedFn: boom, k: 4 });
  assert.equal(agent.calls[0]![1].intent, q);
  const meta = out.shortlist.intent!;
  assert.deepEqual(meta.labels, Object.keys(FULL));
  assert.equal(meta.scores, null);
  assert.equal(meta.passthrough, true);
  const out2 = await predictShortlist(new Recorder(), "x", { intent: q }, { embedFn: boom, k: 99 });
  assert.equal(out2.shortlist.intent!.passthrough, true);
});

test("list criteria stay a list, in rank order", async () => {
  const agent = new Recorder();
  const e = new TableEmbed({ "Which?\nhello": [1, 0], alpha: [0, 1], beta: [1, 0], gamma: [0, 0] });
  const questions = { intent: { type: "choice", instructions: "Which?", criteria: ["alpha", "beta", "gamma"] } };
  await predictShortlist(agent, "hello", questions, { embedFn: e.fn, k: 2 });
  assert.deepEqual(agent.calls[0]![1].intent.criteria, ["beta", "alpha"]);
  assert.deepEqual(questions.intent.criteria, ["alpha", "beta", "gamma"]);
  assert.deepEqual(toInternal(agent.calls[0]![1].intent).labels, ["beta", "alpha"]);
});

test("systemOne is used when predict is absent", async () => {
  const e = new TableEmbed({ hello: [1, 0], alpha: [0, 1], beta: [1, 0] });
  const only = {
    questions: undefined as any,
    systemOne(_s: unknown, questions: any) {
      this.questions = questions;
      return { answers: { intent: { choice: "beta" } } };
    },
  };
  const out = await predictShortlist(only, "hello", { intent: { type: "choice", criteria: ["alpha", "beta"] } }, { embedFn: e.fn, k: 1 });
  assert.equal((out.answers as any).intent.choice, "beta");
  assert.equal(only.questions.intent.criteria.length, 1);
});

test("bad k is rejected", async () => {
  const e = embedFor("pay me", OPTION_TEXTS);
  for (const k of [0, -3, true, 1.5, "2"]) {
    await assert.rejects(shortlistChoice("pay me", CRITERIA, e.fn, k as number), /k must be a positive integer/);
  }
});

test("invalid criteria are rejected", async () => {
  const e = embedFor("pay me", OPTION_TEXTS);
  await assert.rejects(shortlistChoice("pay me", {}, e.fn, 1), /at least one option/);
  await assert.rejects(shortlistChoice("pay me", [], e.fn, 1), /at least one option/);
  await assert.rejects(shortlistChoice("pay me", "alpha", e.fn, 1), TypeError);
  await assert.rejects(shortlistChoice("pay me", ["alpha", "alpha"], e.fn, 1), /duplicated/);
});

test("missing criteria and bad questions are rejected", async () => {
  const e = embedFor("pay me", OPTION_TEXTS);
  await assert.rejects(predictShortlist(new Recorder(), "pay me", { intent: { type: "choice" } }, { embedFn: e.fn, k: 1 }), /has no criteria/);
  await assert.rejects(predictShortlist(new Recorder(), "pay me", [] as never, { embedFn: e.fn, k: 1 }), TypeError);
});

test("a bad embedding shape is rejected before predict", async () => {
  const agent = new Recorder();
  const q = { type: "choice", instructions: "Which desk?", criteria: FULL };
  await assert.rejects(predictShortlist(agent, "pay me", { intent: q }, { embedFn: () => [[0, 0, 0, 0]], k: 2 }), /shape/);
  assert.equal(agent.calls.length, 0);
});

test("no embedFn and no agent.embed is a TypeError (only when embedding is needed)", async () => {
  const q = { type: "choice", instructions: "Which desk?", criteria: FULL };
  await assert.rejects(predictShortlist(new Recorder(), "pay me", { intent: q }, { k: 2 }), /embed_fn must be callable/);
  await predictShortlist(new Recorder(), "pay me", { intent: q }, { k: 4 }); // pass-through needs none
});
