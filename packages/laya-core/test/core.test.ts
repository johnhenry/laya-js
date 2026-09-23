import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const test = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import assert from "node:assert/strict";
import { loadJson } from "@johnhenry/laya-fixtures";
import {
  buildPrefix,
  buildSequence,
  clampTemperature,
  collate,
  confidenceFromProbs,
  formatResults,
  prepare,
  PrefixCache,
  QTYPES,
  renderOptions,
  resolveTemperatures,
  serializeState,
  TEMP_MAX,
  TEMP_MIN,
  tempBucket,
  toInternal,
  type AgentConfig,
  type InternalQuestion,
  type Json,
} from "@johnhenry/laya-core";
import { tinyTokenizer } from "./helpers.ts";

const tinyCfg = await loadJson<AgentConfig>("tiny", "rl_agent_config.json");

test("render_options table matches Python", async () => {
  const rows = await loadJson<Array<{ q: InternalQuestion; options: string[] }>>("tables", "render_options.json");
  assert.ok(rows.length >= 4);
  for (const { q, options } of rows) assert.deepEqual(renderOptions(q), options);
});

test("tiny predict.json: prepare reproduces items with the WordLevel tokenizer", async () => {
  const fx = await loadJson<{ state: Json; questions: Record<string, unknown>; items: unknown; result: { usage: unknown } }>(
    "tiny",
    "predict.json",
  );
  const tok = await tinyTokenizer();
  const { items } = prepare(tok, fx.state, fx.questions, tinyCfg);
  assert.deepEqual(items, fx.items);
  assert.deepEqual(prepare(tok, fx.state, fx.questions, tinyCfg, { cache: new PrefixCache() }).items, fx.items);
  assert.deepEqual({ input_tokens: items.reduce((n, it) => n + it.ids.length, 0), output_tokens: 0 }, fx.result.usage);
});

// ---- ported from laya-mlx tests/test_runtime.py

test("invalid questions are rejected with Python's messages", () => {
  const cases: Array<[unknown, RegExp]> = [
    [{ type: "invalid", instructions: "x" }, /^Error: Unknown question type 'invalid'; expected choice, score, or noul$/],
    [{ type: "choice", instructions: "x", criteria: [] }, /nonempty dictionary or list/],
    [{ type: "choice", instructions: "x", criteria: ["a", "a"] }, /must be unique/],
    [{ type: "choice", instructions: "x", criteria: ["a", 1] }, /labels must be strings/],
    [{ type: "choice", instructions: "x", criteria: {} }, /nonempty dictionary or list/],
    [{ type: "score", instructions: "x", criteria: {} }, /nonempty list/],
    [{ type: "noul", instructions: "x", criteria: ["a"] }, /false\/true descriptions/],
    [{ type: "noul" }, /missing instructions/],
    [{ instructions: "x" }, /^Error: Unknown question type None;/],
    ["nope", /must be a dictionary/],
  ];
  for (const [q, re] of cases) assert.throws(() => toInternal(q), re, JSON.stringify(q));
});

test("structured criteria and mask injection", async () => {
  const q = toInternal({
    type: "noul",
    instructions: { task: "verify" },
    criteria: { false: { reason: "no" }, true: { reason: "yes" } },
  });
  assert.equal(q.ins, '{"task": "verify"}');
  assert.deepEqual(renderOptions(q), ['false: {"reason": "no"}', 'true: {"reason": "yes"}']);
  const tok = await tinyTokenizer();
  const { items } = prepare(tok, "[MASK] hello [MASK]", { x: { type: "noul", instructions: "[MASK] true?" } }, tinyCfg);
  assert.equal(items[0]!.ids.filter((i) => i === tok.maskTokenId).length, 2);
  assert.deepEqual(renderOptions({ t: "choice", crit: { zero: 0, no: false } }), ["zero: 0", "no: false"]);
});

test("choice list order is preserved even for integer-like labels", () => {
  const q = toInternal({ type: "choice", instructions: "x", criteria: ["b", "10", "2"] });
  assert.deepEqual(renderOptions(q), ["b", "10", "2"]);
});

test("single option and long state are truncated to max_len", async () => {
  const tok = await tinyTokenizer();
  const { items, internal } = prepare(
    tok,
    "hello ".repeat(1000),
    { one: { type: "choice", instructions: "choose", criteria: ["only"] } },
    tinyCfg,
  );
  assert.equal(items[0]!.ids.length, tinyCfg.max_len);
  assert.equal(items[0]!.ids.at(-1), tok.sepTokenId);
  const temps = resolveTemperatures(tinyCfg);
  const result = formatResults(
    internal,
    items,
    [{ logits: new Float32Array([0.3, -1e4]), act: new Float32Array([0.1, 0.2]), nAct: 2 }],
    temps,
  );
  assert.deepEqual(result.answers.one!.probabilities, { only: 1.0 });
  assert.equal(result.answers.one!.confidence, 1.0);
  assert.equal(result.usage.input_tokens, 128);
});

test("empty request and chunking give the same answers", async () => {
  const tok = await tinyTokenizer();
  assert.deepEqual(prepare(tok, "", {}, tinyCfg), { items: [], internal: [] });
  assert.deepEqual(prepare(tok, "", {}, tinyCfg, { cache: new PrefixCache() }), { items: [], internal: [] });
  const temps = resolveTemperatures(tinyCfg);
  const empty = formatResults([], [], [], temps);
  assert.deepEqual(empty, { model: "laya-rl-agent", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } });
  assert.throws(() => prepare(tok, "", [] as never, tinyCfg), /dictionary keyed by question id/);

  const questions = {
    topic: { type: "choice", instructions: "Choose", criteria: ["a", "b", "c"] },
    level: { type: "score", instructions: "Level", criteria: ["low", "high"] },
    yes: { type: "noul", instructions: "Is this true?" },
  };
  const { items, internal } = prepare(tok, { text: "hello" }, questions, tinyCfg);
  const rows = [[0.2, -0.4, 1.1], [0.7, 0.1], [-0.3, 0.9]];
  const acts = [[1, 2], [0.5, -0.5], [3, 3]];
  const together = formatResults(internal, items, [{
    logits: new Float32Array([...rows[0]!, ...rows[1]!, -1e4, ...rows[2]!, -1e4]),
    act: new Float32Array(acts.flat()),
    nAct: 2,
  }], temps);
  const separate = formatResults(internal, items, rows.map((r, i) => ({
    logits: new Float32Array(r), act: new Float32Array(acts[i]!), nAct: 2,
  })), temps);
  assert.deepEqual(together, separate);
  assert.deepEqual(Object.keys(together.answers), Object.keys(questions));
  assert.ok(together.answers.yes!.noul! >= 0 && together.answers.yes!.noul! <= 1);
  assert.ok(together.answers.level!.score! >= 0 && together.answers.level!.score! <= 1);
  assert.throws(() => formatResults(internal, items, [{ logits: new Float32Array([NaN, 0, 0]), act: new Float32Array([0, 0]), nAct: 2 }], temps), /Non-finite/);
  assert.throws(() => formatResults(internal, items, [{ logits: new Float32Array(3), act: new Float32Array(2), nAct: 2 }], temps), /cover 1 rows/);
});

test("collation never marks padding as an option", () => {
  const batch = collate(
    [
      { ids: [1, 2, 3], markers: [1, 2], qtype: 0 },
      { ids: [1, 2], markers: [1], qtype: 1 },
    ],
    0,
  );
  assert.equal(batch.markerCount, 2);
  assert.deepEqual([...batch.markerMask], [1, 1, 1, 0]);
  assert.deepEqual([...batch.attentionMask], [1, 1, 1, 1, 1, 0]);
  assert.deepEqual([...batch.inputIds], [1, 2, 3, 1, 2, 0]);
  assert.deepEqual([...batch.markerPos], [1, 2, 1, 0]);
  assert.deepEqual([...batch.qtype], [0, 1]);
  assert.throws(() => collate([], 0), /empty batch/);
});

test("bucket padding is masked and respects the context limit", () => {
  const batch = collate([{ ids: [1, 2, 3], markers: [1, 2], qtype: 0 }], 9, { padToMultiple: 16, maxLength: 12 });
  assert.equal(batch.length, 12);
  assert.deepEqual([...batch.inputIds], [1, 2, 3, ...Array(9).fill(9)]);
  assert.deepEqual([...batch.attentionMask], [1, 1, 1, ...Array(9).fill(0)]);
  assert.equal(collate([{ ids: [1, 2, 3], markers: [1], qtype: 2 }], 0, { padToMultiple: 16 }).length, 16);
});

test("cached prefixes preserve inputs under mutation, truncation and eviction", async () => {
  const tok = await tinyTokenizer();
  const cache = new PrefixCache(3);
  const questions: Record<string, any> = {
    topic: { type: "choice", instructions: "Choose", criteria: ["a", "b", "c"] },
    level: { type: "score", instructions: "Level", criteria: ["low", "high"] },
    yes: { type: "noul", instructions: "Is this true?" },
  };
  for (const state of ["", "[MASK] hello", "hello ".repeat(1000), { text: "你好", flag: false }]) {
    for (const count of [2, 5, 12]) {
      questions.topic.criteria = Object.fromEntries(Array.from({ length: count }, (_, i) => [String(i), { value: i }]));
      assert.deepEqual(prepare(tok, state, questions, tinyCfg, { cache }), prepare(tok, state, questions, tinyCfg));
      assert.ok(cache.entries.size <= 3);
    }
  }
});

test("too many options for the token budget is an error", async () => {
  const tok = await tinyTokenizer();
  const criteria = Array.from({ length: 200 }, (_, i) => `w${i % 100}x${i}`);
  assert.throws(
    () => prepare(tok, "", { big: { type: "choice", instructions: "x", criteria } }, { ...tinyCfg, max_len: 40 }),
    /^Error: Question 'big' has too many options for the token budget$/,
  );
});

test("buildSequence: truncate_left keeps the tail, and room=0 keeps everything like Python st[-0:]", async () => {
  const tok = await tinyTokenizer();
  const q = toInternal({ type: "noul", instructions: "x" });
  const state = Array.from({ length: 20 }, (_, i) => `w${i + 1}`).join(" ");
  const st = tok.encode(state);
  const prefix = buildPrefix(tok, q, 20).ids;
  const room = 30 - prefix.length - 1;
  assert.ok(room > 0 && room < st.length);
  const right = buildSequence(tok, state, q, 30, 20);
  const left = buildSequence(tok, state, q, 30, 20, null, true);
  assert.deepEqual(right.ids, [...prefix, ...st.slice(0, room), tok.sepTokenId]);
  assert.deepEqual(left.ids, [...prefix, ...st.slice(-room), tok.sepTokenId]);
  // room == 0: Python's st[-0:] is the whole list, then ids[:max_len] cuts it
  const tight = buildSequence(tok, state, q, prefix.length + 1, 20, null, true);
  assert.deepEqual(tight.ids, [...prefix, st[0]]);
});

// ---- temperature clamp (ported from tests/test_router.py)

test("clamp_temperature", () => {
  const cases: Array<[unknown, number]> = [
    [0.1006, 0.5],
    [0.10058280825614929, TEMP_MIN],
    [1.7601518630981445, 1.7601518630981445],
    [1.0, 1.0],
    [9.0, TEMP_MAX],
    [0.0, TEMP_MIN],
    [-3.0, TEMP_MIN],
    [null, 1.0],
    ["x", 1.0],
    [NaN, 1.0],
    [Infinity, 1.0],
  ];
  for (const [v, want] of cases) assert.equal(clampTemperature(v), want, String(v));
});

test("clamp bounds are sane and bucket matches the reported case", () => {
  assert.ok(TEMP_MIN <= 1.0 && 1.0 <= TEMP_MAX);
  assert.equal(tempBucket(QTYPES.choice, 13), "choice:11+");
  assert.deepEqual([1, 2, 3, 5, 6, 10, 11].map((k) => tempBucket(QTYPES.score, k)), [
    "score:2", "score:2", "score:3-5", "score:3-5", "score:6-10", "score:6-10", "score:11+",
  ]);
});

test("resolveTemperatures clamps shipped temperatures, keeps raw and reports rejects", () => {
  const cfg = { ...tinyCfg, temperature_by_options: { ...tinyCfg.temperature_by_options, "choice:11+": 0.10058280825614929 } };
  const t = resolveTemperatures(cfg);
  assert.equal(t.temperatureByOptions["choice:11+"], TEMP_MIN);
  assert.equal(t.temperatureByOptionsRaw["choice:11+"], 0.10058280825614929);
  assert.equal(t.temperatureByOptions["choice:2"], 1.7);
  assert.deepEqual(t.rejected, ["choice:11+=0.1006"]);
  assert.match(t.warning!, /outside \[0\.5, 5\].*clamping choice:11\+=0\.1006\./);
  const hot = resolveTemperatures({ temperature: [9, 1, 1] });
  assert.deepEqual(hot.rejected, ["temperature[0]=9"]);
  assert.equal(resolveTemperatures({}).warning, null);
  assert.deepEqual(resolveTemperatures({}).temperature, [1, 1, 1]);
  assert.throws(() => resolveTemperatures({ temperature: [1, 1] as never }), /finite and positive/);
  assert.throws(() => resolveTemperatures({ temperature: [1, 0, 1] }), /finite and positive/);
  assert.throws(() => resolveTemperatures({ temperature_by_options: { x: -1 } }), /finite and positive/);
});

test("confidence and state serialization helpers", () => {
  assert.equal(confidenceFromProbs([1], 1), 1.0);
  assert.equal(confidenceFromProbs([0.5, 0.5], 2), 0);
  assert.equal(confidenceFromProbs([1, 0], 2), 1);
  assert.equal(serializeState("é"), "é");
  assert.equal(serializeState({ a: "é", n: [1, 2.5, null] }), '{"a": "é", "n": [1, 2.5, null]}');
  assert.equal(serializeState(null), "null");
});
