import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const test = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import assert from "node:assert/strict";
import {
  formatResults,
  prepare,
  PrefixCache,
  resolveTemperatures,
  toInternal,
  type AgentConfig,
  type BatchOutputs,
  type PreparedItem,
} from "@johnhenry/laya-core";
import { DIRS, MODELS, real, skipReason, tokenizer } from "./helpers.ts";

const BATCH = 16; // dump_js_fixtures.py batch size

/** Rebuild per-chunk BatchOutputs from the fixture's k-trimmed logits (padded slots = -1e4). */
function chunkOutputs(items: PreparedItem[], rows: Array<{ logits: number[]; act: number[] }>): BatchOutputs[] {
  const out: BatchOutputs[] = [];
  for (let start = 0; start < items.length; start += BATCH) {
    const chunk = rows.slice(start, start + BATCH);
    const m = Math.max(2, ...items.slice(start, start + BATCH).map((it) => it.markers.length));
    const nAct = chunk[0]!.act.length;
    const logits = new Float32Array(chunk.length * m).fill(-1e4);
    const act = new Float32Array(chunk.length * nAct);
    chunk.forEach((r, i) => {
      logits.set(r.logits, i * m);
      act.set(r.act, i * nAct);
    });
    out.push({ logits, act, nAct });
  }
  return out;
}

for (const m of MODELS) {
  test(`${m}: prepare() reproduces every case's items exactly (uncached and cached)`, async () => {
    const dir = DIRS[m]!;
    const fx = await real(m);
    const tok = await tokenizer(dir);
    const cfg = fx.config as AgentConfig;
    const cache = new PrefixCache();
    assert.equal(fx.cases.length, 16);
    for (const c of fx.cases) {
      const { items, internal } = prepare(tok, c.state, c.questions, cfg);
      assert.deepEqual(items, c.items, `${m}/${c.case}`);
      assert.deepEqual(internal.map((q) => q.id), Object.keys(c.questions));
      // twice through the cache: miss then hit
      for (let i = 0; i < 2; i++) assert.deepEqual(prepare(tok, c.state, c.questions, cfg, { cache }).items, c.items);
    }
  }, { skip: skipReason(m) });

  test(`${m}: formatResults(fixture outputs) deep-equals result_fp32 for every case`, async () => {
    const fx = await real(m);
    const temps = resolveTemperatures(fx.config as AgentConfig);
    for (const c of fx.cases) {
      // items come from the fixture, so this test needs no tokenizer
      const internal = Object.entries(c.questions).map(([id, q]) => ({ ...toInternal(q), id }));
      const result = formatResults(internal, c.items, chunkOutputs(c.items, c.outputs), temps);
      assert.deepStrictEqual(result, c.result_fp32, `${m}/${c.case}`);
    }
  });
}
