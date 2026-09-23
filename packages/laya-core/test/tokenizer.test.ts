import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const test = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import assert from "node:assert/strict";
import { loadJson, fixturePath } from "@johnhenry/laya-fixtures";
import { loadTokenizer } from "@johnhenry/laya-core";
import { readFile } from "node:fs/promises";
import { DIRS, MODELS, skipReason, tokenizer, tinyTokenizer } from "./helpers.ts";

interface TokTable {
  special: Record<"cls_token" | "sep_token" | "pad_token" | "mask_token", [string, number]>;
  cases: Array<{ text: string; ids: number[]; ids_special: number[] }>;
}

for (const m of MODELS) {
  test(`tokenizer-${m}: ids and special tokens match the Rust tokenizer`, async () => {
    const tok = await tokenizer(DIRS[m]!);
    const table = await loadJson<TokTable>("tables", `tokenizer-${m}.json`);
    assert.deepEqual([tok.clsToken, tok.clsTokenId], table.special.cls_token);
    assert.deepEqual([tok.sepToken, tok.sepTokenId], table.special.sep_token);
    assert.deepEqual([tok.padToken, tok.padTokenId], table.special.pad_token);
    assert.deepEqual([tok.maskToken, tok.maskTokenId], table.special.mask_token);
    assert.ok(table.cases.length >= 10);
    for (const c of table.cases) {
      assert.deepEqual(tok.encode(c.text), c.ids, JSON.stringify(c.text));
      assert.deepEqual(tok.encodeWithSpecialTokens(c.text), c.ids_special, JSON.stringify(c.text));
    }
  }, { skip: skipReason(m) });
}

test("Metaspace split patch is applied only where the Rust config asks for it", async () => {
  assert.deepEqual((await tokenizer(DIRS.multilingual!)).patches, ["Metaspace.split"]);
  if (DIRS.english) assert.deepEqual((await tokenizer(DIRS.english)).patches, []);
}, { skip: skipReason("multilingual") });

test("tiny WordLevel tokenizer: Whitespace pre-tokenizer is Unicode-aware like Rust", async () => {
  const tok = await tinyTokenizer();
  assert.deepEqual(tok.patches, ["Whitespace.unicode", "WordLevel.unk"]);
  assert.equal(tok.maskToken, "[MASK]");
  // "héllo" is one Rust \w+ word (-> [UNK]); ASCII \w would split it into three pieces
  assert.deepEqual(tok.encode("héllo"), [tok.tokenToId("[UNK]")]);
  assert.deepEqual(tok.encode("hello, w1"), [tok.tokenToId("hello"), tok.tokenToId(","), tok.tokenToId("w1")]);
});

test("loadTokenizer accepts raw JSON text and rejects a config without special tokens", async () => {
  const json = await readFile(fixturePath("tiny", "tokenizer", "tokenizer.json"), "utf8");
  const cfg = await readFile(fixturePath("tiny", "tokenizer", "tokenizer_config.json"), "utf8");
  const tok = loadTokenizer(json, cfg);
  assert.equal(tok.padTokenId, 0);
  assert.throws(() => loadTokenizer(json, { cls_token: "[CLS]", sep_token: "[SEP]", pad_token: "[PAD]" }), /missing a valid mask_token/);
  assert.throws(() => loadTokenizer(json, { ...JSON.parse(cfg), mask_token: "<nope>" }), /missing a valid mask_token/);
  const dictCfg = { ...JSON.parse(cfg), mask_token: { content: "[MASK]" } };
  assert.equal(loadTokenizer(json, dictCfg).maskTokenId, 4);
});
