/** `laya predict` on the tiny fixture checkpoint (CPU), output format vs Python json.dumps, usage errors. */
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
import { makeTest } from "./harness.ts";
const test = makeTest(bun);
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { fixturePath, loadJson } from "@johnhenry/laya-fixtures";
import { dumpsIndent, main, pythonFloats, timing, workload } from "../src/index.ts";

const TINY = fixturePath("tiny");
const fx = await loadJson<{ state: unknown; questions: unknown; result: any }>("tiny", "predict.json");
const run = async (argv: string[]) => {
  let out = "", err = "";
  const code = await main(argv, { out: (s) => void (out += s), err: (s) => void (err += s) });
  return { code, out, err };
};

let pyDump: ((v: unknown) => string) | null = null;
try {
  execFileSync("python3", ["-c", "print(1)"]);
  pyDump = (v) => execFileSync("python3", ["-c", "import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))"], { input: JSON.stringify(v), encoding: "utf8" });
} catch {
  /* no python: format comparison skipped */
}

test("laya predict: tiny checkpoint on cpu equals Python's result", async () => {
  const r = await run(["predict", "--model", TINY, "--backend", "cpu", "--state", JSON.stringify(fx.state), "--questions", JSON.stringify(fx.questions)]);
  assert.equal(r.code, 0, r.err);
  assert.deepEqual(JSON.parse(r.out), fx.result);
});

test("laya predict prints what Python's json.dumps(indent=2, ensure_ascii=False) prints", async () => {
  const r = await run(["predict", "--model", TINY, "--backend", "cpu", "--state", JSON.stringify(fx.state), "--questions", JSON.stringify(fx.questions)]);
  // predict.json was written by Python: json.load keeps its floats as floats (0.0 stays 0.0)
  const want = execFileSync("python3", ["-c", "import json; print(json.dumps(json.load(open('" + fixturePath("tiny", "predict.json") + "'))['result'], ensure_ascii=False, indent=2))"], { encoding: "utf8" });
  assert.equal(r.out, want);
}, { skip: pyDump ? false : "python3 not available" });

test("dumpsIndent matches json.dumps(indent=2) on nested/empty/unicode values", () => {
  const v = { a: [], b: {}, c: [1, 2.5, "ü🙂", null, true, { d: [[]] }], "é": "\n" };
  assert.equal(dumpsIndent(v) + "\n", pyDump!(v));
}, { skip: pyDump ? false : "python3 not available" });

test("pythonFloats marks result floats (1 → 1.0) and leaves ints alone", () => {
  const s = dumpsIndent(pythonFloats({ answers: { q: { type: "noul", confidence: 1, noul: 0, action: { act_probability: 1 } } }, usage: { input_tokens: 3 } }));
  assert.match(s, /"confidence": 1\.0/);
  assert.match(s, /"noul": 0\.0/);
  assert.match(s, /"act_probability": 1\.0/);
  assert.match(s, /"input_tokens": 3\n/);
});

test("--state plain text, @file and --state-file; --questions as a path", async () => {
  const { writeFileSync, rmSync, mkdtempSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const q = mkdtempSync(tmpdir() + "/laya-cli-") + "/q.json";
  writeFileSync(q, JSON.stringify(fx.questions));
  writeFileSync(q + ".state", JSON.stringify(fx.state));
  try {
    const a = await run(["predict", "--model", TINY, "--backend", "cpu", "--state-file", q + ".state", "--questions", q]);
    assert.equal(a.code, 0, a.err);
    assert.deepEqual(JSON.parse(a.out), fx.result);
    const b = await run(["predict", "--model", TINY, "--backend", "cpu", "--state", `@${q}.state`, "--questions", `@${q}`]);
    assert.deepEqual(JSON.parse(b.out), fx.result);
    const c = await run(["predict", "--model", TINY, "--backend", "cpu", "--state", "hello w1 plain text", "--questions", `@${q}`]);
    assert.equal(c.code, 0, c.err);
    assert.deepEqual(Object.keys(JSON.parse(c.out).answers), Object.keys(fx.questions as object));
  } finally {
    rmSync(q);
    rmSync(q + ".state");
  }
});

test("usage errors exit 2, runtime errors exit 1", async () => {
  assert.equal((await run([])).code, 2);
  assert.equal((await run(["predict", "--questions", "{}"])).code, 2);
  assert.equal((await run(["predict", "--state", "x"])).code, 2);
  assert.equal((await run(["predict", "--state", "x", "--questions", "not json"])).code, 2);
  assert.equal((await run(["predict", "--state", "x", "--questions", "{}", "--dtype", "f64"])).code, 2);
  assert.equal((await run(["frobnicate"])).code, 2);
  assert.equal((await run(["predict", "--bogus"])).code, 2);
  const missing = await run(["predict", "--model", "./no-such-dir", "--state", "x", "--questions", "{}"]);
  assert.equal(missing.code, 1);
  assert.match(missing.err, /Local model directory does not exist/);
  const help = await run(["--help"]);
  assert.equal(help.code, 0);
  assert.match(help.out, /laya <command>/);
});

test("bin/laya.js runs from a source checkout (re-exec under the source condition)", () => {
  const bin = fileURLToPath(new URL("../bin/laya.js", import.meta.url));
  const r = spawnSync(process.execPath, [bin, "--version"], { encoding: "utf8", env: { ...process.env, LAYA_CLI_SOURCE: "1" } });
  assert.equal(r.status, 0, r.stderr);
  assert.equal(r.stdout.trim(), "0.0.0");
});

test("bench workload and numpy-style percentiles", () => {
  const w = workload(50);
  assert.equal(Object.keys(w.questions).length, 50);
  assert.deepEqual(w.questions.q3, w.questions.q0);
  const t = timing([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  assert.equal(t.p50_ms, 5.5);
  assert.ok(Math.abs(t.p95_ms - 9.55) < 1e-12);
  assert.equal(t.mean_ms, 5.5);
});

test("laya quantize: tiny checkpoint → complete q8/q4 directory that predict loads", async () => {
  const { mkdtempSync, rmSync, existsSync, readFileSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const { readSafetensors } = await import("@johnhenry/math-plus-safetensors");
  const root = mkdtempSync(tmpdir() + "/laya-quantize-");
  try {
    for (const bits of ["8", "4"]) {
      const out = `${root}/tiny-q${bits}`;
      const r = await run(["quantize", "--model", TINY, "--bits", bits, "--out", out]);
      assert.equal(r.code, 0, r.err);
      assert.match(r.out, new RegExp(`q${bits} \\(group 64\\).*tensors quantized`));
      for (const f of ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "README.md"]) {
        assert.ok(existsSync(`${out}/${f}`), f);
      }
      assert.equal(readFileSync(`${out}/rl_agent_config.json`, "utf8"), readFileSync(`${TINY}/rl_agent_config.json`, "utf8"));
      assert.match(readFileSync(`${out}/README.md`, "utf8"), /derived from the Apache-2\.0 Laya checkpoint/);
      const meta = readSafetensors(readFileSync(`${out}/model.safetensors`)).metadata;
      assert.equal(meta.laya_quant, `q${bits}`);
      assert.equal(meta.laya_quant_source, TINY);
      const p = await run(["predict", "--model", out, "--backend", "cpu", "--state", JSON.stringify(fx.state), "--questions", JSON.stringify(fx.questions)]);
      assert.equal(p.code, 0, p.err);
      assert.deepEqual(Object.keys(JSON.parse(p.out).answers), Object.keys(fx.result.answers));
      // refuses to overwrite without --force
      const again = await run(["quantize", "--model", TINY, "--bits", bits, "--out", out]);
      assert.equal(again.code, 1);
      assert.match(again.err, /exists \(use --force/);
      assert.equal((await run(["quantize", "--model", TINY, "--bits", bits, "--out", out, "--force"])).code, 0);
    }
    const mixed = await run(["quantize", "--model", TINY, "--bits", "4", "--q8", "attn\\.", "--no-embeddings", "--out", `${root}/mixed`]);
    assert.equal(mixed.code, 0, mixed.err);
    assert.match(mixed.out, /\(12 at q8\)/); // encoder attn Wqkv+Wo × 4, head self_attn in/out_proj × 2
    const f = readSafetensors(readFileSync(`${root}/mixed/model.safetensors`));
    assert.equal(f.info("encoder.embeddings.tok_embeddings.weight").dtype, "F32");
    assert.equal(f.info("encoder.layers.0.attn.Wo.weight").dtype, "I8");
    // usage errors
    assert.equal((await run(["quantize", "--model", TINY, "--out", `${root}/x`])).code, 2);
    assert.equal((await run(["quantize", "--model", TINY, "--bits", "3", "--out", `${root}/x`])).code, 2);
    assert.equal((await run(["quantize", "--model", TINY, "--bits", "8"])).code, 2);
    assert.equal((await run(["quantize", "--model", TINY, "--bits", "8", "--q8", "x", "--out", `${root}/x`])).code, 2);
    assert.equal((await run(["quantize", "--model", TINY, "--bits", "8", "--exclude", "(", "--out", `${root}/x`])).code, 2);
    assert.equal((await run(["quantize", "--model", TINY, "--bits", "8", "--out", TINY])).code, 1);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
