/**
 * Quantized checkpoints (quant.ts): quantize/dequantize round trip with
 * bounded error, the exact on-disk layout and metadata, loader detection
 * (readWeights / load, local directory and http URL) on the tiny fixture.
 */
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
import { makeTest } from "./harness.ts";
const test = makeTest(bun);
import assert from "node:assert/strict";
import { cp, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fixturePath, loadJson } from "@johnhenry/laya-fixtures";
import { openSafetensors, readSafetensors, writeSafetensors } from "@johnhenry/math-plus-safetensors";
import type { PredictResult, Questions } from "@johnhenry/laya-core";
import {
  dequantizeMatrix,
  load,
  quantMetadata,
  quantizeMatrix,
  quantizeSafetensors,
  readWeights,
  shouldQuantize,
  QUANT_FORMAT_VERSION,
  type QuantBits,
} from "../src/index.ts";

/** Deterministic weights with a few outliers (like real checkpoints). */
function weights(rows: number, cols: number, seed = 1): Float32Array {
  const w = new Float32Array(rows * cols);
  let s = seed;
  const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0), s / 2 ** 32);
  for (let i = 0; i < w.length; i++) w[i] = (rnd() - 0.5) * 0.1 * (rnd() < 0.01 ? 20 : 1);
  return w;
}
const sse = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let e = 0;
  for (let i = 0; i < a.length; i++) e += (a[i]! - b[i]!) ** 2;
  return e;
};
const eps16 = (x: number) => Math.abs(x) * 2 ** -11 + 2 ** -24; // half an f16 ulp (+ subnormal floor)

for (const bits of [8, 4] as QuantBits[]) {
  for (const g of [64, 32]) {
    test(`q${bits} group ${g}: range fit error ≤ scale/2 per value; refinement never increases the squared error`, () => {
      const rows = 96, cols = 192;
      const w = weights(rows, cols, bits * g);
      const plain = quantizeMatrix(w, rows, cols, bits, g, { refine: false });
      const G = cols / g;
      assert.equal(plain.scales.length, rows * G);
      assert.equal(plain.data.length, bits === 8 ? rows * cols : (rows * cols) / 2);
      assert.equal(plain.data instanceof (bits === 8 ? Int8Array : Uint8Array), true);
      assert.equal(plain.biases === undefined, bits === 8);
      const d32 = dequantizeMatrix(plain, "f32");
      for (let r = 0; r < rows; r++) {
        for (let i = 0; i < cols; i++) {
          const k = r * cols + i, s = plain.scales[r * G + Math.floor(i / g)]!;
          const bound = s / 2 + eps16(w[k]!) * 2 + 1e-7;
          assert.ok(Math.abs(w[k]! - d32[k]!) <= bound, `(${r},${i}): |${w[k]} − ${d32[k]}| > ${bound}`);
        }
      }
      const refined = quantizeMatrix(w, rows, cols, bits, g);
      assert.ok(sse(w, dequantizeMatrix(refined, "f32")) <= sse(w, d32), "refined error ≤ range-fit error");
      // f16 output = the f32 output rounded once
      const d16 = dequantizeMatrix(refined, "f16");
      const r32 = dequantizeMatrix(refined, "f32");
      assert.deepEqual(Array.from(d16), Array.from(Float16Array.from(r32)));
    });
  }
}

test("q8 per row (group = cols) and all-zero groups", () => {
  const w = weights(64, 128);
  w.fill(0, 0, 128); // row 0 all zero
  const m = quantizeMatrix(w, 64, 128, 8);
  assert.equal(m.groupSize, 128);
  assert.equal(m.scales.length, 64);
  assert.equal(m.scales[0], 0);
  const d = dequantizeMatrix(m, "f32");
  for (let i = 0; i < 128; i++) assert.equal(d[i], 0);
  const rel = Math.sqrt(sse(w, d) / sse(w, new Float32Array(w.length)));
  assert.ok(rel < 0.05, `relative RMS error ${rel}`);
});

test("q4 layout: low nibble = even column, high nibble = odd column; w = q·scale + bias", () => {
  // one row, one group of 16 values: exactly the 16 levels 0..15 → q = level
  const w = Float32Array.from({ length: 16 }, (_, i) => [0, 15, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9, 7, 8][i]! * 0.5 - 1);
  const m = quantizeMatrix(w, 1, 16, 4, 16);
  assert.equal(m.scales[0], 0.5);
  assert.equal(m.biases![0], -1);
  assert.deepEqual(Array.from(m.data), [0 | (15 << 4), 1 | (14 << 4), 2 | (13 << 4), 3 | (12 << 4), 4 | (11 << 4), 5 | (10 << 4), 6 | (9 << 4), 7 | (8 << 4)]);
  assert.deepEqual(Array.from(dequantizeMatrix(m, "f32")), Array.from(w));
});

test("quantizeMatrix / dequantizeMatrix reject bad shapes", () => {
  assert.throws(() => quantizeMatrix(new Float32Array(10), 2, 4, 8), /values for/);
  assert.throws(() => quantizeMatrix(new Float32Array(12), 2, 6, 8, 4), /must divide/);
  assert.throws(() => quantizeMatrix(new Float32Array(6), 2, 3, 4, 3), /even/);
  assert.throws(() => quantizeMatrix(new Float32Array(4), 1, 4, 2 as QuantBits), /bits must be 8 or 4/);
  const m = quantizeMatrix(weights(2, 64), 2, 64, 4);
  assert.throws(() => dequantizeMatrix({ ...m, biases: undefined }, "f16"), /biases/);
  assert.throws(() => dequantizeMatrix({ ...m, scales: new Float16Array(1) }, "f16"), /scales/);
});

test("quantMetadata: null for float checkpoints, parsed scheme, loud errors for unknown schemes/versions", () => {
  assert.equal(quantMetadata({}), null);
  assert.equal(quantMetadata(undefined), null);
  assert.deepEqual(quantMetadata({ laya_quant: "q4", group_size: "64", version: "1", laya_quant_source_dtype: "F16" }), {
    scheme: "q4", bits: 4, groupSize: 64, version: "1", sourceDtype: "F16",
  });
  assert.equal(quantMetadata({ laya_quant: "q8", group_size: "row", version: "1" })!.groupSize, null);
  assert.throws(() => quantMetadata({ laya_quant: "q3", version: "1" }), /unknown quantization scheme/);
  assert.throws(() => quantMetadata({ laya_quant: "q8", version: "2" }), /format version "2" is not supported/);
  assert.throws(() => quantMetadata({ laya_quant: "q8", version: "1", group_size: "0" }), /invalid group_size/);
});

test("shouldQuantize picks the large 2-D weights only", () => {
  const o = { bits: 4 as const };
  assert.ok(shouldQuantize("encoder.layers.0.attn.Wqkv.weight", "F16", [3072, 1024], o));
  assert.ok(shouldQuantize("encoder.embeddings.tok_embeddings.weight", "F16", [50368, 1024], o));
  assert.ok(!shouldQuantize("encoder.embeddings.tok_embeddings.weight", "F16", [50368, 1024], { ...o, embeddings: false }));
  assert.ok(shouldQuantize("head.layers.1.linear2.weight", "F16", [1024, 4096], o));
  assert.ok(!shouldQuantize("encoder.layers.1.attn_norm.weight", "F16", [1024], o));
  assert.ok(!shouldQuantize("head.layers.0.norm1.weight", "F16", [1024], o));
  assert.ok(!shouldQuantize("head.layers.0.linear1.bias", "F16", [4096], o));
  assert.ok(!shouldQuantize("type_emb.weight", "F16", [3, 1024], o));
  assert.ok(!shouldQuantize("scorer.layers.3.weight", "F16", [1, 1024], o));
  assert.ok(!shouldQuantize("act_head.layers.0.weight", "F16", [256, 1028], o), "1028 is not a multiple of 64");
  assert.ok(shouldQuantize("act_head.layers.0.weight", "F16", [256, 1028], { bits: 8, groupSize: "row" }));
  assert.ok(!shouldQuantize("encoder.layers.0.mlp.Wi.weight", "F16", [5248, 1024], { ...o, exclude: [/mlp/] }));
  assert.ok(!shouldQuantize("x.weight", "I32", [128, 128], o));
});

// ------------------------------------------------------------------ tiny checkpoint
const TINY = fixturePath("tiny");
const fx = await loadJson<{ state: any; questions: Questions; result: PredictResult }>("tiny", "predict.json");
const tinyBytes = await readFile(join(TINY, "model.safetensors"));

async function quantizedTiny(bits: QuantBits, extra: object = {}) {
  const file = await openSafetensors(tinyBytes);
  return quantizeSafetensors(file, { bits, metadata: { laya_quant_source: "tiny" }, ...extra });
}

/** Tiny checkpoint dir with the quantized weights (configs/tokenizer copied). */
async function tinyDir(bytes: Uint8Array): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "laya-quant-"));
  await cp(TINY, dir, { recursive: true });
  await writeFile(join(dir, "model.safetensors"), bytes);
  return dir;
}

function maxDelta(got: PredictResult, want: PredictResult): number {
  let m = 0;
  for (const [qid, w] of Object.entries(want.answers) as [string, any][]) {
    const g = got.answers[qid] as any;
    m = Math.max(m, Math.abs(g.confidence - w.confidence), Math.abs(g.action.act_probability - w.action.act_probability));
    for (const k of ["noul", "score"]) if (w[k] !== undefined) m = Math.max(m, Math.abs(g[k] - w[k]));
    for (const [l, p] of Object.entries(w.probabilities ?? {}) as [string, number][]) m = Math.max(m, Math.abs(g.probabilities[l] - p));
  }
  return m;
}

for (const bits of [8, 4] as QuantBits[]) {
  test(`quantizeSafetensors q${bits} on the tiny checkpoint: exact metadata, companions, untouched small tensors`, async () => {
    const { bytes, report } = await quantizedTiny(bits);
    const f = readSafetensors(bytes);
    assert.deepEqual({ ...f.metadata }, { laya_quant_source: "tiny", laya_quant: `q${bits}`, group_size: "64", version: QUANT_FORMAT_VERSION, laya_quant_source_dtype: "F32" });
    const src = readSafetensors(tinyBytes);
    assert.equal(report.quantized.length + report.kept.length, src.names().length);
    assert.equal(report.bytesIn, src.names().reduce((n, k) => n + src.bytes(k).byteLength, 0));
    assert.equal(report.bytesOut, bytes.byteLength);
    assert.ok(report.quantized.includes("encoder.layers.0.attn.Wqkv.weight"));
    assert.ok(report.quantized.includes("encoder.embeddings.tok_embeddings.weight"));
    for (const name of report.quantized) {
      const [rows, cols] = src.info(name).shape as [number, number];
      assert.deepEqual(f.info(name).shape, bits === 8 ? [rows, cols] : [rows, cols / 2]);
      assert.equal(f.info(name).dtype, bits === 8 ? "I8" : "U8");
      assert.deepEqual(f.info(name + ".scales"), { ...f.info(name + ".scales"), dtype: "F16", shape: [rows, cols / 64] });
      assert.equal(f.has(name + ".biases"), bits === 4);
    }
    for (const name of report.kept) {
      assert.ok(Buffer.from(f.bytes(name)).equals(Buffer.from(src.bytes(name))), name);
      assert.ok(!f.has(name + ".scales"));
    }
    assert.ok(report.kept.includes("temperature") && report.kept.includes("type_emb.weight") && report.kept.includes("encoder.final_norm.weight"));
    // the weights come back within the per-group bound
    const ws = await readWeights(bytes, { dtype: "f32" });
    const name = "encoder.layers.1.mlp.Wi.weight";
    const got = ws.get(name)!;
    assert.equal(got.dtype, "f32");
    assert.deepEqual(got.shape, src.info(name).shape);
    const want = src.toF32(name), scales = f.toF32(name + ".scales");
    const cols = got.shape[1]!;
    for (let i = 0; i < want.length; i++) {
      const s = scales[Math.floor(i / cols) * (cols / 64) + Math.floor((i % cols) / 64)]!;
      assert.ok(Math.abs((got.data as Float32Array)[i]! - want[i]!) <= (bits === 8 ? 1 : 1.5) * s + 1e-6, `${i}`);
    }
    assert.equal(ws.get(name), undefined, "consumed");
    assert.ok(!ws.remaining().includes(name));
    assert.ok(!ws.remaining().some((n) => n.endsWith(".scales") || n.endsWith(".biases")), "companions are never listed");
  });

  test(`load() detects a q${bits} checkpoint directory and predicts close to the float result (cpu)`, async () => {
    const { bytes } = await quantizedTiny(bits);
    const dir = await tinyDir(bytes);
    try {
      const agent = await load(dir, { backend: "cpu", batchSize: 2, warn: () => {} });
      const got = await agent.predict(fx.state, fx.questions);
      agent.dispose();
      assert.deepEqual(Object.keys(got.answers), Object.keys(fx.result.answers));
      assert.deepEqual(got.usage, fx.result.usage);
      const d = maxDelta(got, fx.result);
      assert.ok(d <= (bits === 8 ? 2e-3 : 2e-2), `max |Δ| ${d}`);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
}

test("q4 with q8 patterns stores the matching tensors as I8 (mixed precision) and loads", async () => {
  const { bytes, report } = await quantizedTiny(4, { q8: [/attn\./] });
  const f = readSafetensors(bytes);
  assert.equal(f.info("encoder.layers.0.attn.Wqkv.weight").dtype, "I8");
  assert.equal(f.has("encoder.layers.0.attn.Wqkv.weight.biases"), false);
  assert.equal(f.info("encoder.layers.0.mlp.Wi.weight").dtype, "U8");
  assert.deepEqual(report.promoted.sort(), report.quantized.filter((n) => /attn\./.test(n)).sort());
  const ws = await readWeights(bytes);
  const t = ws.get("encoder.layers.0.attn.Wqkv.weight")!;
  assert.equal(t.dtype, "f16");
  assert.ok(t.data instanceof Float16Array);
});

test("readWeights: plain checkpoints are untouched; corrupt quantized files fail loudly", async () => {
  const plain = await readWeights(tinyBytes, { dtype: "f16" });
  assert.equal(plain.get("encoder.layers.0.attn.Wqkv.weight")!.dtype, "f32", "dtype only applies to dequantized tensors");
  const { bytes } = await quantizedTiny(4);
  const f = readSafetensors(bytes);
  const tensors = new Map(f.names().map((n) => [n, { dtype: f.info(n).dtype, shape: f.info(n).shape, data: f.bytes(n) }]));
  tensors.delete("encoder.layers.0.attn.Wqkv.weight.biases");
  await assert.rejects(readWeights(writeSafetensors(tensors, { ...f.metadata })), /q4 needs \.biases/);
  await assert.rejects(readWeights(writeSafetensors(new Map(), { laya_quant: "q4", version: "9" })), /format version "9"/);
  await assert.rejects(quantizeSafetensors(await openSafetensors(bytes), { bits: 8 }), /already quantized/);
});

test("load() from an http URL in Node: quantized checkpoint over Range requests", async () => {
  const { bytes } = await quantizedTiny(8);
  const dir = await tinyDir(bytes);
  let ranges = 0;
  const server = createServer(async (req, res) => {
    try {
      const body = await readFile(join(dir, decodeURIComponent(req.url!.replace(/^\/m\//, ""))));
      const range = req.headers.range?.match(/bytes=(\d+)-(\d*)/);
      if (range) {
        ranges++;
        const start = Number(range[1]), end = range[2] ? Math.min(Number(range[2]), body.length - 1) : body.length - 1;
        res.writeHead(206, { "Content-Range": `bytes ${start}-${end}/${body.length}`, "Content-Length": end - start + 1 });
        res.end(body.subarray(start, end + 1));
      } else res.writeHead(200, { "Content-Length": body.length }).end(body);
    } catch {
      res.writeHead(404).end();
    }
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  try {
    const url = `http://127.0.0.1:${(server.address() as { port: number }).port}/m`;
    const agent = await load(url, { backend: "cpu", batchSize: 2, warn: () => {} });
    const got = await agent.predict(fx.state, fx.questions);
    agent.dispose();
    assert.ok(maxDelta(got, fx.result) <= 2e-3);
    assert.ok(ranges >= 1, "weights were read with Range requests");
    await assert.rejects(load(url + "/nope", { backend: "cpu" }), /Not a complete Laya checkpoint/);
  } finally {
    server.close();
    await rm(dir, { recursive: true, force: true });
  }
});
