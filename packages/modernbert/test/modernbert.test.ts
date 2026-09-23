import * as nodeTest from "node:test";
// Bun 1.2's node:test shim only registers tests from the first file of a run.
// @ts-ignore -- bun types are not installed
const bunTest: unknown = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const { test } = (bunTest ?? nodeTest) as Pick<typeof nodeTest, "test">;
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createCpuBackend, type CpuTensor } from "@johnhenry/backend-cpu";
import { fixturePath, loadJson, type EncodedTensor } from "@johnhenry/laya-fixtures";
import { readSafetensors } from "@johnhenry/math-plus-safetensors";
import { decodeTensor, loadOpCases } from "@johnhenry/tensor-backend/conformance";
import { toF32 } from "@johnhenry/tensor-backend";
import {
  attentionMasks,
  detectPrefix,
  loadModernBert,
  parseModernBertConfig,
  safetensorsWeights,
  type WeightGetter,
} from "../src/index.ts";

interface Activations {
  inputs: Record<string, EncodedTensor>;
  stages: Record<string, EncodedTensor>;
  layer_types: string[];
}

const tinyConfig = () => loadJson<Record<string, unknown>>("tiny", "encoder", "config.json");
const tinyWeights = async () => safetensorsWeights(readSafetensors(await readFile(fixturePath("tiny", "model.safetensors"))));

test("parseModernBertConfig: defaults, layer_types, rope bases", async () => {
  const act = await loadJson<Activations>("tiny", "activations.json");
  const c = parseModernBertConfig(await tinyConfig());
  assert.deepEqual(c.layerTypes, act.layer_types); // oracle: EncoderConfig.from_dict in the fixture
  assert.equal(c.headDim, 32);
  assert.equal(c.normEps, 1e-5);
  assert.equal(c.normBias, false);
  assert.deepEqual(c.ropeBase, { full_attention: 160000, sliding_attention: 10000 });
  const r = parseModernBertConfig({
    ...(await tinyConfig()),
    global_rope_theta: 1, rope_parameters: { full_attention: { rope_theta: 5, rope_type: "default" }, sliding_attention: { rope_theta: 7 } },
  });
  assert.deepEqual(r.ropeBase, { full_attention: 5, sliding_attention: 7 });
  const base = await tinyConfig();
  assert.throws(() => parseModernBertConfig({ ...base, model_type: "bert" }), /Unsupported encoder/);
  assert.throws(() => parseModernBertConfig({ ...base, hidden_activation: "silu" }), /activation/);
  assert.throws(() => parseModernBertConfig({ ...base, num_attention_heads: 3 }), /head dimension/);
  assert.throws(() => parseModernBertConfig({ ...base, layer_types: ["full_attention"] }), /layer_types/);
  assert.throws(() => parseModernBertConfig({ ...base, rope_parameters: { full_attention: { rope_type: "yarn" } } }), /RoPE/);
});

test("attentionMasks reproduces laya-mlx attention_masks (ops.json sdpa masks)", async () => {
  const cases = await loadOpCases();
  const full = decodeTensor(cases.find((c) => c.name === "sdpa/full_attention")!.inputs[3]!);
  const sliding = decodeTensor(cases.find((c) => c.name === "sdpa/sliding_attention")!.inputs[3]!);
  const [B, , , L] = full.shape as [number, number, number, number];
  const valid = full.data as Uint8Array; // [B,1,1,L] == valid
  const got = attentionMasks(valid, B, L, 16); // generator used window 16
  assert.deepEqual(got.full.shape, full.shape);
  assert.deepEqual([...got.full.data], [...full.data]);
  assert.deepEqual(got.sliding.shape, sliding.shape);
  assert.deepEqual([...got.sliding.data], [...sliding.data]);
});

test("tiny encoder: every stage matches MLX within 1e-5 on CPU; HF 'model.' names load identically", async () => {
  const act = await loadJson<Activations>("tiny", "activations.json");
  const config = parseModernBertConfig(await tinyConfig());
  const get = await tinyWeights();
  assert.equal(detectPrefix(get), "encoder.");
  const backend = createCpuBackend();
  const enc = loadModernBert(backend, config, get);
  const ids = decodeTensor(act.inputs.input_ids!), mask = decodeTensor(act.inputs.attention_mask!);
  const [B, L] = ids.shape as [number, number];
  const stages = new Map<string, CpuTensor>();
  const out = enc.forward(ids.data as Int32Array, mask.data as Uint8Array, B, L, {
    onStage: (name, t) => (stages.set(name, t), true),
  });
  assert.equal(stages.get("final_norm"), out);
  const fixtureName = (n: string) => (n === "embeddings" ? n : `encoder.${n}`);
  let worst = 0;
  for (const [name, t] of stages) {
    const want = decodeTensor(act.stages[fixtureName(name)]!).data as Float32Array;
    const got = toF32(await backend.read(t));
    for (let i = 0; i < want.length; i++) {
      const d = Math.abs(got[i]! - want[i]!);
      worst = Math.max(worst, d);
      assert.ok(d <= 1e-5 + 1e-5 * Math.abs(want[i]!), `${name}[${i}] ${got[i]} vs ${want[i]}`);
    }
  }
  assert.equal(stages.size, 2 + config.numHiddenLayers);
  console.log(`tiny encoder max abs err ${worst.toExponential(2)}`);

  // Same weights under HF names: model.embeddings..., model.layers.N..., model.final_norm...
  const hf: WeightGetter = (n) => (n.startsWith("model.") ? get("encoder." + n.slice(6)) : undefined);
  assert.equal(detectPrefix(hf), "model.");
  const enc2 = loadModernBert(backend, config, { get: hf });
  const out2 = enc2.forward(ids.data as Int32Array, mask.data as Uint8Array, B, L);
  assert.deepEqual(toF32(await backend.read(out2)), toF32(await backend.read(out)));

  // embed(): masked mean of final_norm over valid tokens (laya-mlx embed_fn_from_agent)
  const pooled = await enc.embedToHost(ids.data as Int32Array, mask.data as Uint8Array, B, L);
  const fin = decodeTensor(act.stages["encoder.final_norm"]!).data as Float32Array;
  const H = config.hiddenSize, m = mask.data as Uint8Array;
  for (let b = 0; b < B; b++) {
    let n = 0;
    for (let l = 0; l < L; l++) n += m[b * L + l]!;
    for (let d = 0; d < H; d++) {
      let s = 0;
      for (let l = 0; l < L; l++) if (m[b * L + l]) s += fin[(b * L + l) * H + d]!;
      assert.ok(Math.abs(pooled[b * H + d]! - s / n) <= 1e-5, "embed");
    }
  }
  for (const t of stages.values()) backend.dispose(t);
  backend.dispose(out2);
  enc.dispose();
  enc2.dispose();
});

test("loadModernBert validates names and shapes", async () => {
  const config = parseModernBertConfig(await tinyConfig());
  const get = await tinyWeights();
  const backend = createCpuBackend();
  assert.throws(() => loadModernBert(backend, config, () => undefined), /no embeddings\.tok_embeddings\.weight/);
  const noNorm: WeightGetter = (n) => (n === "encoder.layers.2.attn_norm.weight" ? undefined : get(n));
  assert.throws(() => loadModernBert(backend, config, noNorm), /missing weight encoder\.layers\.2\.attn_norm\.weight/);
  const bigger = parseModernBertConfig({ ...(await tinyConfig()), intermediate_size: 100 });
  assert.throws(() => loadModernBert(backend, bigger, get), /Wi\.weight has shape \[192,64\], want \[200,64\]/);
  assert.throws(() => loadModernBert(backend, config, get, { dtype: "f16" }), /does not support f16/);
});
