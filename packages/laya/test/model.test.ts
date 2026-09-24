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
import { safetensorsWeights } from "@johnhenry/modernbert";
import { decodeTensor } from "@johnhenry/tensor-backend/conformance";
import { toF32 } from "@johnhenry/tensor-backend";
import type { Batch } from "@johnhenry/laya-core";
import { loadDecisionModel } from "../src/model.ts";
import { errStats, fmt } from "./helpers.ts";

interface Activations {
  inputs: Record<"input_ids" | "attention_mask" | "marker_pos" | "marker_mask" | "qtype", EncodedTensor>;
  stages: Record<string, EncodedTensor>;
  layer_types: string[];
}

const TOL = 1e-5;

test("tiny checkpoint: every stage matches MLX within 1e-5 (CPU backend)", async () => {
  const act = await loadJson<Activations>("tiny", "activations.json");
  const encoderConfig = await loadJson<Record<string, unknown>>("tiny", "encoder", "config.json");
  const agentConfig = await loadJson<Record<string, unknown>>("tiny", "rl_agent_config.json");
  const file = readSafetensors(await readFile(fixturePath("tiny", "model.safetensors")));
  const backend = createCpuBackend();
  const model = await loadDecisionModel(backend, { encoderConfig, agentConfig, weights: safetensorsWeights(file) });

  const ids = decodeTensor(act.inputs.input_ids), mask = decodeTensor(act.inputs.attention_mask);
  const mpos = decodeTensor(act.inputs.marker_pos), mmask = decodeTensor(act.inputs.marker_mask);
  const [B, L] = ids.shape as [number, number];
  const M = mpos.shape[1]!;
  const batch: Batch = {
    size: B, length: L, markerCount: M,
    inputIds: ids.data as Int32Array, attentionMask: mask.data as Uint8Array,
    markerPos: mpos.data as Int32Array, markerMask: mmask.data as Uint8Array,
    qtype: decodeTensor(act.inputs.qtype).data as Int32Array,
  };
  const staged = new Map<string, CpuTensor>();
  const { logits, act: actT } = await model.forwardTensors(batch, { onStage: (name, t) => (staged.set(name, t), true) });
  staged.set("logits", logits);
  staged.set("act", actT);
  assert.deepEqual([...staged.keys()].sort(), Object.keys(act.stages).sort(), "stage names");

  const valid: number[] = [], padded: number[] = [];
  const D = model.hiddenSize;
  for (let p = 0; p < B * L; p++) for (let d = 0; d < D; d++) ((batch.attentionMask[p] ? valid : padded)).push(p * D + d);
  const report: string[] = [];
  let worstValid = 0, worstPadded = 0;
  for (const [name, enc] of Object.entries(act.stages)) {
    const want = decodeTensor(enc);
    const got = await backend.read(staged.get(name)!);
    assert.deepEqual(got.shape, want.shape, name);
    const g = toF32(got), w = want.data as Float32Array;
    if (want.shape.length === 3) {
      const sv = errStats(g, w, TOL, TOL, valid), sp = errStats(g, w, TOL, TOL, padded);
      worstValid = Math.max(worstValid, sv.maxAbs);
      worstPadded = Math.max(worstPadded, sp.maxAbs);
      report.push(`${name.padEnd(20)} valid ${fmt(sv)} | padded ${fmt(sp)}`);
      assert.ok(sv.worst <= 1, `${name} valid positions: ${fmt(sv)}`);
      assert.ok(sp.worst <= 1, `${name} padded positions: ${fmt(sp)}`);
    } else {
      const s = errStats(g, w, TOL, TOL);
      report.push(`${name.padEnd(20)} ${fmt(s)}`);
      assert.ok(s.worst <= 1, `${name}: ${fmt(s)}`);
    }
  }
  // masked logit slots are exactly -1e4
  const lg = toF32(await backend.read(logits));
  for (let i = 0; i < B * M; i++) if (!batch.markerMask[i]) assert.equal(lg[i], -1e4);
  console.log(`tiny parity (max abs, valid ${worstValid.toExponential(2)}, padded ${worstPadded.toExponential(2)}):\n  ` + report.join("\n  "));
  for (const t of staged.values()) backend.dispose(t);
  model.dispose();
});

test("forward() returns host logits/act with nAct and frees intermediates", async () => {
  const act = await loadJson<Activations>("tiny", "activations.json");
  const file = readSafetensors(await readFile(fixturePath("tiny", "model.safetensors")));
  const backend = createCpuBackend();
  const model = await loadDecisionModel(backend, {
    encoderConfig: await loadJson("tiny", "encoder", "config.json"),
    agentConfig: await loadJson("tiny", "rl_agent_config.json"),
    weights: safetensorsWeights(file),
  });
  const ids = decodeTensor(act.inputs.input_ids);
  const [B, L] = ids.shape as [number, number];
  const out = await model.forward({
    size: B, length: L, markerCount: 3,
    inputIds: ids.data as Int32Array,
    attentionMask: decodeTensor(act.inputs.attention_mask).data as Uint8Array,
    markerPos: decodeTensor(act.inputs.marker_pos).data as Int32Array,
    markerMask: decodeTensor(act.inputs.marker_mask).data as Uint8Array,
    qtype: decodeTensor(act.inputs.qtype).data as Int32Array,
  });
  assert.equal(out.nAct, 2);
  assert.equal(out.logits.length, B * 3);
  assert.equal(out.act.length, B * 2);
  const want = decodeTensor(act.stages.act!).data as Float32Array;
  assert.ok(errStats(out.act, want, TOL, TOL).worst <= 1);
  model.dispose();
});

test("loadDecisionModel rejects missing weights and wrong shapes", async () => {
  const file = readSafetensors(await readFile(fixturePath("tiny", "model.safetensors")));
  const get = safetensorsWeights(file);
  const encoderConfig = await loadJson<Record<string, unknown>>("tiny", "encoder", "config.json");
  // Count live uploads: a rejected load must free every tensor it uploaded.
  const cpu = createCpuBackend();
  const live = new Set<CpuTensor>();
  const backend = new Proxy(cpu, {
    get(t, p) {
      if (p === "fromHost") return async (h: Parameters<typeof cpu.fromHost>[0]) => { const x = await t.fromHost(h); live.add(x); return x; };
      if (p === "dispose") return (x: CpuTensor) => { live.delete(x); t.dispose(x); };
      const v = Reflect.get(t, p, t) as unknown;
      return typeof v === "function" ? (v as (...a: unknown[]) => unknown).bind(t) : v;
    },
  });
  await assert.rejects(
    () => loadDecisionModel(backend, { encoderConfig, agentConfig: { head_layers: 3 }, weights: get }),
    /missing weight head\.layers\.2/,
  );
  assert.equal(live.size, 0, "encoder uploads freed after a head validation error");
  await assert.rejects(
    () => loadDecisionModel(backend, { encoderConfig, agentConfig: { head_layers: 2, act_costs: { a: 1, b: 2 } }, weights: get }),
    /act_head\.layers\.2\.weight has shape \[2,256\], want \[3,256\]/,
  );
  assert.equal(live.size, 0);
  const ok = await loadDecisionModel(backend, { encoderConfig, agentConfig: await loadJson("tiny", "rl_agent_config.json"), weights: get });
  assert.ok(live.size > 0);
  ok.dispose();
  assert.equal(live.size, 0, "dispose frees every weight and constant");
});
