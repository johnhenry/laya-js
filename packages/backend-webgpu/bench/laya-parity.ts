/**
 * Laya DecisionModel on the WebGPU backend vs the Python MLX fixtures.
 * Workspace-only script (imports @johnhenry/laya's model by path).
 *
 *   node --conditions=source bench/laya-parity.ts [--real] [--dtype f16|f32|both]
 *
 * 1. tiny checkpoint: every stage vs activations.json (f32 and f16).
 * 2. --real: English checkpoint (fp16 safetensors from the local HF cache),
 *    all 63 fixture questions one at a time: max |Δlogit|, argmax agreement,
 *    and latency of the shortest question.
 */
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { readSafetensors } from "@johnhenry/math-plus-safetensors";
import { safetensorsWeights } from "@johnhenry/modernbert";
import { fixturePath, loadJson, loadReal, type EncodedTensor } from "@johnhenry/laya-fixtures";
import { decodeTensor } from "@johnhenry/tensor-backend/conformance";
import { toF32 } from "@johnhenry/tensor-backend";
import type { AgentConfig, Batch } from "@johnhenry/laya-core";
import { loadDecisionModel } from "../../laya/src/model.ts";
import { collateItems } from "../../laya/test/helpers.ts";
import { createWebGpuBackend, type WebGpuTensor } from "../src/index.ts";

const args = process.argv.slice(2);
const dtArg = args.includes("--dtype") ? args[args.indexOf("--dtype") + 1]! : "both";
const dtypes = (dtArg === "both" ? ["f32", "f16"] : [dtArg]) as ("f32" | "f16")[];

const backend = await createWebGpuBackend();
const { limits: _l, ...info } = backend.adapterInfo;
console.log("adapter:", JSON.stringify(info));

// ---------------------------------------------------------------- tiny
interface Activations {
  inputs: Record<"input_ids" | "attention_mask" | "marker_pos" | "marker_mask" | "qtype", EncodedTensor>;
  stages: Record<string, EncodedTensor>;
}
{
  const act = await loadJson<Activations>("tiny", "activations.json");
  const file = readSafetensors(await readFile(fixturePath("tiny", "model.safetensors")));
  const ids = decodeTensor(act.inputs.input_ids), mask = decodeTensor(act.inputs.attention_mask);
  const mpos = decodeTensor(act.inputs.marker_pos);
  const [B, L] = ids.shape as [number, number];
  const batch: Batch = {
    size: B, length: L, markerCount: mpos.shape[1]!,
    inputIds: ids.data as Int32Array, attentionMask: mask.data as Uint8Array,
    markerPos: mpos.data as Int32Array, markerMask: decodeTensor(act.inputs.marker_mask).data as Uint8Array,
    qtype: decodeTensor(act.inputs.qtype).data as Int32Array,
  };
  for (const dtype of dtypes) {
    const model = await loadDecisionModel(backend, {
      encoderConfig: await loadJson("tiny", "encoder", "config.json"),
      agentConfig: await loadJson("tiny", "rl_agent_config.json"),
      weights: safetensorsWeights(file),
      dtype,
    });
    const staged = new Map<string, WebGpuTensor>();
    const { logits, act: a } = await model.forwardTensors(batch, { onStage: (n, t) => (staged.set(n, t), true) });
    staged.set("logits", logits);
    staged.set("act", a);
    let worst = 0, worstName = "";
    const D = model.hiddenSize;
    for (const [name, enc] of Object.entries(act.stages)) {
      const want = decodeTensor(enc).data as Float32Array;
      const got = toF32(await backend.read(staged.get(name)!));
      let m = 0;
      for (let i = 0; i < want.length; i++) {
        // hidden-state stages: valid (unpadded) positions only
        if (enc.shape.length === 3 && !batch.attentionMask[Math.floor(i / D)]) continue;
        m = Math.max(m, Math.abs(got[i]! - want[i]!));
      }
      if (m > worst) (worst = m), (worstName = name);
      if (name === "logits" || name === "act") console.log(`tiny ${dtype} ${name}: max|Δ| ${m.toExponential(2)}`);
    }
    console.log(`tiny ${dtype}: worst stage ${worstName} max|Δ| ${worst.toExponential(2)} (valid positions)`);
    for (const t of staged.values()) backend.dispose(t);
    model.dispose();
  }
}

// ---------------------------------------------------------------- real English
if (args.includes("--real")) {
  const fx = await loadReal("english");
  const path = `${fx.model_dir}/model.safetensors`;
  if (!existsSync(path)) {
    console.log(`SKIP real: ${path} not in the HF cache`);
  } else {
    const encoderConfig = JSON.parse(await readFile(`${fx.model_dir}/encoder/config.json`, "utf8")) as Record<string, unknown>;
    const tok = await loadJson<{ special: Record<string, [string, number]> }>("tables", "tokenizer-english.json");
    const padId = tok.special.pad_token![1];
    const t0 = performance.now();
    const file = readSafetensors(await readFile(path));
    console.log(`read safetensors: ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    const argmax = (xs: ArrayLike<number>, n: number) => {
      let best = 0;
      for (let i = 1; i < n; i++) if (xs[i]! > xs[best]!) best = i;
      return best;
    };
    for (const dtype of dtypes) {
      const tl = performance.now();
      const model = await loadDecisionModel(backend, { encoderConfig, agentConfig: fx.config as AgentConfig, weights: safetensorsWeights(file), dtype });
      await backend.sync();
      const loadS = (performance.now() - tl) / 1000;
      let maxLogit = 0, maxAct = 0, agree = 0, total = 0;
      let shortest = { len: Infinity, batch: null as Batch | null };
      const tr = performance.now();
      for (const c of fx.cases) {
        c.items.forEach((item, i) => {
          if (item.ids.length < shortest.len) shortest = { len: item.ids.length, batch: collateItems([item], padId) };
        });
        for (let i = 0; i < c.items.length; i++) {
          const out = await model.forward(collateItems([c.items[i]!], padId));
          const want = c.outputs[i]!, k = want.logits.length;
          for (let j = 0; j < k; j++) maxLogit = Math.max(maxLogit, Math.abs(out.logits[j]! - want.logits[j]!));
          for (let j = 0; j < want.act.length; j++) maxAct = Math.max(maxAct, Math.abs(out.act[j]! - want.act[j]!));
          if (argmax(out.logits, k) === argmax(want.logits, k)) agree++;
          total++;
        }
      }
      const runS = (performance.now() - tr) / 1000;
      // Latency: shortest question, B=1, warm.
      const times: number[] = [];
      for (let i = 0; i < 25; i++) {
        const ts = performance.now();
        await model.forward(shortest.batch!);
        if (i >= 5) times.push(performance.now() - ts);
      }
      times.sort((a, b) => a - b);
      console.log(
        `real english ${dtype}: load ${loadS.toFixed(1)} s; ${total} questions in ${runS.toFixed(1)} s; ` +
          `argmax ${agree}/${total}; max|Δlogit| ${maxLogit.toExponential(3)}; max|Δact| ${maxAct.toExponential(3)}; ` +
          `latency (L=${shortest.len}, B=1) median ${times[times.length >> 1]!.toFixed(1)} ms, min ${times[0]!.toFixed(1)} ms`,
      );
      model.dispose();
      backend.rt.trim();
    }
  }
}
console.log("stats:", JSON.stringify(backend.rt.stats));
backend.destroy();
