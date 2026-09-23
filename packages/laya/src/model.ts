/**
 * Laya DecisionModel on any @johnhenry/tensor-backend Backend: ModernBERT
 * encoder + type embedding + PyTorch-style decision head layers + scorer +
 * action head. Exact port of laya-mlx `DecisionModel.__call__`
 * (laya_mlx/model.py). Tensor-in / numbers-out; prompts, collation and
 * result formatting live in @johnhenry/laya-core.
 */
import type { Backend, DType, HostTensor, Tensor } from "@johnhenry/tensor-backend";
import { toF32 } from "@johnhenry/tensor-backend";
import {
  loadModernBert,
  parseModernBertConfig,
  toWeightGetter,
  uploadAs,
  type ModernBert,
  type ModernBertConfig,
  type NormWeights,
  type WeightSource,
} from "@johnhenry/modernbert";
import type { AgentConfig, Batch, BatchOutputs } from "@johnhenry/laya-core";

/** PyTorch nn.LayerNorm / MLX nn.LayerNorm default. */
const HEAD_EPS = 1e-5;

export interface HeadLayerWeights<T> {
  readonly norm1: NormWeights<T>;
  readonly norm2: NormWeights<T>;
  readonly inProj: T;
  readonly inProjBias: T;
  readonly outProj: T;
  readonly outProjBias: T;
  readonly linear1: T;
  readonly linear1Bias: T;
  readonly linear2: T;
  readonly linear2Bias: T;
}

export interface DecisionWeights<T> {
  readonly head: readonly HeadLayerWeights<T>[];
  readonly typeEmb: T;
  /** scorer = LayerNorm → Linear(D, D) → GELU → Linear(D, 1) */
  readonly scorer: { readonly norm: NormWeights<T>; readonly w1: T; readonly b1: T; readonly w2: T; readonly b2: T };
  /** act_head = Linear(D + 4, 256) → GELU → Linear(256, nAct) */
  readonly act: { readonly w1: T; readonly b1: T; readonly w2: T; readonly b2: T };
}

export interface LoadDecisionModelOptions {
  /** encoder/config.json (raw JSON) or an already parsed ModernBertConfig. */
  readonly encoderConfig: ModernBertConfig | Record<string, unknown>;
  /** rl_agent_config.json (head_layers, act_costs). */
  readonly agentConfig: AgentConfig;
  /** Checkpoint tensors by name (laya-mlx names; upstream PyTorch names are also accepted). */
  readonly weights: WeightSource;
  /** Storage/compute dtype (default "f32"). Logits, softmax and act outputs are always f32. */
  readonly dtype?: "f32" | "f16" | "bf16";
}

export interface DecisionForwardOptions<T> {
  /**
   * Stage observer, names as in the laya-js fixtures: "embeddings",
   * "encoder.layers.<i>", "encoder.final_norm", "type_emb_added",
   * "head.layers.<j>". Return true to take ownership of the tensor
   * (you must dispose it); otherwise it is freed by the forward pass.
   */
  readonly onStage?: (name: string, t: T) => boolean | void;
}

export class DecisionModel<T extends Tensor = Tensor> {
  readonly backend: Backend<T>;
  readonly encoder: ModernBert<T>;
  readonly weights: DecisionWeights<T>;
  readonly dtype: DType;
  /** Number of action logits (len(act_costs) + 1). */
  readonly nAct: number;
  readonly hiddenSize: number;
  /** Decision-head attention heads: max(1, D // 64). */
  readonly headHeads: number;

  constructor(backend: Backend<T>, encoder: ModernBert<T>, weights: DecisionWeights<T>, dtype: DType, nAct: number) {
    this.backend = backend;
    this.encoder = encoder;
    this.weights = weights;
    this.dtype = dtype;
    this.nAct = nAct;
    this.hiddenSize = encoder.config.hiddenSize;
    this.headHeads = Math.max(1, Math.floor(this.hiddenSize / 64));
  }

  /** PyTorch pre-norm nn.TransformerEncoderLayer (ReLU FFN), mask bool [B, 1, 1, L]. */
  headLayer(j: number, x: T, mask: T): T {
    const b = this.backend, w = this.weights.head[j]!;
    return b.scope(() => {
      const [B, L, D] = x.shape as [number, number, number];
      const nh = this.headHeads, hd = D / nh;
      const h = b.layerNorm(x, w.norm1.weight, w.norm1.bias, HEAD_EPS);
      const qkv = b.transpose(b.reshape(b.linear(h, w.inProj, w.inProjBias), [B, L, 3, nh, hd]), [2, 0, 3, 1, 4]);
      const [q, k, v] = b.split(qkv, 3, 0).map((t) => b.reshape(t, [B, nh, L, hd])) as [T, T, T];
      const att = b.reshape(b.transpose(b.sdpa(q, k, v, mask, hd ** -0.5), [0, 2, 1, 3]), [B, L, D]);
      const x1 = b.add(x, b.linear(att, w.outProj, w.outProjBias));
      const f = b.linear(b.relu(b.linear(b.layerNorm(x1, w.norm2.weight, w.norm2.bias, HEAD_EPS), w.linear1, w.linear1Bias)), w.linear2, w.linear2Bias);
      return b.add(x1, f);
    });
  }

  /**
   * Runs one collated batch; returns backend tensors logits f32 [B, M]
   * (masked slots = -1e4) and act f32 [B, nAct]. Caller disposes both.
   */
  forwardTensors(batch: Batch, opts: DecisionForwardOptions<T> = {}): { logits: T; act: T } {
    const b = this.backend, w = this.weights;
    const { size: B, length: L, markerCount: M } = batch;
    const D = this.hiddenSize;
    if (M < 2) throw new RangeError("DecisionModel: markerCount must be >= 2 (collate pads to two slots)");
    const kept: T[] = [];
    const emit = (name: string, t: T): void => {
      if (opts.onStage?.(name, t) === true) kept.push(t);
    };
    const f32 = (shape: number[], data: ArrayLike<number>): HostTensor => ({ dtype: "f32", shape, data: Float32Array.from(data) });
    const out = b.scope(() => {
      const enc = this.encoder.forward(batch.inputIds, batch.attentionMask, B, L, {
        onStage: opts.onStage
          ? (name, t) => {
              const own = opts.onStage!(name === "embeddings" ? name : `encoder.${name}`, t) === true;
              if (own) kept.push(t); // must also survive this scope
              return own;
            }
          : undefined,
      });
      const qtype = b.fromHost({ dtype: "i32", shape: [B], data: batch.qtype });
      let h = b.add(enc, b.reshape(b.embedding(w.typeEmb, qtype), [B, 1, D]));
      emit("type_emb_added", h);
      const headMask = b.fromHost({ dtype: "bool", shape: [B, 1, 1, L], data: batch.attentionMask });
      for (let j = 0; j < w.head.length; j++) {
        h = this.headLayer(j, h, headMask);
        emit(`head.layers.${j}`, h);
      }
      // markers = h[arange(B)[:, None], maximum(marker_pos, 0)]
      const pos = new Int32Array(B * M);
      for (let i = 0; i < pos.length; i++) pos[i] = Math.max(batch.markerPos[i]!, 0);
      const markers = b.gatherRows(h, b.fromHost({ dtype: "i32", shape: [B, M], data: pos }));
      const s = w.scorer;
      const scored = b.linear(b.gelu(b.linear(b.layerNorm(markers, s.norm.weight, s.norm.bias, HEAD_EPS), s.w1, s.b1)), s.w2, s.b2);
      const markerMask = b.fromHost({ dtype: "bool", shape: [B, M], data: batch.markerMask });
      const logits = b.where(markerMask, b.cast(b.reshape(scored, [B, M]), "f32"), b.fromHost(f32([1], [-1e4])));
      // action features [top1, top1 - top2, entropy / log(k), k / 255], k = max(#markers, 2)
      const p = b.softmax(logits, -1);
      const k = new Float32Array(B);
      for (let r = 0; r < B; r++) {
        let n = 0;
        for (let m = 0; m < M; m++) n += batch.markerMask[r * M + m] ? 1 : 0;
        k[r] = Math.max(n, 2);
      }
      const kT = b.fromHost({ dtype: "f32", shape: [B], data: k });
      const plogp = b.mul(p, b.log(b.maximum(p, b.fromHost(f32([1], [1e-9])))));
      const entropy = b.div(b.scale(b.sum(plogp, -1), -1), b.log(kT));
      const top = b.slice(b.sort(p, -1), [0, M - 2], [B, M]);
      const top2 = b.slice(top, [0, 0], [B, 1]);
      const top1 = b.slice(top, [0, 1], [B, 2]);
      const kScaled = b.div(b.reshape(kT, [B, 1]), b.fromHost(f32([1], [255])));
      const features = b.concat([top1, b.sub(top1, top2), b.reshape(entropy, [B, 1]), kScaled], -1);
      const cls = b.cast(b.reshape(b.slice(h, [0, 0, 0], [B, 1, D]), [B, D]), "f32");
      const pooled = b.cast(b.concat([cls, features], -1), this.dtype);
      const a = w.act;
      const act = b.cast(b.linear(b.gelu(b.linear(pooled, a.w1, a.b1)), a.w2, a.b2), "f32");
      return [logits, act, ...kept];
    });
    return { logits: out[0]!, act: out[1]! };
  }

  /** Runs one collated batch and reads the outputs back (f32). */
  async forward(batch: Batch, opts: DecisionForwardOptions<T> = {}): Promise<BatchOutputs> {
    const { logits, act } = this.forwardTensors(batch, opts);
    try {
      this.backend.flush?.(logits, act);
      const [l, a] = await Promise.all([this.backend.read(logits), this.backend.read(act)]);
      return { logits: toF32(l), act: toF32(a), nAct: this.nAct };
    } finally {
      this.backend.dispose(logits);
      this.backend.dispose(act);
    }
  }

  /** Frees all weights (encoder included). */
  dispose(): void {
    const b = this.backend, w = this.weights;
    const free = (...ts: (T | null)[]) => ts.forEach((t) => t && b.dispose(t));
    this.encoder.dispose();
    for (const l of w.head) {
      free(l.norm1.weight, l.norm1.bias, l.norm2.weight, l.norm2.bias, l.inProj, l.inProjBias, l.outProj, l.outProjBias);
      free(l.linear1, l.linear1Bias, l.linear2, l.linear2Bias);
    }
    free(w.typeEmb, w.scorer.norm.weight, w.scorer.norm.bias, w.scorer.w1, w.scorer.b1, w.scorer.w2, w.scorer.b2);
    free(w.act.w1, w.act.b1, w.act.w2, w.act.b2);
  }
}

function isParsed(c: ModernBertConfig | Record<string, unknown>): c is ModernBertConfig {
  return typeof (c as ModernBertConfig).hiddenSize === "number" && Array.isArray((c as ModernBertConfig).layerTypes);
}

/**
 * Loads a Laya checkpoint's DecisionModel. Weight names follow laya-mlx
 * (`encoder.*`, `head.layers.N.self_attn.in_proj.weight`, `scorer.layers.{0,1,3}`,
 * `act_head.layers.{0,2}`, `type_emb.weight`); upstream PyTorch spellings
 * (`in_proj_weight`, `scorer.0.weight`, `act_head.0.weight`) are accepted too.
 * Call outside any `backend.scope` (weights must outlive it).
 */
export function loadDecisionModel<T extends Tensor>(backend: Backend<T>, opts: LoadDecisionModelOptions): DecisionModel<T> {
  const dtype = opts.dtype ?? "f32";
  const config = isParsed(opts.encoderConfig) ? opts.encoderConfig : parseModernBertConfig(opts.encoderConfig);
  const get = toWeightGetter(opts.weights);
  const D = config.hiddenSize;
  const headLayers = opts.agentConfig.head_layers ?? 2;
  const nAct = Object.keys(opts.agentConfig.act_costs ?? {}).length + 1;
  const nh = Math.max(1, Math.floor(D / 64));
  if (D % nh) throw new Error("Decision head dimensions must be divisible by its head count");

  const encoder = loadModernBert(backend, config, get, { dtype, prefix: "encoder." });
  const loaded: T[] = [];
  const need = (names: string[], shape: number[]): T => {
    for (const n of names) {
      const h = get(n);
      if (!h) continue;
      if (h.shape.length !== shape.length || h.shape.some((d, i) => d !== shape[i])) {
        throw new Error(`DecisionModel: ${n} has shape [${h.shape}], want [${shape}]`);
      }
      const t = uploadAs(backend, h, dtype);
      loaded.push(t);
      return t;
    }
    throw new Error(`DecisionModel: missing weight ${names[0]}`);
  };
  const seq = (prefix: string, i: number, what: string) => [`${prefix}.layers.${i}.${what}`, `${prefix}.${i}.${what}`];
  try {
    const head: HeadLayerWeights<T>[] = [];
    for (let j = 0; j < headLayers; j++) {
      const p = `head.layers.${j}`;
      head.push({
        norm1: { weight: need([`${p}.norm1.weight`], [D]), bias: need([`${p}.norm1.bias`], [D]) },
        norm2: { weight: need([`${p}.norm2.weight`], [D]), bias: need([`${p}.norm2.bias`], [D]) },
        inProj: need([`${p}.self_attn.in_proj.weight`, `${p}.self_attn.in_proj_weight`], [3 * D, D]),
        inProjBias: need([`${p}.self_attn.in_proj.bias`, `${p}.self_attn.in_proj_bias`], [3 * D]),
        outProj: need([`${p}.self_attn.out_proj.weight`], [D, D]),
        outProjBias: need([`${p}.self_attn.out_proj.bias`], [D]),
        linear1: need([`${p}.linear1.weight`], [4 * D, D]),
        linear1Bias: need([`${p}.linear1.bias`], [4 * D]),
        linear2: need([`${p}.linear2.weight`], [D, 4 * D]),
        linear2Bias: need([`${p}.linear2.bias`], [D]),
      });
    }
    const typeEmb = need(["type_emb.weight"], [3, D]);
    const scorer = {
      norm: { weight: need(seq("scorer", 0, "weight"), [D]), bias: need(seq("scorer", 0, "bias"), [D]) },
      w1: need(seq("scorer", 1, "weight"), [D, D]),
      b1: need(seq("scorer", 1, "bias"), [D]),
      w2: need(seq("scorer", 3, "weight"), [1, D]),
      b2: need(seq("scorer", 3, "bias"), [1]),
    };
    const act = {
      w1: need(seq("act_head", 0, "weight"), [256, D + 4]),
      b1: need(seq("act_head", 0, "bias"), [256]),
      w2: need(seq("act_head", 2, "weight"), [nAct, 256]),
      b2: need(seq("act_head", 2, "bias"), [nAct]),
    };
    return new DecisionModel(backend, encoder, { head, typeEmb, scorer, act }, dtype, nAct);
  } catch (e) {
    encoder.dispose();
    for (const t of loaded) backend.dispose(t);
    throw e;
  }
}
