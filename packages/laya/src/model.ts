/**
 * Laya DecisionModel on any @johnhenry/tensor-backend Backend: ModernBERT
 * encoder + type embedding + PyTorch-style decision head layers + scorer +
 * action head. Exact port of laya-mlx `DecisionModel.__call__`
 * (laya_mlx/model.py). Tensor-in / numbers-out; prompts, collation and
 * result formatting live in @johnhenry/laya-core.
 */
import type { Backend, DType, Tensor } from "@johnhenry/tensor-backend";
import { embeddingAny, isQuantized, linearAny, toF32 } from "@johnhenry/tensor-backend";
import {
  attentionMasks,
  disposeWeight,
  hasQuantizedWeights,
  isHostQuantized,
  loadInBatch,
  loadModernBert,
  parseModernBertConfig,
  settleUploads,
  toWeightGetter,
  type DeviceWeight,
  type HostWeight,
  type MatrixWeight,
  type ModernBert,
  type ModernBertConfig,
  type NormWeights,
  type WeightSource,
} from "@johnhenry/modernbert";
import type { AgentConfig, Batch, BatchOutputs } from "@johnhenry/laya-core";

/** PyTorch nn.LayerNorm / MLX nn.LayerNorm default. */
const HEAD_EPS = 1e-5;

/** Linear weights (and `typeEmb`) may be quantized matrices (`MatrixWeight`). */
export interface HeadLayerWeights<T> {
  readonly norm1: NormWeights<T>;
  readonly norm2: NormWeights<T>;
  readonly inProj: MatrixWeight<T>;
  readonly inProjBias: T;
  readonly outProj: MatrixWeight<T>;
  readonly outProjBias: T;
  readonly linear1: MatrixWeight<T>;
  readonly linear1Bias: T;
  readonly linear2: MatrixWeight<T>;
  readonly linear2Bias: T;
}

export interface DecisionWeights<T> {
  readonly head: readonly HeadLayerWeights<T>[];
  readonly typeEmb: MatrixWeight<T>;
  /** scorer = LayerNorm → Linear(D, D) → GELU → Linear(D, 1) */
  readonly scorer: { readonly norm: NormWeights<T>; readonly w1: MatrixWeight<T>; readonly b1: T; readonly w2: MatrixWeight<T>; readonly b2: T };
  /** act_head = Linear(D + 4, 256) → GELU → Linear(256, nAct) */
  readonly act: { readonly w1: MatrixWeight<T>; readonly b1: T; readonly w2: MatrixWeight<T>; readonly b2: T };
  /**
   * f32 [1] constants of the forward pass (masked-logit fill −1e4, entropy
   * floor 1e-9, option-count scale 255), uploaded with the weights so
   * `forwardCore` stays synchronous (uploads are async).
   */
  readonly constants: { readonly maskedLogit: T; readonly probFloor: T; readonly kScale: T };
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

/** Device-side inputs of one batch (see `DecisionModel.uploadBatch`). */
export interface ForwardInputs<T> {
  /** i32 [B, L] */
  readonly inputIds: T;
  /** bool [B, 1, 1, L] and [B, 1, L, L] (modernbert `attentionMasks`) */
  readonly fullMask: T;
  readonly slidingMask: T;
  /** bool [B, 1, 1, L] decision-head key-padding mask */
  readonly headMask: T;
  /** i32 [B] */
  readonly qtype: T;
  /** i32 [B, M], clamped to >= 0 */
  readonly markerPos: T;
  /** bool [B, M] */
  readonly markerMask: T;
  /** f32 [B]: max(#markers, 2) per row */
  readonly k: T;
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
      const qkv = b.transpose(b.reshape(linearAny(b, h, w.inProj, w.inProjBias), [B, L, 3, nh, hd]), [2, 0, 3, 1, 4]);
      const [q, k, v] = b.split(qkv, 3, 0).map((t) => b.reshape(t, [B, nh, L, hd])) as [T, T, T];
      const att = b.reshape(b.transpose(b.sdpa(q, k, v, mask, hd ** -0.5), [0, 2, 1, 3]), [B, L, D]);
      const x1 = b.add(x, linearAny(b, att, w.outProj, w.outProjBias));
      const f = linearAny(b, b.relu(linearAny(b, b.layerNorm(x1, w.norm2.weight, w.norm2.bias, HEAD_EPS), w.linear1, w.linear1Bias)), w.linear2, w.linear2Bias);
      return b.add(x1, f);
    });
  }

  /**
   * Host → device upload of one collated batch: token ids, both encoder
   * attention masks (`attentionMasks`), the head key-padding mask, clamped
   * marker positions, marker mask, qtype and per-row option counts
   * k = max(#markers, 2). All eight uploads are started together and
   * awaited once. Caller disposes (`disposeInputs`).
   */
  async uploadBatch(batch: Batch): Promise<ForwardInputs<T>> {
    const b = this.backend;
    const { size: B, length: L, markerCount: M } = batch;
    if (M < 2) throw new RangeError("DecisionModel: markerCount must be >= 2 (collate pads to two slots)");
    const hm = attentionMasks(batch.attentionMask, B, L, this.encoder.config.localAttention);
    const pos = new Int32Array(B * M);
    for (let i = 0; i < pos.length; i++) pos[i] = Math.max(batch.markerPos[i]!, 0);
    const k = new Float32Array(B);
    for (let r = 0; r < B; r++) {
      let n = 0;
      for (let m = 0; m < M; m++) n += batch.markerMask[r * M + m] ? 1 : 0;
      k[r] = Math.max(n, 2);
    }
    const t = await settleUploads(b, new Map<keyof ForwardInputs<T>, Promise<T>>([
      ["inputIds", b.fromHost({ dtype: "i32", shape: [B, L], data: batch.inputIds })],
      ["fullMask", b.fromHost(hm.full)],
      ["slidingMask", b.fromHost(hm.sliding)],
      ["headMask", b.fromHost({ dtype: "bool", shape: [B, 1, 1, L], data: batch.attentionMask })],
      ["qtype", b.fromHost({ dtype: "i32", shape: [B], data: batch.qtype })],
      ["markerPos", b.fromHost({ dtype: "i32", shape: [B, M], data: pos })],
      ["markerMask", b.fromHost({ dtype: "bool", shape: [B, M], data: batch.markerMask })],
      ["k", b.fromHost({ dtype: "f32", shape: [B], data: k })],
    ]));
    return Object.fromEntries(t) as unknown as ForwardInputs<T>;
  }

  /** Frees `uploadBatch` results. */
  disposeInputs(inp: ForwardInputs<T>): void {
    for (const t of Object.values(inp) as T[]) this.backend.dispose(t);
  }

  /**
   * The whole decision forward pass as a pure function of device tensors
   * (what `compile` traces). Returns logits f32 [B, M] (masked slots = -1e4)
   * and act f32 [B, nAct]; intermediates not claimed through `onStage` are freed.
   */
  forwardCore(inp: ForwardInputs<T>, opts: DecisionForwardOptions<T> = {}): { logits: T; act: T } {
    const b = this.backend, w = this.weights, enc = this.encoder;
    const [B, L] = inp.inputIds.shape as [number, number];
    const M = inp.markerPos.shape[1]!;
    const D = this.hiddenSize;
    const kept: T[] = [];
    const emit = (name: string, t: T): boolean => {
      const own = opts.onStage?.(name, t) === true;
      if (own) kept.push(t);
      return own;
    };
    const cst = w.constants;
    const out = b.scope(() => {
      // ModernBERT encoder (same stage order and names as ModernBert.forward, prefixed "encoder.")
      const masks = { full_attention: inp.fullMask, sliding_attention: inp.slidingMask };
      let x = enc.embeddings(inp.inputIds);
      let owned = emit("embeddings", x);
      for (let i = 0; i < enc.config.numHiddenLayers; i++) {
        const next = enc.layer(i, x, masks[enc.config.layerTypes[i]!]);
        if (!owned) b.dispose(x);
        x = next;
        owned = emit(`encoder.layers.${i}`, x);
      }
      const encOut = enc.finalNorm(x);
      if (!owned) b.dispose(x);
      emit("encoder.final_norm", encOut);
      let h = b.add(encOut, b.reshape(embeddingAny(b, w.typeEmb, inp.qtype), [B, 1, D]));
      emit("type_emb_added", h);
      for (let j = 0; j < w.head.length; j++) {
        h = this.headLayer(j, h, inp.headMask);
        emit(`head.layers.${j}`, h);
      }
      // markers = h[arange(B)[:, None], maximum(marker_pos, 0)]
      const markers = b.gatherRows(h, inp.markerPos);
      const s = w.scorer;
      const scored = linearAny(b, b.gelu(linearAny(b, b.layerNorm(markers, s.norm.weight, s.norm.bias, HEAD_EPS), s.w1, s.b1)), s.w2, s.b2);
      const logits = b.where(inp.markerMask, b.cast(b.reshape(scored, [B, M]), "f32"), cst.maskedLogit);
      // action features [top1, top1 - top2, entropy / log(k), k / 255], k = max(#markers, 2)
      const p = b.softmax(logits, -1);
      const plogp = b.mul(p, b.log(b.maximum(p, cst.probFloor)));
      const entropy = b.div(b.scale(b.sum(plogp, -1), -1), b.log(inp.k));
      const top = b.slice(b.sort(p, -1), [0, M - 2], [B, M]);
      const top2 = b.slice(top, [0, 0], [B, 1]);
      const top1 = b.slice(top, [0, 1], [B, 2]);
      const kScaled = b.div(b.reshape(inp.k, [B, 1]), cst.kScale);
      const features = b.concat([top1, b.sub(top1, top2), b.reshape(entropy, [B, 1]), kScaled], -1);
      const cls = b.cast(b.reshape(b.slice(h, [0, 0, 0], [B, 1, D]), [B, D]), "f32");
      const pooled = b.cast(b.concat([cls, features], -1), this.dtype);
      const a = w.act;
      const act = b.cast(linearAny(b, b.gelu(linearAny(b, pooled, a.w1, a.b1)), a.w2, a.b2), "f32");
      return [logits, act, ...kept];
    });
    return { logits: out[0]!, act: out[1]! };
  }

  /**
   * Runs one collated batch; returns backend tensors logits f32 [B, M]
   * (masked slots = -1e4) and act f32 [B, nAct]. Caller disposes both.
   */
  async forwardTensors(batch: Batch, opts: DecisionForwardOptions<T> = {}): Promise<{ logits: T; act: T }> {
    const inp = await this.uploadBatch(batch);
    try {
      return this.forwardCore(inp, opts);
    } finally {
      this.disposeInputs(inp);
    }
  }

  /**
   * A `forwardTensors` equivalent through `backend.compile` (MLX: traced once
   * per input shape signature, like Python `mx.compile(model)`); returns
   * null when the backend has no `compile`. No stage observation.
   */
  compiled(): ((batch: Batch) => Promise<{ logits: T; act: T }>) | null {
    const b = this.backend;
    if (!b.compile) return null;
    const fn = b.compile((...ts: T[]): T[] => {
      const [inputIds, fullMask, slidingMask, headMask, qtype, markerPos, markerMask, k] = ts as [T, T, T, T, T, T, T, T];
      const r = this.forwardCore({ inputIds, fullMask, slidingMask, headMask, qtype, markerPos, markerMask, k });
      return [r.logits, r.act];
    });
    return async (batch) => {
      const inp = await this.uploadBatch(batch);
      try {
        const [logits, act] = fn(inp.inputIds, inp.fullMask, inp.slidingMask, inp.headMask, inp.qtype, inp.markerPos, inp.markerMask, inp.k);
        return { logits: logits!, act: act! };
      } finally {
        this.disposeInputs(inp);
      }
    };
  }

  /** Reads one `forwardTensors` result back to the host (f32) and frees it. */
  async readOutputs(r: { logits: T; act: T }): Promise<BatchOutputs> {
    const { logits, act } = r;
    try {
      this.backend.flush?.(logits, act);
      const [l, a] = await Promise.all([this.backend.read(logits), this.backend.read(act)]);
      return { logits: toF32(l), act: toF32(a), nAct: this.nAct };
    } finally {
      this.backend.dispose(logits);
      this.backend.dispose(act);
    }
  }

  /** Runs one collated batch and reads the outputs back (f32). */
  async forward(batch: Batch, opts: DecisionForwardOptions<T> = {}): Promise<BatchOutputs> {
    return this.readOutputs(await this.forwardTensors(batch, opts));
  }

  /** Whether any weight is held quantized on the device (a quantized checkpoint loaded with `quantized: "device"`). */
  get quantizedOnDevice(): boolean {
    const w = this.weights;
    const qs = [w.typeEmb, w.scorer.w1, w.scorer.w2, w.act.w1, w.act.w2, ...w.head.flatMap((l) => [l.inProj, l.outProj, l.linear1, l.linear2])];
    return hasQuantizedWeights(this.encoder) || qs.some((t) => isQuantized(t));
  }

  /** Frees all weights (encoder included). */
  dispose(): void {
    this.encoder.dispose();
    disposeWeights(this.backend, this.weights);
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
 * Every upload (encoder and heads) is started before any is awaited.
 * Call outside any `backend.scope` (weights must outlive it).
 */
export async function loadDecisionModel<T extends Tensor>(backend: Backend<T>, opts: LoadDecisionModelOptions): Promise<DecisionModel<T>> {
  const dtype = opts.dtype ?? "f32";
  const config = isParsed(opts.encoderConfig) ? opts.encoderConfig : parseModernBertConfig(opts.encoderConfig);
  const get = toWeightGetter(opts.weights);
  const D = config.hiddenSize;
  const headLayers = opts.agentConfig.head_layers ?? 2;
  const nAct = Object.keys(opts.agentConfig.act_costs ?? {}).length + 1;
  const nh = Math.max(1, Math.floor(D / 64));
  if (D % nh) throw new Error("Decision head dimensions must be divisible by its head count");

  const hosts = new Map<string, HostWeight>();
  const f32 = (v: number) => ({ dtype: "f32" as const, shape: [1], data: Float32Array.of(v) });
  const [encoder, heads, consts] = await Promise.allSettled([
    loadModernBert(backend, config, get, { dtype, prefix: "encoder." }),
    loadInBatch(backend, (upload) => {
      const need = (names: string[], shape: number[], matrix = false): DeviceWeight<T> => {
        for (const n of names) {
          const h = hosts.get(n) ?? get(n);
          if (!h) continue;
          hosts.set(n, h);
          if (h.shape.length !== shape.length || h.shape.some((d, i) => d !== shape[i])) {
            throw new Error(`DecisionModel: ${n} has shape [${h.shape}], want [${shape}]`);
          }
          if (!matrix && isHostQuantized(h)) throw new Error(`DecisionModel: ${n} cannot be quantized (only Linear weights and type_emb)`);
          return upload(n, h);
        }
        throw new Error(`DecisionModel: missing weight ${names[0]}`);
      };
      return buildHeadWeights<T>(D, headLayers, nAct, need);
    }, dtype),
    // f32 whatever the model dtype
    settleUploads(backend, new Map([
      ["maskedLogit", backend.fromHost(f32(-1e4))],
      ["probFloor", backend.fromHost(f32(1e-9))],
      ["kScale", backend.fromHost(f32(255))],
    ])),
  ]);
  hosts.clear();
  if (encoder.status === "rejected" || heads.status === "rejected" || consts.status === "rejected") {
    if (encoder.status === "fulfilled") encoder.value.dispose();
    if (heads.status === "fulfilled") disposeWeights(backend, heads.value);
    if (consts.status === "fulfilled") consts.value.forEach((t) => backend.dispose(t));
    throw ([encoder, heads, consts].find((r) => r.status === "rejected") as PromiseRejectedResult).reason;
  }
  const c = consts.value;
  const constants = { maskedLogit: c.get("maskedLogit")!, probFloor: c.get("probFloor")!, kScale: c.get("kScale")! };
  return new DecisionModel(backend, encoder.value, { ...heads.value, constants }, dtype, nAct);
}

function buildHeadWeights<T extends Tensor>(
  D: number,
  headLayers: number,
  nAct: number,
  needAny: (names: string[], shape: number[], matrix?: boolean) => DeviceWeight<T>,
): Omit<DecisionWeights<T>, "constants"> {
  const need = (names: string[], shape: number[]) => needAny(names, shape) as T;
  const mat = (names: string[], shape: number[]) => needAny(names, shape, true);
  const seq = (prefix: string, i: number, what: string) => [`${prefix}.layers.${i}.${what}`, `${prefix}.${i}.${what}`];
  const head: HeadLayerWeights<T>[] = [];
  for (let j = 0; j < headLayers; j++) {
    const p = `head.layers.${j}`;
    head.push({
      norm1: { weight: need([`${p}.norm1.weight`], [D]), bias: need([`${p}.norm1.bias`], [D]) },
      norm2: { weight: need([`${p}.norm2.weight`], [D]), bias: need([`${p}.norm2.bias`], [D]) },
      inProj: mat([`${p}.self_attn.in_proj.weight`, `${p}.self_attn.in_proj_weight`], [3 * D, D]),
      inProjBias: need([`${p}.self_attn.in_proj.bias`, `${p}.self_attn.in_proj_bias`], [3 * D]),
      outProj: mat([`${p}.self_attn.out_proj.weight`], [D, D]),
      outProjBias: need([`${p}.self_attn.out_proj.bias`], [D]),
      linear1: mat([`${p}.linear1.weight`], [4 * D, D]),
      linear1Bias: need([`${p}.linear1.bias`], [4 * D]),
      linear2: mat([`${p}.linear2.weight`], [D, 4 * D]),
      linear2Bias: need([`${p}.linear2.bias`], [D]),
    });
  }
  const typeEmb = mat(["type_emb.weight"], [3, D]);
  const scorer = {
    norm: { weight: need(seq("scorer", 0, "weight"), [D]), bias: need(seq("scorer", 0, "bias"), [D]) },
    w1: mat(seq("scorer", 1, "weight"), [D, D]),
    b1: need(seq("scorer", 1, "bias"), [D]),
    w2: mat(seq("scorer", 3, "weight"), [1, D]),
    b2: need(seq("scorer", 3, "bias"), [1]),
  };
  const act = {
    w1: mat(seq("act_head", 0, "weight"), [256, D + 4]),
    b1: need(seq("act_head", 0, "bias"), [256]),
    w2: mat(seq("act_head", 2, "weight"), [nAct, 256]),
    b2: need(seq("act_head", 2, "bias"), [nAct]),
  };
  return { head, typeEmb, scorer, act };
}

/** Frees decision-head weights (and the constants, when present). */
function disposeWeights<T extends Tensor>(b: Backend<T>, w: Omit<DecisionWeights<T>, "constants"> & Partial<Pick<DecisionWeights<T>, "constants">>): void {
  const free = (...ts: (MatrixWeight<T> | null)[]) => ts.forEach((t) => disposeWeight(b, t));
  for (const l of w.head) {
    free(l.norm1.weight, l.norm1.bias, l.norm2.weight, l.norm2.bias, l.inProj, l.inProjBias, l.outProj, l.outProjBias);
    free(l.linear1, l.linear1Bias, l.linear2, l.linear2Bias);
  }
  free(w.typeEmb, w.scorer.norm.weight, w.scorer.norm.bias, w.scorer.w1, w.scorer.b1, w.scorer.w2, w.scorer.b2);
  free(w.act.w1, w.act.b1, w.act.w2, w.act.b2);
  if (w.constants) free(w.constants.maskedLogit, w.constants.probFloor, w.constants.kScale);
}
