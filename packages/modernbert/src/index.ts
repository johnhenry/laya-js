/**
 * @johnhenry/modernbert — ModernBERT / mmBERT encoder on any
 * @johnhenry/tensor-backend Backend (CPU reference, MLX, WebGPU).
 *
 * Port of laya-mlx `laya_mlx/model.py` (EncoderConfig, Embeddings,
 * EncoderAttention, EncoderMLP, EncoderLayer, attention_masks, ModernBert),
 * which follows Hugging Face ModernBERT. Uses only Backend ops, so every
 * backend gets the encoder for free; masks are built on the host.
 */
import type { Backend, DType, HostTensor, Tensor } from "@johnhenry/tensor-backend";
import { geglu, meanPool, toF32 } from "@johnhenry/tensor-backend";

// ------------------------------------------------------------------ config
export type AttentionKind = "full_attention" | "sliding_attention";

export interface ModernBertConfig {
  readonly modelType: "modernbert";
  readonly vocabSize: number;
  readonly hiddenSize: number;
  readonly intermediateSize: number;
  readonly numHiddenLayers: number;
  readonly numAttentionHeads: number;
  readonly headDim: number;
  readonly normEps: number;
  readonly normBias: boolean;
  readonly attentionBias: boolean;
  readonly mlpBias: boolean;
  readonly hiddenActivation: "gelu";
  /** Sliding window width; keys with |i - j| <= localAttention // 2 are visible. */
  readonly localAttention: number;
  readonly globalAttnEveryNLayers: number;
  readonly maxPositionEmbeddings: number;
  readonly layerTypes: readonly AttentionKind[];
  /** RoPE theta per attention kind (rope_parameters[kind].rope_theta, else global/local_rope_theta). */
  readonly ropeBase: Readonly<Record<AttentionKind, number>>;
}

type Json = Record<string, unknown>;

function num(v: unknown, fallback: number, name: string): number {
  if (v === undefined || v === null) return fallback;
  if (typeof v !== "number" || !Number.isFinite(v)) throw new TypeError(`modernbert config: ${name} must be a number`);
  return v;
}
function int(v: unknown, name: string, fallback?: number): number {
  if ((v === undefined || v === null) && fallback !== undefined) return fallback;
  if (typeof v !== "number" || !Number.isInteger(v)) throw new TypeError(`modernbert config: ${name} must be an integer`);
  return v;
}
function bool(v: unknown, fallback: boolean): boolean {
  return v === undefined || v === null ? fallback : Boolean(v);
}

/**
 * Parses a Hugging Face ModernBERT `config.json` (port of EncoderConfig.from_dict,
 * including its validation). Unknown keys are ignored.
 */
export function parseModernBertConfig(json: Json): ModernBertConfig {
  const modelType = json.model_type ?? "modernbert";
  if (modelType !== "modernbert") throw new Error(`Unsupported encoder: ${JSON.stringify(modelType)}; expected modernbert`);
  const act = json.hidden_activation ?? "gelu";
  if (act !== "gelu") throw new Error(`Unsupported encoder activation: ${JSON.stringify(act)}`);
  const hiddenSize = int(json.hidden_size, "hidden_size");
  const numAttentionHeads = int(json.num_attention_heads, "num_attention_heads");
  const numHiddenLayers = int(json.num_hidden_layers, "num_hidden_layers");
  if (numAttentionHeads <= 0 || hiddenSize % numAttentionHeads || (hiddenSize / numAttentionHeads) % 2) {
    throw new Error("ModernBERT requires an even, integral attention head dimension");
  }
  const every = int(json.global_attn_every_n_layers, "global_attn_every_n_layers", 3);
  let layerTypes = json.layer_types as AttentionKind[] | null | undefined;
  if (layerTypes === undefined || layerTypes === null) {
    layerTypes = Array.from({ length: numHiddenLayers }, (_, i) => (i % every === 0 ? "full_attention" : "sliding_attention"));
  }
  if (
    !Array.isArray(layerTypes) ||
    layerTypes.length !== numHiddenLayers ||
    layerTypes.some((k) => k !== "full_attention" && k !== "sliding_attention")
  ) {
    throw new Error("Invalid ModernBERT layer_types");
  }
  const ropeParams = (json.rope_parameters ?? {}) as Record<string, { rope_type?: string; rope_theta?: number } | undefined>;
  for (const kind of new Set(layerTypes)) {
    if ((ropeParams[kind]?.rope_type ?? "default") !== "default") throw new Error("Only default (unscaled) ModernBERT RoPE is supported");
  }
  const globalTheta = num(json.global_rope_theta, 160000, "global_rope_theta");
  const localTheta = num(json.local_rope_theta, 10000, "local_rope_theta");
  const ropeBase = {
    full_attention: Number(ropeParams.full_attention?.rope_theta ?? globalTheta),
    sliding_attention: Number(ropeParams.sliding_attention?.rope_theta ?? localTheta),
  };
  return Object.freeze({
    modelType: "modernbert",
    vocabSize: int(json.vocab_size, "vocab_size"),
    hiddenSize,
    intermediateSize: int(json.intermediate_size, "intermediate_size"),
    numHiddenLayers,
    numAttentionHeads,
    headDim: hiddenSize / numAttentionHeads,
    normEps: num(json.norm_eps, 1e-5, "norm_eps"),
    normBias: bool(json.norm_bias, false),
    attentionBias: bool(json.attention_bias, false),
    mlpBias: bool(json.mlp_bias, false),
    hiddenActivation: "gelu",
    localAttention: int(json.local_attention, "local_attention", 128),
    globalAttnEveryNLayers: every,
    maxPositionEmbeddings: int(json.max_position_embeddings, "max_position_embeddings", 8192),
    layerTypes: Object.freeze([...layerTypes]),
    ropeBase: Object.freeze(ropeBase),
  } as const);
}

// ------------------------------------------------------------------ weights
/** Looks up a checkpoint tensor by name; `undefined` when absent. */
export type WeightGetter = (name: string) => HostTensor | undefined;
export type WeightSource = WeightGetter | { get(name: string): HostTensor | undefined };

/** Structural subset of @johnhenry/math-plus-safetensors' SafetensorsFile. */
export interface SafetensorsLike {
  has(name: string): boolean;
  info(name: string): { dtype: string; shape: readonly number[] };
  view(name: string): ArrayLike<number> | ArrayBufferView;
}

/**
 * Adapts an in-memory safetensors file (e.g. `readSafetensors(bytes)` from
 * @johnhenry/math-plus-safetensors) to a WeightGetter. F16 → f16
 * (Float16Array), BF16 → bf16 (raw Uint16Array bits), F32 → f32, I32 → i32.
 * Views are zero-copy; backends copy/convert on `fromHost`.
 */
export function safetensorsWeights(file: SafetensorsLike): WeightGetter {
  const map: Record<string, DType> = { F16: "f16", BF16: "bf16", F32: "f32", I32: "i32", BOOL: "bool", U8: "bool" };
  return (name) => {
    if (!file.has(name)) return undefined;
    const info = file.info(name);
    const dtype = map[info.dtype];
    if (!dtype) throw new TypeError(`modernbert: unsupported safetensors dtype ${info.dtype} for ${name}`);
    return { dtype, shape: [...info.shape], data: file.view(name) as HostTensor["data"] };
  };
}

export function toWeightGetter(src: WeightSource): WeightGetter {
  return typeof src === "function" ? src : (name) => src.get(name);
}

const FLOATS = new Set<DType>(["f32", "f16", "bf16"]);

/**
 * Uploads a host tensor as `dtype` on `backend`. Float data is converted
 * as needed (via host f32 when the backend cannot store the source dtype).
 */
export function uploadAs<T extends Tensor>(backend: Backend<T>, h: HostTensor, dtype: DType): T {
  if (!FLOATS.has(h.dtype)) return backend.fromHost(h);
  if (!backend.supports(dtype)) throw new Error(`${backend.name} backend does not support ${dtype}`);
  const src = h.dtype === dtype || backend.supports(h.dtype) ? h : { dtype: "f32" as const, shape: h.shape, data: toF32(h) };
  const t = backend.fromHost(src);
  if (t.dtype === dtype) return t;
  const c = backend.cast(t, dtype);
  backend.dispose(t);
  return c;
}

// ------------------------------------------------------------------ masks
/**
 * Host-built boolean attention masks, exactly like laya-mlx `attention_masks`:
 * - full    [B, 1, 1, L]: key j visible iff valid[b, j]
 * - sliding [B, 1, L, L]: (|i - j| <= window // 2  OR  query i is padding) AND valid[b, j]
 * Padded query rows see every valid key (avoids all-masked softmax rows);
 * they are never used as keys or pooled, so valid outputs are unaffected.
 */
export function attentionMasks(attentionMask: Uint8Array, B: number, L: number, window: number): { full: HostTensor; sliding: HostTensor } {
  if (attentionMask.length !== B * L) throw new RangeError(`attentionMasks: mask has ${attentionMask.length} values, want ${B * L}`);
  const half = Math.floor(window / 2);
  const full = new Uint8Array(B * L);
  const sliding = new Uint8Array(B * L * L);
  for (let b = 0; b < B; b++) {
    for (let j = 0; j < L; j++) full[b * L + j] = attentionMask[b * L + j] ? 1 : 0;
    for (let i = 0; i < L; i++) {
      const padQuery = !attentionMask[b * L + i];
      const row = (b * L + i) * L;
      for (let j = 0; j < L; j++) {
        sliding[row + j] = full[b * L + j] && (padQuery || Math.abs(i - j) <= half) ? 1 : 0;
      }
    }
  }
  return {
    full: { dtype: "bool", shape: [B, 1, 1, L], data: full },
    sliding: { dtype: "bool", shape: [B, 1, L, L], data: sliding },
  };
}

// ------------------------------------------------------------------ model
export interface NormWeights<T> {
  readonly weight: T;
  readonly bias: T | null;
}

export interface EncoderLayerWeights<T> {
  readonly kind: AttentionKind;
  /** null for layer 0 (Identity). */
  readonly attnNorm: NormWeights<T> | null;
  readonly Wqkv: T;
  readonly WqkvBias: T | null;
  readonly Wo: T;
  readonly WoBias: T | null;
  readonly mlpNorm: NormWeights<T>;
  readonly Wi: T;
  readonly WiBias: T | null;
  readonly WoMlp: T;
  readonly WoMlpBias: T | null;
}

export interface ModernBertWeights<T> {
  readonly tokEmbeddings: T;
  readonly embNorm: NormWeights<T>;
  readonly layers: readonly EncoderLayerWeights<T>[];
  readonly finalNorm: NormWeights<T>;
}

export interface LoadModernBertOptions {
  /** Compute/storage dtype for weights and activations (default "f32"). */
  readonly dtype?: "f32" | "f16" | "bf16";
  /**
   * Parameter-name prefix: "encoder." (laya-mlx / Laya checkpoints),
   * "model." (Hugging Face ModernBertModel / MaskedLM) or "". Default:
   * auto-detected from which `<prefix>embeddings.tok_embeddings.weight` exists.
   */
  readonly prefix?: string;
}

export interface ForwardOptions<T> {
  /**
   * Called with each stage output: "embeddings", "layers.<i>", "final_norm".
   * Return true to take ownership (you must dispose it); otherwise the
   * tensor is disposed once the next stage has consumed it.
   */
  readonly onStage?: (name: string, t: T) => boolean | void;
}

export class ModernBert<T extends Tensor = Tensor> {
  readonly backend: Backend<T>;
  readonly config: ModernBertConfig;
  readonly dtype: DType;
  readonly weights: ModernBertWeights<T>;

  constructor(backend: Backend<T>, config: ModernBertConfig, weights: ModernBertWeights<T>, dtype: DType) {
    this.backend = backend;
    this.config = config;
    this.weights = weights;
    this.dtype = dtype;
  }

  /** Token embeddings + LayerNorm: ids i32 [B, L] → [B, L, H]. */
  embeddings(ids: T): T {
    const b = this.backend, w = this.weights;
    return b.scope(() => b.layerNorm(b.embedding(w.tokEmbeddings, ids), w.embNorm.weight, w.embNorm.bias, this.config.normEps));
  }

  /** One pre-norm encoder layer; `mask` is the upload of the matching kind from `attentionMasks`. */
  layer(i: number, x: T, mask: T): T {
    const b = this.backend, c = this.config, w = this.weights.layers[i]!;
    return b.scope(() => {
      const [B, L, H] = x.shape as [number, number, number];
      const nh = c.numAttentionHeads, hd = c.headDim;
      const h = w.attnNorm ? b.layerNorm(x, w.attnNorm.weight, w.attnNorm.bias, c.normEps) : x;
      const qkv = b.transpose(b.reshape(b.linear(h, w.Wqkv, w.WqkvBias), [B, L, 3, nh, hd]), [2, 0, 3, 1, 4]);
      const [q, k, v] = b.split(qkv, 3, 0).map((t) => b.reshape(t, [B, nh, L, hd])) as [T, T, T];
      const base = c.ropeBase[w.kind];
      const att = b.sdpa(b.rope(q, base), b.rope(k, base), v, mask, hd ** -0.5);
      const merged = b.reshape(b.transpose(att, [0, 2, 1, 3]), [B, L, H]);
      const x1 = b.add(x, b.linear(merged, w.Wo, w.WoBias));
      const m = b.linear(b.layerNorm(x1, w.mlpNorm.weight, w.mlpNorm.bias, c.normEps), w.Wi, w.WiBias);
      return b.add(x1, b.linear(geglu(b, m), w.WoMlp, w.WoMlpBias));
    });
  }

  finalNorm(x: T): T {
    const b = this.backend, w = this.weights;
    return b.scope(() => b.layerNorm(x, w.finalNorm.weight, w.finalNorm.bias, this.config.normEps));
  }

  /**
   * Full encoder: inputIds [B*L] (row-major), attentionMask [B*L] (0/1)
   * → hidden states [B, L, H] in `this.dtype`. Masks are built on the host.
   * Intermediates are disposed; the caller owns the result.
   */
  forward(inputIds: Int32Array, attentionMask: Uint8Array, B: number, L: number, opts: ForwardOptions<T> = {}): T {
    const b = this.backend, c = this.config;
    if (inputIds.length !== B * L) throw new RangeError(`modernbert: inputIds has ${inputIds.length} values, want ${B * L}`);
    const kept: T[] = [];
    const emit = (name: string, t: T, last: boolean): void => {
      if (opts.onStage?.(name, t) === true && !last) kept.push(t);
    };
    const out = b.scope(() => {
      const ids = b.fromHost({ dtype: "i32", shape: [B, L], data: inputIds });
      const hm = attentionMasks(attentionMask, B, L, c.localAttention);
      const masks: Record<AttentionKind, T | null> = {
        full_attention: c.layerTypes.includes("full_attention") ? b.fromHost(hm.full) : null,
        sliding_attention: c.layerTypes.includes("sliding_attention") ? b.fromHost(hm.sliding) : null,
      };
      let x = this.embeddings(ids);
      emit("embeddings", x, false);
      for (let i = 0; i < c.numHiddenLayers; i++) {
        const next = this.layer(i, x, masks[c.layerTypes[i]!]!);
        if (!kept.includes(x)) b.dispose(x);
        x = next;
        emit(`layers.${i}`, x, false);
      }
      const y = this.finalNorm(x);
      if (!kept.includes(x)) b.dispose(x);
      emit("final_norm", y, true);
      return [y, ...kept];
    });
    return out[0]!;
  }

  /**
   * Mean-pooled sentence vectors over valid tokens (laya-mlx
   * `embed_fn_from_agent`): [B, H] f32. Pass ids tokenized WITH special tokens.
   */
  embed(inputIds: Int32Array, attentionMask: Uint8Array, B: number, L: number): T {
    const b = this.backend;
    return b.scope(() => {
      const h = this.forward(inputIds, attentionMask, B, L);
      const m = b.fromHost({ dtype: "bool", shape: [B, L], data: attentionMask });
      return meanPool(b, h, m);
    });
  }

  /** `embed` + read back: Float32Array [B * H]. */
  async embedToHost(inputIds: Int32Array, attentionMask: Uint8Array, B: number, L: number): Promise<Float32Array> {
    const t = this.embed(inputIds, attentionMask, B, L);
    try {
      return toF32(await this.backend.read(t));
    } finally {
      this.backend.dispose(t);
    }
  }

  /** Frees all weight tensors. */
  dispose(): void {
    const b = this.backend, w = this.weights;
    const free = (t: T | null | undefined) => {
      if (t) b.dispose(t);
    };
    const norm = (n: NormWeights<T> | null) => {
      if (n) {
        free(n.weight);
        free(n.bias);
      }
    };
    free(w.tokEmbeddings);
    norm(w.embNorm);
    norm(w.finalNorm);
    for (const l of w.layers) {
      norm(l.attnNorm);
      norm(l.mlpNorm);
      for (const t of [l.Wqkv, l.WqkvBias, l.Wo, l.WoBias, l.Wi, l.WiBias, l.WoMlp, l.WoMlpBias]) free(t);
    }
  }
}

/** Detects the encoder parameter prefix ("encoder.", "model." or ""). */
export function detectPrefix(get: WeightGetter): string {
  for (const p of ["encoder.", "model.", ""]) if (get(`${p}embeddings.tok_embeddings.weight`)) return p;
  throw new Error("modernbert: no embeddings.tok_embeddings.weight under prefixes encoder., model. or ''");
}

/**
 * Loads encoder weights from `weights` (laya-mlx names `encoder.layers.N.attn.Wqkv.weight`
 * or HF names `model.layers.N.attn.Wqkv.weight`, `model.embeddings...`, `model.final_norm...`).
 * Layer 0 has no attn_norm. Biases are loaded only when the config enables them.
 */
export function loadModernBert<T extends Tensor>(
  backend: Backend<T>,
  config: ModernBertConfig,
  weights: WeightSource,
  opts: LoadModernBertOptions = {},
): ModernBert<T> {
  const dtype = opts.dtype ?? "f32";
  if (!backend.supports(dtype)) throw new Error(`${backend.name} backend does not support ${dtype}`);
  const get = toWeightGetter(weights);
  const prefix = opts.prefix ?? detectPrefix(get);
  const loaded: T[] = [];
  const need = (name: string, shape?: readonly number[]): T => {
    const h = get(prefix + name);
    if (!h) throw new Error(`modernbert: missing weight ${prefix + name}`);
    if (shape && (h.shape.length !== shape.length || h.shape.some((d, i) => d !== shape[i]))) {
      throw new Error(`modernbert: ${prefix + name} has shape [${h.shape}], want [${shape}]`);
    }
    const t = uploadAs(backend, h, dtype);
    loaded.push(t);
    return t;
  };
  const maybe = (on: boolean, name: string, shape: readonly number[]): T | null => (on ? need(name, shape) : null);
  const H = config.hiddenSize, I = config.intermediateSize;
  const normW = (p: string): NormWeights<T> => ({ weight: need(`${p}.weight`, [H]), bias: maybe(config.normBias, `${p}.bias`, [H]) });
  try {
    const tokEmbeddings = need("embeddings.tok_embeddings.weight", [config.vocabSize, H]);
    const embNorm = normW("embeddings.norm");
    const layers: EncoderLayerWeights<T>[] = [];
    for (let i = 0; i < config.numHiddenLayers; i++) {
      const p = `layers.${i}`;
      layers.push({
        kind: config.layerTypes[i]!,
        attnNorm: i === 0 ? null : normW(`${p}.attn_norm`),
        Wqkv: need(`${p}.attn.Wqkv.weight`, [3 * H, H]),
        WqkvBias: maybe(config.attentionBias, `${p}.attn.Wqkv.bias`, [3 * H]),
        Wo: need(`${p}.attn.Wo.weight`, [H, H]),
        WoBias: maybe(config.attentionBias, `${p}.attn.Wo.bias`, [H]),
        mlpNorm: normW(`${p}.mlp_norm`),
        Wi: need(`${p}.mlp.Wi.weight`, [2 * I, H]),
        WiBias: maybe(config.mlpBias, `${p}.mlp.Wi.bias`, [2 * I]),
        WoMlp: need(`${p}.mlp.Wo.weight`, [H, I]),
        WoMlpBias: maybe(config.mlpBias, `${p}.mlp.Wo.bias`, [H]),
      });
    }
    const finalNorm = normW("final_norm");
    return new ModernBert(backend, config, { tokEmbeddings, embNorm, layers, finalNorm }, dtype);
  } catch (e) {
    for (const t of loaded) backend.dispose(t);
    throw e;
  }
}
