/**
 * LayaAgent: port of laya-mlx `Agent` (agent.py) on any tensor backend.
 * No I/O here: `createAgent` takes parsed configs, a tokenizer and a weight
 * source; `load()` (load.ts) resolves and reads a checkpoint first.
 */
import type { Backend, Tensor } from "@johnhenry/tensor-backend";
import { parseModernBertConfig, toWeightGetter, type ModernBertConfig, type WeightGetter, type WeightSource } from "@johnhenry/modernbert";
import {
  PrefixCache,
  chunkItems,
  collate,
  formatResults,
  prepare as corePrepare,
  resolveTemperatures,
  type AgentConfig,
  type Batch,
  type BatchOutputs,
  type InternalQuestion,
  type LayaTokenizer,
  type PredictResult,
  type PreparedItem,
  type Questions,
  type State,
  type Temperatures,
} from "@johnhenry/laya-core";
import { loadDecisionModel, type DecisionModel } from "./model.ts";

export type Dtype = "f16" | "f32";

/** Tokenizers that can also add the post-processor's special tokens (needed by `embed`). */
export interface EmbeddingTokenizer extends LayaTokenizer {
  encodeWithSpecialTokens(text: string): number[];
}

export interface AgentOptions {
  /** Rows per forward pass (default 16, like `batch_size`). */
  batchSize?: number;
  /** Round the padded length up to a multiple (capped at max_len); default null. */
  padToMultiple?: number | null;
  /** Reuse tokenized question prefixes (`cache_prompts=True`); default false. */
  cachePrompts?: boolean;
  /** Use `backend.compile` when the backend has it (`compile=True`); default false. */
  compile?: boolean;
}

export interface AgentParts<T extends Tensor = Tensor> extends AgentOptions {
  backend: Backend<T>;
  /** encoder/config.json (raw JSON) or a parsed ModernBertConfig. */
  encoderConfig: Record<string, unknown> | ModernBertConfig;
  /** rl_agent_config.json */
  agentConfig: AgentConfig;
  /** Checkpoint tensors by name (see `safetensorsWeights` / `readWeights`). */
  weights: WeightSource;
  tokenizer: LayaTokenizer;
  /** Requested dtype (default "f16"); f32 when the backend cannot compute in f16 (e.g. cpu). */
  dtype?: Dtype;
  /** Reported as `agent.modelId` (default "<local>"). */
  modelId?: string;
  /** Destroy `backend` in `dispose()` (default false: the caller owns it). */
  ownsBackend?: boolean;
  /** Where the temperature-clamping warning goes (default console.warn). */
  warn?: (message: string) => void;
}

export interface LayaAgent {
  readonly modelId: string;
  readonly backend: Backend;
  readonly dtype: Dtype;
  readonly config: AgentConfig;
  readonly encoderConfig: ModernBertConfig;
  readonly tokenizer: LayaTokenizer;
  readonly temperature: number[];
  readonly temperatureByOptions: Record<string, number>;
  readonly temperatureRaw: number[];
  readonly temperatureByOptionsRaw: Record<string, number>;
  readonly batchSize: number;
  readonly padToMultiple: number | null;
  readonly model: DecisionModel;
  /** Answer every question in `questions` about `state` (`Agent.system_one`). */
  predict(state: State, questions: Questions): Promise<PredictResult>;
  /** Alias of `predict`. */
  systemOne(state: State, questions: Questions): Promise<PredictResult>;
  /** Upstream-compatible model inputs (`Agent.prepare`); `internal[i].id` is the question id. */
  prepare(state: State, questions: Questions): { items: PreparedItem[]; internal: InternalQuestion[] };
  /** Run one collated batch (`Agent.forward`), outputs read back as f32. */
  forward(batch: Batch): Promise<BatchOutputs>;
  /** Mean-pooled encoder vectors (`embed_fn_from_agent`), one Float32Array per text. */
  embed(texts: string[], opts?: { maxLength?: number; batchSize?: number }): Promise<Float32Array[]>;
  /** Frees the weights (and the backend when the agent created it). Idempotent. */
  dispose(): void;
}

const isPosInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v) && v >= 1;

/** `Agent.__init__` config validation (agent.py). Throws Python's messages. */
export function validateConfig(cfg: AgentConfig, enc: ModernBertConfig): void {
  if (!cfg || typeof cfg !== "object" || !("encoder" in cfg) || !("head_layers" in cfg)) {
    throw new Error("Laya config must specify encoder and head_layers");
  }
  const maxLen = cfg.max_len ?? 512;
  const headMaxLen = cfg.head_max_len ?? 192;
  if (!(4 < headMaxLen && headMaxLen < maxLen && maxLen <= enc.maxPositionEmbeddings)) {
    throw new Error("Expected 4 < head_max_len < max_len <= max_position_embeddings");
  }
}

class Agent<T extends Tensor> implements LayaAgent {
  readonly modelId: string;
  readonly backend: Backend<T>;
  readonly dtype: Dtype;
  readonly config: AgentConfig;
  readonly encoderConfig: ModernBertConfig;
  readonly tokenizer: LayaTokenizer;
  readonly temperature: number[];
  readonly temperatureByOptions: Record<string, number>;
  readonly temperatureRaw: number[];
  readonly temperatureByOptionsRaw: Record<string, number>;
  readonly batchSize: number;
  readonly padToMultiple: number | null;
  readonly model: DecisionModel<T>;
  private readonly temps: Temperatures;
  private readonly cache: PrefixCache | null;
  private readonly run: (batch: Batch) => Promise<{ logits: T; act: T }>;
  private readonly ownsBackend: boolean;
  private disposed = false;

  /** Use `createAgent` (weights upload asynchronously). */
  constructor(parts: AgentParts<T>, v: Validated, model: DecisionModel<T>) {
    const { enc, temps, batchSize, pad } = v;
    this.backend = parts.backend;
    this.dtype = v.dtype;
    this.modelId = parts.modelId ?? "<local>";
    this.config = parts.agentConfig;
    this.encoderConfig = enc;
    this.tokenizer = parts.tokenizer;
    this.batchSize = batchSize;
    this.padToMultiple = pad;
    this.temps = temps;
    this.temperature = temps.temperature;
    this.temperatureByOptions = temps.temperatureByOptions;
    this.temperatureRaw = temps.temperatureRaw as number[];
    this.temperatureByOptionsRaw = temps.temperatureByOptionsRaw as Record<string, number>;
    this.cache = parts.cachePrompts ? new PrefixCache() : null;
    this.ownsBackend = parts.ownsBackend ?? false;
    this.model = model;
    const compiled = parts.compile ? this.model.compiled() : null;
    this.run = compiled ?? ((batch) => this.model.forwardTensors(batch));
  }

  private live(): void {
    if (this.disposed) throw new Error("LayaAgent has been disposed");
  }

  prepare(state: State, questions: Questions): { items: PreparedItem[]; internal: InternalQuestion[] } {
    return corePrepare(this.tokenizer, state, questions, this.config, { cache: this.cache });
  }

  async forward(batch: Batch): Promise<BatchOutputs> {
    this.live();
    return this.model.readOutputs(await this.run(batch));
  }

  async predict(state: State, questions: Questions): Promise<PredictResult> {
    this.live();
    const { items, internal } = this.prepare(state, questions);
    const maxLength = this.config.max_len ?? 512;
    const outputs: BatchOutputs[] = [];
    for (const chunk of chunkItems(items, this.batchSize)) {
      const batch = collate(chunk, this.tokenizer.padTokenId, { padToMultiple: this.padToMultiple, maxLength });
      outputs.push(await this.forward(batch));
    }
    return formatResults(internal, items, outputs, this.temps);
  }

  systemOne(state: State, questions: Questions): Promise<PredictResult> {
    return this.predict(state, questions);
  }

  async embed(texts: string[], opts: { maxLength?: number; batchSize?: number } = {}): Promise<Float32Array[]> {
    this.live();
    const maxLength = opts.maxLength ?? 512;
    const batchSize = opts.batchSize ?? 32;
    if (!isPosInt(maxLength)) throw new Error(`max_length must be a positive integer, got ${String(maxLength)}`);
    if (!isPosInt(batchSize)) throw new Error(`batch_size must be a positive integer, got ${String(batchSize)}`);
    const tok = this.tokenizer as Partial<EmbeddingTokenizer>;
    if (typeof tok.encodeWithSpecialTokens !== "function") {
      throw new Error("embed needs a tokenizer with encodeWithSpecialTokens (e.g. laya-core loadTokenizer)");
    }
    const H = this.encoderConfig.hiddenSize;
    const rows = texts.map((t) => (t === null || t === undefined ? "" : String(t)));
    const out: Float32Array[] = [];
    for (let start = 0; start < rows.length; start += batchSize) {
      const encoded = rows.slice(start, start + batchSize).map((t) => tok.encodeWithSpecialTokens!(t).slice(0, maxLength));
      // A text with no tokens pools to the zero vector (Python: sum(h * 0) / max(0, 1)); it is
      // not run, because an all-masked attention row is undefined on the tensor backends.
      const live = encoded.map((ids, i) => [ids, i] as const).filter(([ids]) => ids.length > 0);
      const vecs: Float32Array[] = encoded.map(() => new Float32Array(H));
      if (live.length) {
        const B = live.length;
        const L = Math.max(...live.map(([ids]) => ids.length));
        const ids = new Int32Array(B * L).fill(this.tokenizer.padTokenId);
        const mask = new Uint8Array(B * L);
        live.forEach(([row], r) => {
          ids.set(row, r * L);
          mask.fill(1, r * L, r * L + row.length);
        });
        const pooled = await this.model.encoder.embedToHost(ids, mask, B, L);
        live.forEach(([, i], r) => (vecs[i] = pooled.slice(r * H, (r + 1) * H)));
      }
      out.push(...vecs);
    }
    return out;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.model.dispose();
    if (this.ownsBackend) this.backend.destroy?.();
  }
}

function isParsed(c: Record<string, unknown> | ModernBertConfig): c is ModernBertConfig {
  return typeof (c as ModernBertConfig).hiddenSize === "number" && Array.isArray((c as ModernBertConfig).layerTypes);
}

interface Validated {
  enc: ModernBertConfig;
  temps: Temperatures;
  batchSize: number;
  pad: number | null;
  dtype: Dtype;
}

/** `Agent.__init__` option and config validation; throws Python's messages. */
function validateParts<T extends Tensor>(parts: AgentParts<T>): Validated {
  const batchSize = parts.batchSize ?? 16;
  if (!isPosInt(batchSize)) throw new Error("batch_size must be a positive integer");
  const pad = parts.padToMultiple ?? null;
  if (pad !== null && !isPosInt(pad)) throw new Error("pad_to_multiple must be a positive integer or None");
  const want = parts.dtype ?? "f16";
  if (want !== "f16" && want !== "f32") throw new Error(`dtype must be one of ["f16", "f32"]`);
  const enc = isParsed(parts.encoderConfig) ? parts.encoderConfig : parseModernBertConfig(parts.encoderConfig);
  validateConfig(parts.agentConfig, enc);
  const temps = resolveTemperatures(parts.agentConfig);
  if (temps.warning) (parts.warn ?? console.warn)(temps.warning);
  const dtype: Dtype = want === "f16" && parts.backend.name !== "cpu" && parts.backend.supports("f16") ? "f16" : "f32";
  return { enc, temps, batchSize, pad, dtype };
}

/**
 * Builds an agent from parts already in memory (no I/O): for browsers,
 * tests, or checkpoints from elsewhere. Validates like `Agent.__init__`,
 * clamps the calibration temperatures (warning through `warn`), and uploads
 * the weights to `backend` (all tensors in one batch; the Promise resolves
 * once they are on the device).
 */
export async function createAgent<T extends Tensor>(parts: AgentParts<T>): Promise<LayaAgent> {
  const v = validateParts(parts);
  const model = await loadDecisionModel(parts.backend, {
    encoderConfig: v.enc,
    agentConfig: parts.agentConfig,
    weights: toWeightGetter(parts.weights),
    dtype: v.dtype,
  });
  parts.backend.flush?.();
  return new Agent(parts, v, model);
}
