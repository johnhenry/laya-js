/**
 * Question normalization, input preparation, collation, calibration and result
 * formatting (port of laya-mlx `agent.py` minus the tensors).
 */
import { dumps, pyFloatRepr, pyFormatG, pyRound } from "@johnhenry/pyjson";
import {
  buildSequence,
  choiceLabels,
  clampTemperature,
  confidenceFromProbs,
  pyFloatOf,
  renderOptions,
  TEMP_MAX,
  TEMP_MIN,
  tempBucket,
} from "./common.ts";
import { f32, expF32, softmaxF32, sumF32, sumF64 } from "./numpy.ts";
import type { PrefixCache } from "./prefix-cache.ts";
import {
  QTYPES,
  type AgentConfig,
  type Answer,
  type Batch,
  type BatchOutputs,
  type InternalQuestion,
  type Json,
  type LayaTokenizer,
  type PredictResult,
  type PreparedItem,
  type QuestionType,
} from "./types.ts";

const isDict = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/** Python `repr()` for the values an error message may quote. */
export function pyRepr(v: unknown): string {
  if (v === null || v === undefined) return "None";
  if (v === true) return "True";
  if (v === false) return "False";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : pyFloatRepr(v);
  if (typeof v === "string") {
    const q = v.includes("'") && !v.includes('"') ? '"' : "'";
    const body = v
      .replaceAll("\\", "\\\\")
      .replaceAll("\n", "\\n")
      .replaceAll("\r", "\\r")
      .replaceAll("\t", "\\t")
      .replaceAll(q, "\\" + q);
    return q + body + q;
  }
  return dumps(v);
}

/** Validate and normalize one public question (`Agent._to_internal`). Throws like Python's ValueError. */
export function toInternal(qdef: unknown): InternalQuestion {
  if (!isDict(qdef)) throw new Error("Each question must be a dictionary");
  const kind = qdef.type;
  if (typeof kind !== "string" || !Object.hasOwn(QTYPES, kind)) {
    throw new Error(`Unknown question type ${pyRepr(kind)}; expected choice, score, or noul`);
  }
  if (!("instructions" in qdef)) throw new Error("Question is missing instructions");
  let criteria = (qdef.criteria ?? null) as unknown;
  let labels: string[] | undefined;
  if (kind === "choice") {
    if (Array.isArray(criteria)) {
      if (!criteria.every((c) => typeof c === "string")) throw new Error("Choice labels must be strings");
      if (new Set(criteria).size !== criteria.length) throw new Error("Choice labels must be unique");
      labels = [...(criteria as string[])];
      const obj: Record<string, Json> = {};
      for (const c of labels) obj[c] = null;
      criteria = obj;
    }
    if (!isDict(criteria) || Object.keys(criteria).length === 0) {
      throw new Error("Choice criteria must be a nonempty dictionary or list");
    }
  } else if (kind === "score") {
    if (!Array.isArray(criteria) || criteria.length === 0) throw new Error("Score criteria must be a nonempty list");
  } else if (criteria !== null && !isDict(criteria)) {
    throw new Error("Noul criteria must be a dictionary with false/true descriptions");
  }
  const instructions = qdef.instructions;
  const ins = typeof instructions === "string" ? instructions : dumps(instructions);
  const q: InternalQuestion = { t: kind as QuestionType, ins, crit: criteria as InternalQuestion["crit"] };
  if (labels) q.labels = labels;
  return q;
}

export interface Prepared {
  items: PreparedItem[];
  internal: InternalQuestion[];
}

export interface PrepareOptions {
  /** Reuse tokenized question prefixes across calls (`cache_prompts=True`). */
  cache?: PrefixCache | null;
}

/** Upstream-compatible model inputs (`Agent.prepare`). `internal[i].id` is the question id. */
export function prepare(
  tok: LayaTokenizer,
  state: unknown,
  questions: unknown,
  cfg: AgentConfig = {},
  options: PrepareOptions = {},
): Prepared {
  if (options.cache) return options.cache.prepare(tok, state, questions, cfg);
  if (!isDict(questions)) throw new Error("questions must be a dictionary keyed by question id");
  const maxLen = cfg.max_len ?? 512;
  const headLen = cfg.head_max_len ?? 192;
  const items: PreparedItem[] = [];
  const internal: InternalQuestion[] = [];
  for (const [qid, definition] of Object.entries(questions)) {
    const q = toInternal(definition);
    const { ids, markers } = buildSequence(tok, state, q, maxLen, headLen);
    if (markers.length !== renderOptions(q).length) {
      throw new Error(`Question ${pyRepr(qid)} has too many options for the token budget`);
    }
    items.push({ ids, markers, qtype: QTYPES[q.t] });
    internal.push({ ...q, id: qid });
  }
  return { items, internal };
}

export interface CollateOptions {
  padToMultiple?: number | null;
  maxLength?: number | null;
}

/** Pad a chunk of prepared items into a {@link Batch} (`collate_items`). */
export function collate(items: readonly PreparedItem[], padId: number, options: CollateOptions = {}): Batch {
  if (items.length === 0) throw new Error("Cannot collate an empty batch");
  const { padToMultiple, maxLength } = options;
  const n = items.length;
  let length = Math.max(...items.map((it) => it.ids.length));
  if (padToMultiple) {
    length = Math.ceil(length / padToMultiple) * padToMultiple;
    if (maxLength !== undefined && maxLength !== null) length = Math.min(length, maxLength);
  }
  const count = Math.max(2, ...items.map((it) => it.markers.length));
  const inputIds = new Int32Array(n * length).fill(padId);
  const attentionMask = new Uint8Array(n * length);
  const markerPos = new Int32Array(n * count);
  const markerMask = new Uint8Array(n * count);
  const qtype = new Int32Array(n);
  items.forEach((it, i) => {
    if (it.ids.length > length) throw new Error("Item is longer than the batch length");
    inputIds.set(it.ids, i * length);
    attentionMask.fill(1, i * length, i * length + it.ids.length);
    markerPos.set(it.markers, i * count);
    markerMask.fill(1, i * count, i * count + it.markers.length);
    qtype[i] = it.qtype;
  });
  return { size: n, length, markerCount: count, inputIds, attentionMask, markerPos, markerMask, qtype };
}

export interface Temperatures {
  /** Clamped per-qtype temperatures (applied). */
  temperature: [number, number, number];
  /** Clamped per-bucket temperatures (applied). */
  temperatureByOptions: Record<string, number>;
  /** What the checkpoint shipped. */
  temperatureRaw: unknown[];
  temperatureByOptionsRaw: Record<string, unknown>;
  /** Entries outside [TEMP_MIN, TEMP_MAX], formatted like Python (`choice:11+=0.1006`). */
  rejected: string[];
  /** The RuntimeWarning text Python emits when anything was clamped, else null. */
  warning: string | null;
}

/** Validate, clamp and report the checkpoint's calibration temperatures (agent.py:134-164). */
export function resolveTemperatures(cfg: AgentConfig): Temperatures {
  const raw = (cfg.temperature ?? [1.0, 1.0, 1.0]) as unknown[];
  const byRaw = (cfg.temperature_by_options ?? {}) as Record<string, unknown>;
  if (!Array.isArray(raw) || raw.length !== 3 || [...raw, ...Object.values(byRaw)].some((t) => {
    const x = pyFloatOf(t);
    return !Number.isFinite(x) || x <= 0;
  })) {
    throw new Error("Calibration temperatures must be finite and positive");
  }
  const temperature = raw.map((t) => clampTemperature(t)) as [number, number, number];
  const temperatureByOptions: Record<string, number> = {};
  for (const [k, v] of Object.entries(byRaw)) temperatureByOptions[k] = clampTemperature(v);
  const rejected = [
    ...Object.entries(byRaw)
      .filter(([, v]) => clampTemperature(v) !== pyFloatOf(v))
      .map(([k, v]) => `${k}=${pyFormatG(pyFloatOf(v), 4)}`),
    ...raw.flatMap((t, i) => (clampTemperature(t) !== pyFloatOf(t) ? [`temperature[${i}]=${pyFormatG(pyFloatOf(t), 4)}`] : [])),
  ];
  const warning = rejected.length
    ? `laya: this checkpoint ships temperatures outside [${pyFormatG(TEMP_MIN)}, ${pyFormatG(TEMP_MAX)}] which would ` +
      `distort confidence; clamping ${rejected.join(", ")}. Treat confidence from the affected buckets ` +
      "as uncalibrated."
    : null;
  return { temperature, temperatureByOptions, temperatureRaw: raw, temperatureByOptionsRaw: byRaw, rejected, warning };
}

/** The temperature applied to a row: bucket override, else the per-qtype value. */
export function temperatureFor(temps: Pick<Temperatures, "temperature" | "temperatureByOptions">, qtype: number, k: number): number {
  return temps.temperatureByOptions[tempBucket(qtype, k)] ?? temps.temperature[qtype]!;
}

/**
 * Turn raw model outputs into the published result (`Agent.system_one`), bit-identical to
 * Python: probabilities in float32 like numpy, rounding with CPython `round(x, 4)`.
 *
 * `outputs[c]` holds chunk `c` (consecutive items, any chunk size); its row count is
 * `act.length / nAct` and its marker stride is `logits.length / rows`.
 */
export function formatResults(
  internal: readonly InternalQuestion[],
  items: readonly PreparedItem[],
  outputs: readonly BatchOutputs[],
  temps: Pick<Temperatures, "temperature" | "temperatureByOptions">,
): PredictResult {
  if (internal.length !== items.length) throw new Error("internal and items must have the same length");
  const answers: Record<string, Answer> = {};
  let offset = 0;
  for (const out of outputs) {
    const { logits, act, nAct } = out;
    const rows = act.length / nAct;
    if (!Number.isInteger(rows) || rows < 1) throw new Error("act length is not a multiple of nAct");
    const stride = logits.length / rows;
    if (!Number.isInteger(stride)) throw new Error("logits length is not a multiple of the row count");
    for (const v of logits) if (!Number.isFinite(v)) throw new Error("Non-finite model outputs; retry with dtype='float32'");
    for (const v of act) if (!Number.isFinite(v)) throw new Error("Non-finite model outputs; retry with dtype='float32'");
    for (let row = 0; row < rows; row++) {
      const index = offset + row;
      const item = items[index];
      const q = internal[index];
      if (!item || !q) throw new Error("More output rows than prepared items");
      // act softmax over the row (f32)
      const a = act.subarray(row * nAct, (row + 1) * nAct);
      let am = -Infinity;
      for (const v of a) if (v > am) am = v;
      const ea = Float32Array.from(a, (v) => expF32(f32(v - am)));
      const actProb = f32(ea[0]! / sumF32(ea));

      const k = item.markers.length;
      if (k > stride) throw new Error("Item has more markers than the output row");
      const scale = temperatureFor(temps, item.qtype, k);
      const p = softmaxF32(logits.subarray(row * stride, row * stride + k), scale);
      const answer: Answer = {
        type: q.t,
        confidence: pyRound(confidenceFromProbs(p, k), 4),
        action: { act_probability: pyRound(actProb, 4) },
      };
      if (q.t === "choice") {
        const labels = choiceLabels(q);
        let best = 0;
        for (let i = 1; i < k; i++) if (p[i]! > p[best]!) best = i;
        answer.choice = labels[best]!;
        const probs: Record<string, number> = {};
        labels.forEach((label, i) => {
          if (i < k) probs[label] = pyRound(p[i]!, 4);
        });
        answer.probabilities = probs;
      } else if (q.t === "score") {
        const crit = q.crit as Json[];
        answer.score = pyRound(sumF64(Array.from(p, (v, i) => i * v)), 4);
        const legend: Record<string, Json> = {};
        crit.forEach((v, i) => (legend[String(i)] = v));
        answer.legend = legend;
        const probs: Record<string, number> = {};
        p.forEach((v, i) => (probs[String(i)] = pyRound(v, 4)));
        answer.probabilities = probs;
      } else {
        const p1 = p[1]!;
        answer.noul = pyRound(p1, 4);
        answer.confidence = pyRound(Math.max(p1, 1.0 - p1), 4);
      }
      const qid = q.id ?? String(index);
      answers[qid] = answer;
    }
    offset += rows;
  }
  if (offset !== items.length) throw new Error(`Outputs cover ${offset} rows but ${items.length} items were prepared`);
  return {
    model: "laya-rl-agent",
    answers,
    usage: { input_tokens: items.reduce((n, it) => n + it.ids.length, 0), output_tokens: 0 },
  };
}

/** Split prepared items into chunks of `batchSize` (the order `formatResults` expects). */
export function chunkItems<T>(items: readonly T[], batchSize: number): T[][] {
  if (!Number.isInteger(batchSize) || batchSize < 1) throw new Error("batch_size must be a positive integer");
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += batchSize) out.push(items.slice(i, i + batchSize));
  return out;
}
