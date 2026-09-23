/**
 * Opt-in embedding shortlist for high-cardinality choice questions
 * (port of laya-mlx `shortlist.py`). Embeds the state and each option,
 * keeps the top `k` by cosine similarity, and runs a single `predict` on the
 * reduced criteria.
 */
import { dumps } from "@johnhenry/pyjson";
import { renderOptions, serializeState, type Json } from "@johnhenry/laya-core";

export const DEFAULT_SHORTLIST_K = 20;

/** Maps texts to vectors, one per text (any numeric array-likes of equal length). */
export type EmbedFn = (texts: string[]) => Promise<ArrayLike<number>[]> | ArrayLike<number>[];

export interface ShortlistMeta {
  /** Kept labels in rank order (every label, in order, on pass-through). */
  labels: string[];
  /** Cosine similarities of `labels`, or null when nothing was dropped. */
  scores: number[] | null;
  k: number;
  n: number;
  passthrough: boolean;
}

/** Anything with `predict` (preferred) or `systemOne`: a LayaAgent, a Router, a mock. */
export interface Predictor {
  predict?(state: any, questions: any, ...rest: any[]): any;
  systemOne?(state: any, questions: any, ...rest: any[]): any;
}

export interface ShortlistOptions {
  /** Default 20. */
  k?: number;
  /** Default: `agent.embed` (the checkpoint's own encoder, mean-pooled). */
  embedFn?: EmbedFn;
  /** Forwarded as the third argument of predict (e.g. `{ model: "english" }` on a Router). */
  predictOptions?: unknown;
}

function checkK(k: unknown): number {
  if (typeof k !== "number" || !Number.isInteger(k) || k < 1) throw new Error(`k must be a positive integer, got ${String(k)}`);
  return k;
}

const isDict = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

function criteriaItems(criteria: unknown): [string, Json][] {
  let items: [string, Json][];
  if (isDict(criteria)) items = Object.entries(criteria) as [string, Json][];
  else if (Array.isArray(criteria)) items = criteria.map((c) => [c as string, null]);
  else throw new TypeError(`choice criteria must be a dict or list, got ${criteria === null ? "NoneType" : typeof criteria}`);
  if (!items.length) throw new Error("choice criteria must contain at least one option");
  const seen = new Set<string>();
  for (const [key] of items) {
    if (seen.has(key)) throw new Error(`choice criteria label ${JSON.stringify(key)} is duplicated`);
    seen.add(key);
  }
  return items;
}

function optionTexts(items: [string, Json][]): string[] {
  const crit: Record<string, Json> = {};
  for (const [k, v] of items) crit[k] = v;
  const texts = renderOptions({ t: "choice", crit, labels: items.map(([k]) => k) });
  if (texts.length !== items.length) throw new Error("could not render every choice option");
  return texts;
}

function queryText(state: unknown, instructions: unknown): string {
  const body = serializeState(state);
  if (instructions === null || instructions === undefined || instructions === "") return body;
  const ins = typeof instructions === "string" ? instructions : dumps(instructions, { ensureAscii: false });
  return `${ins}\n${body}`;
}

async function embeddings(embedFn: EmbedFn, texts: string[]): Promise<Float64Array[]> {
  if (typeof embedFn !== "function") throw new TypeError("embed_fn must be callable");
  const raw = await embedFn([...texts]);
  const rows = Array.isArray(raw) ? raw : [];
  const dim = rows[0]?.length ?? 0;
  if (rows.length !== texts.length || dim < 1 || rows.some((r) => !r || r.length !== dim)) {
    throw new Error(`embed_fn must return an array of shape (${texts.length}, dim), got ${rows.length} rows of ${dim}`);
  }
  // np.nan_to_num(nan=0, posinf=0, neginf=0)
  return rows.map((r) => Float64Array.from(r, (x) => (Number.isFinite(x) ? x : 0)));
}

function cosine(query: Float64Array, docs: Float64Array[]): Float64Array {
  const norm = (v: Float64Array) => Math.hypot(...v);
  const qn = norm(query);
  const sims = new Float64Array(docs.length);
  if (qn === 0) return sims;
  docs.forEach((d, i) => {
    const denom = norm(d) * qn;
    if (denom > 0) {
      let dot = 0;
      for (let j = 0; j < d.length; j++) dot += d[j]! * query[j]!;
      sims[i] = dot / denom;
    }
  });
  return sims;
}

async function rank(state: unknown, criteria: unknown, embedFn: EmbedFn, k: number, instructions: unknown) {
  const checked = checkK(k);
  const items = criteriaItems(criteria);
  const n = items.length;
  const keys = items.map(([key]) => key);
  if (checked >= n) return { labels: keys, scores: null, passthrough: true, n };
  const matrix = await embeddings(embedFn, [queryText(state, instructions), ...optionTexts(items)]);
  const sims = cosine(matrix[0]!, matrix.slice(1));
  // np.argsort(-sims, kind="mergesort"): stable, so ties keep the earlier label
  const order = keys.map((_, i) => i).sort((a, b) => sims[b]! - sims[a]! || a - b).slice(0, checked);
  return { labels: order.map((i) => keys[i]!), scores: order.map((i) => sims[i]!), passthrough: false, n };
}

/**
 * Top-`k` choice labels for `state` (`shortlist_choice`). `embedFn` is called
 * once with the query text first, then one string per option (as
 * `renderOptions` renders a choice); not at all when `k >= #labels`.
 */
export async function shortlistChoice(
  state: unknown,
  criteria: unknown,
  embedFn: EmbedFn,
  k: number = DEFAULT_SHORTLIST_K,
  opts: { instructions?: unknown } = {},
): Promise<string[]> {
  return (await rank(state, criteria, embedFn, k, opts.instructions)).labels;
}

/**
 * Shortlist each choice question, then call `predict` (or `systemOne`) once
 * (`predict_shortlist`). The result gains `shortlist[qid] = { labels, scores,
 * k, n, passthrough }` for every choice question; probabilities on a
 * shortlisted question are over the kept labels only. `questions` is not mutated.
 */
export async function predictShortlist<R = any>(
  agent: Predictor & { embed?: (texts: string[]) => Promise<ArrayLike<number>[]> },
  state: unknown,
  questions: Record<string, unknown>,
  opts: ShortlistOptions = {},
): Promise<R & { shortlist: Record<string, ShortlistMeta> }> {
  if (!isDict(questions)) throw new TypeError("questions must be a dict of question id -> definition");
  const k = checkK(opts.k ?? DEFAULT_SHORTLIST_K);
  const embedFn: EmbedFn | undefined = opts.embedFn ?? (typeof agent.embed === "function" ? (t) => agent.embed!(t) : undefined);
  const reduced: Record<string, unknown> = {};
  const meta: Record<string, ShortlistMeta> = {};
  for (const [qid, qdef] of Object.entries(questions)) {
    if (!isDict(qdef) || qdef.type !== "choice") {
      reduced[qid] = qdef;
      continue;
    }
    if (!("criteria" in qdef)) throw new Error(`question ${JSON.stringify(qid)} is a choice but has no criteria`);
    const r = await rank(state, qdef.criteria, embedFn ?? missingEmbed, k, qdef.instructions);
    meta[qid] = { labels: [...r.labels], scores: r.scores, k, n: r.n, passthrough: r.passthrough };
    if (r.passthrough) {
      reduced[qid] = qdef;
      continue;
    }
    const crit = qdef.criteria;
    reduced[qid] = {
      ...qdef,
      criteria: isDict(crit) ? Object.fromEntries(r.labels.map((l) => [l, crit[l]])) : [...r.labels],
    };
  }
  const fn = agent.predict ?? agent.systemOne;
  if (typeof fn !== "function") throw new TypeError("agent must provide predict or systemOne");
  const extra = opts.predictOptions === undefined ? [] : [opts.predictOptions];
  const result = await fn.call(agent, state, reduced, ...extra);
  if (!isDict(result)) throw new TypeError(`predict/systemOne must return an object, got ${result === null ? "null" : typeof result}`);
  return { ...(result as R), shortlist: meta };
}

const missingEmbed: EmbedFn = () => {
  throw new TypeError("embed_fn must be callable (pass opts.embedFn, or an agent with embed())");
};
