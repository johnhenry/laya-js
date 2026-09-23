/**
 * Prompt construction and calibration (port of laya-mlx `laya_mlx/common.py`).
 */
import { dumps } from "@johnhenry/pyjson";
import type { InternalQuestion, Json, LayaTokenizer, QuestionType } from "./types.ts";
import { f32, logF32, sumF32 } from "./numpy.ts";

export const QTYPE_NAMES: Readonly<Record<0 | 1 | 2, QuestionType>> = { 0: "choice", 1: "score", 2: "noul" };

/** A string passes through; anything else is `json.dumps(state, ensure_ascii=False)`. */
export function serializeState(state: unknown): string {
  if (typeof state === "string") return state;
  return dumps(state, { ensureAscii: false });
}

/** Strings pass through; anything structured becomes compact JSON (`", "`, `": "`). */
export function renderCriterion(value: unknown): string {
  if (typeof value === "string") return value;
  return dumps(value, { ensureAscii: false, separators: [", ", ": "] });
}

const isDict = (v: unknown): v is Record<string, Json> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/** Choice labels in order (`list(q["crit"])`). */
export function choiceLabels(q: Pick<InternalQuestion, "crit" | "labels">): string[] {
  return q.labels ? [...q.labels] : Object.keys(q.crit as Record<string, Json>);
}

/** Option texts in label-index order. Noul is always [false, true]. */
export function renderOptions(q: Pick<InternalQuestion, "t" | "crit" | "labels">): string[] {
  const { t } = q;
  if (t === "choice") {
    const crit = q.crit as Record<string, Json>;
    // only None/"" mean "no description"; 0 and false are legitimate criterion values
    return choiceLabels(q).map((k) => {
      const v = crit[k];
      return v === null || v === undefined || v === "" ? k : `${k}: ${renderCriterion(v)}`;
    });
  }
  if (t === "score") {
    return (q.crit as Json[]).map((c, i) => `level ${i}: ${renderCriterion(c)}`);
  }
  const crit = isDict(q.crit) ? q.crit : {};
  const f = crit.false;
  const tr = crit.true;
  const blank = (v: unknown) => v === null || v === undefined || v === "";
  return [
    "false: " + (blank(f) ? "no, the statement does not hold" : renderCriterion(f)),
    "true: " + (blank(tr) ? "yes, the statement holds" : renderCriterion(tr)),
  ];
}

/** `str.replace(old, new)` (all occurrences; an empty `old` inserts between every character). */
const pyReplace = (s: string, old: string, rep: string) => s.split(old).join(rep);

export interface Sequence {
  ids: number[];
  markers: number[];
}

/** The question-only prefix, before state tokens and final truncation. */
export function buildPrefix(
  tok: LayaTokenizer,
  q: Pick<InternalQuestion, "t" | "ins" | "crit" | "labels">,
  headMaxLen = 192,
  optionOrder?: readonly number[] | null,
): Sequence {
  const maskTok = tok.maskToken;
  const opts = renderOptions(q);
  const order = optionOrder ?? opts.map((_, i) => i);
  const ins = pyReplace(String(q.ins), maskTok, " ");
  let headIds = tok.encode(`${q.t} question: ${ins}`);
  let optIds = order.map((i) => [tok.maskTokenId, ...tok.encode(" " + pyReplace(opts[i]!, maskTok, " ")).slice(0, 48)]);
  const total = () => optIds.reduce((n, o) => n + o.length, 0);
  let optBudget = headMaxLen - total();
  if (optBudget < 16) {
    const per = Math.max(4, Math.floor((headMaxLen - 16) / Math.max(1, optIds.length)));
    optIds = optIds.map((o) => o.slice(0, per));
    optBudget = headMaxLen - total();
  }
  headIds = headIds.slice(0, Math.max(8, optBudget));
  const ids = [tok.clsTokenId, ...headIds, tok.sepTokenId];
  const markers: number[] = [];
  for (const o of optIds) {
    markers.push(ids.length);
    ids.push(...o);
  }
  ids.push(tok.sepTokenId);
  return { ids, markers };
}

/** Python `lst[-n:]` (note `lst[-0:]` is the whole list). */
const tail = <T>(a: T[], n: number): T[] => (n === 0 ? a : a.slice(-n));

/** `[CLS] <type> question: ins [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`. */
export function buildSequence(
  tok: LayaTokenizer,
  state: unknown,
  q: Pick<InternalQuestion, "t" | "ins" | "crit" | "labels">,
  maxLen = 512,
  headMaxLen = 192,
  optionOrder?: readonly number[] | null,
  truncateLeft = false,
): Sequence {
  const prefix = buildPrefix(tok, q, headMaxLen, optionOrder);
  const room = Math.max(0, maxLen - prefix.ids.length - 1);
  let st = tok.encode(pyReplace(serializeState(state), tok.maskToken, " "));
  st = truncateLeft ? tail(st, room) : st.slice(0, room);
  const ids = [...prefix.ids, ...st, tok.sepTokenId];
  return { ids: ids.slice(0, maxLen), markers: prefix.markers.filter((m) => m < maxLen) };
}

/** Normalized Shannon entropy confidence `1 - H(p)/log(k)`, computed in float32 like numpy. */
export function confidenceFromProbs(p: ArrayLike<number>, k: number): number {
  if (k < 2) return 1.0;
  const lo = f32(1e-12);
  const terms = new Float32Array(k);
  for (let i = 0; i < k; i++) {
    const pi = f32(p[i]!);
    const c = pi < lo ? lo : pi > 1 ? 1 : pi;
    terms[i] = f32(pi * logF32(c));
  }
  const ent = -sumF32(terms);
  const v = f32(1.0 - f32(ent / f32(Math.log(k))));
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Calibration bucket name, e.g. `choice:3-5`. */
export function tempBucket(qtype: number, k: number): string {
  const size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  const name = QTYPE_NAMES[qtype as 0 | 1 | 2];
  if (!name) throw new Error(`Unknown qtype ${qtype}`);
  return `${name}:${size}`;
}

/** Temperatures below this sharpen logits (see common.py); refuse to apply them. */
export const TEMP_MIN = 0.5;
export const TEMP_MAX = 5.0;

/** Python `float(t)` for the inputs a JSON config can hold; NaN when it would raise. */
export function pyFloatOf(t: unknown): number {
  if (typeof t === "number") return t;
  if (typeof t === "boolean") return t ? 1 : 0;
  if (typeof t === "string") {
    const s = t.trim().toLowerCase().replaceAll("_", "");
    if (/^[+-]?(inf|infinity)$/.test(s)) return s.startsWith("-") ? -Infinity : Infinity;
    if (/^[+-]?nan$/.test(s)) return NaN;
    if (/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/.test(s)) return Number(s);
  }
  return NaN;
}

/** `t` confined to [lo, hi], falling back to 1.0 when it is not a finite number. */
export function clampTemperature(t: unknown, lo = TEMP_MIN, hi = TEMP_MAX): number {
  const x = pyFloatOf(t);
  if (!Number.isFinite(x)) return 1.0;
  return Math.min(hi, Math.max(lo, x));
}
