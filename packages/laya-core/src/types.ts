/**
 * Contract types for Laya typed decisions (mirrors laya-mlx agent.py/common.py).
 * Tensor-free: nothing here depends on a backend.
 */

export type QuestionType = "choice" | "score" | "noul";

/** choice = 0, score = 1, noul = 2 (the type_emb row index). */
export const QTYPES: Readonly<Record<QuestionType, 0 | 1 | 2>> = { choice: 0, score: 1, noul: 2 };

/** JSON-ish value, rendered with Python json.dumps semantics (see @johnhenry/pyjson). */
export type Json = null | boolean | number | string | Json[] | { [k: string]: Json };

/** Public question definition, as accepted by `predict`. */
export type Question =
  | { type: "choice"; instructions: Json; criteria: string[] | Record<string, Json> }
  | { type: "score"; instructions: Json; criteria: Json[] }
  | { type: "noul"; instructions: Json; criteria?: { true?: Json; false?: Json } | null };

export type Questions = Record<string, Question>;

/** State: a string passes through; anything else is json.dumps(ensure_ascii=False). */
export type State = string | Json;

/** Normalized question (agent.py `_to_internal`). choice lists become {label: null}. */
export interface InternalQuestion {
  t: QuestionType;
  ins: string;
  crit: Record<string, Json> | Json[] | { true?: Json; false?: Json } | null;
  /**
   * Choice labels in Python order. Set when criteria came as a list: a JS object
   * reorders integer-like keys ("10" before "a"), a Python dict does not.
   * When absent, `Object.keys(crit)` is the label order.
   */
  labels?: string[];
  /** Question id (the key in `questions`), set by `prepare`; used by `formatResults`. */
  id?: string;
}

/** Minimal tokenizer surface (add_special_tokens is always false). */
export interface LayaTokenizer {
  encode(text: string): number[];
  readonly clsToken: string; readonly clsTokenId: number;
  readonly sepToken: string; readonly sepTokenId: number;
  readonly padToken: string; readonly padTokenId: number;
  readonly maskToken: string; readonly maskTokenId: number;
}

/** One prepared row: build_sequence output + qtype. */
export interface PreparedItem {
  ids: number[];
  markers: number[];
  qtype: 0 | 1 | 2;
}

/** collate_items output; row-major flat arrays. */
export interface Batch {
  size: number;          // B
  length: number;        // L (padded)
  markerCount: number;   // M = max(2, max markers)
  inputIds: Int32Array;      // [B, L], pad_id padded
  attentionMask: Uint8Array; // [B, L]
  markerPos: Int32Array;     // [B, M], 0 padded
  markerMask: Uint8Array;    // [B, M]
  qtype: Int32Array;         // [B]
}

/** Raw model outputs for one batch (f32). */
export interface BatchOutputs {
  logits: Float32Array; // [B, M], masked slots = -1e4
  act: Float32Array;    // [B, nAct] (nAct = 2 for shipped checkpoints)
  nAct: number;
}

/** rl_agent_config.json (fields used by inference). */
export interface AgentConfig {
  max_len?: number;          // default 512
  head_max_len?: number;     // default 192
  head_layers?: number;      // default 2
  act_costs?: Record<string, number>;
  temperature?: [number, number, number];
  temperature_by_options?: Record<string, number>;
  [extra: string]: unknown;
}

export interface Answer {
  type: QuestionType;
  confidence: number;
  action: { act_probability: number };
  choice?: string;
  score?: number;
  legend?: Record<string, Json>;
  noul?: number;
  probabilities?: Record<string, number>;
}

export interface PredictResult {
  model: "laya-rl-agent";
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: 0 };
  routing?: unknown;
  shortlist?: Record<string, unknown>;
}
