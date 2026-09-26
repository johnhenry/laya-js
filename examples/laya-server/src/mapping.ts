/**
 * Pure result/request mapping between laya-core's `PredictResult`/`Questions`
 * and the two remote SDKs (`@receptron/laya`'s ONNX export, `@typesafe-ai/sdk`'s
 * Jev API). No I/O -- what's actually testable without a real model or a
 * real network call.
 *
 * Both SDKs' generic `SystemOneResult<Q>` types resolve each answer from the
 * question map's shape, which collapses to a single branch when `Q` isn't a
 * literal object type (as it never is here -- these come from a parsed HTTP
 * body). The plain, ungeneric answer unions below are what actually describes
 * a value at runtime, and what these functions are typed against.
 */
import type { Answer, Json, Questions } from "@johnhenry/laya-core";
import type { ChoiceAnswer as OnnxChoiceAnswer, NoulAnswer as OnnxNoulAnswer, ScoreAnswer as OnnxScoreAnswer } from "@receptron/laya";
import type {
  ChoiceQuestion as JevChoiceQuestion,
  ChoiceResponse as JevChoiceResponse,
  EntryType as JevEntryType,
  NoulResponse as JevNoulResponse,
  Questions as JevQuestions,
  ScoreResponse as JevScoreResponse,
} from "@typesafe-ai/sdk";

type OnnxAnswer = OnnxChoiceAnswer | OnnxScoreAnswer | OnnxNoulAnswer;

export interface OnnxSystemOneResult {
  model: string;
  answers: Record<string, OnnxAnswer>;
  usage: { input_tokens: number; output_tokens: number };
}

type JevAnswer = JevChoiceResponse | JevScoreResponse | JevNoulResponse;

export interface JevSystemOneResult {
  model: string;
  answers: Record<string, JevAnswer>;
  usage: { input_tokens: number; output_tokens: number };
}

/**
 * Same shape as laya-core's `PredictResult`, except `usage.output_tokens` is a
 * real `number` here -- unlike Laya's own single-forward-pass architecture
 * (always 0, hence laya-core's literal-`0` type), neither remote SDK guarantees
 * zero output tokens, and forcing one to 0 would misreport real usage.
 */
export interface RemotePredictResult {
  model: string;
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number };
}

/** `max(p, 1-p)`, laya-core's own noul-confidence formula (agent.ts) -- reused because neither SDK reports a noul confidence of its own. */
function noulConfidence(p: number): number {
  return Math.max(p, 1 - p);
}

/** `@receptron/laya`'s `Answer` shapes carry a real `rl_agent.act_probability` -- no gap to paper over here. */
export function mapOnnxResult(raw: OnnxSystemOneResult): RemotePredictResult {
  const answers: Record<string, Answer> = {};
  for (const [id, a] of Object.entries(raw.answers)) {
    if (a.type === "choice") {
      answers[id] = { type: "choice", confidence: a.confidence, action: a.rl_agent, choice: a.choice, probabilities: a.probabilities };
    } else if (a.type === "score") {
      answers[id] = { type: "score", confidence: a.confidence, action: a.rl_agent, score: a.score, legend: a.legend, probabilities: a.probabilities };
    } else {
      answers[id] = { type: "noul", confidence: noulConfidence(a.noul), action: a.rl_agent, noul: a.noul };
    }
  }
  return { model: "laya-rl-agent", answers, usage: raw.usage };
}

/** Sentinel for "this backend doesn't report an act probability" -- rendered as N/A, never faked as 0. */
export const NO_ACT_PROBABILITY = Number.NaN;

/**
 * Jev's own `ChoiceResponse`/`ScoreResponse`/`NoulResponse` have no RL-agent
 * "should I act" concept at all (confirmed absent from `@typesafe-ai/sdk`'s
 * type declarations) -- `action.act_probability` is `NaN` for every answer here,
 * not a guess.
 */
export function mapJevResult(raw: JevSystemOneResult): RemotePredictResult {
  const answers: Record<string, Answer> = {};
  for (const [id, a] of Object.entries(raw.answers)) {
    const action = { act_probability: NO_ACT_PROBABILITY };
    if (a.type === "choice") {
      answers[id] = { type: "choice", confidence: a.confidence, action, choice: a.choice, probabilities: a.probabilities };
    } else if (a.type === "score") {
      answers[id] = { type: "score", confidence: a.confidence, action, score: a.score, legend: a.legend as unknown as Record<string, Json>, probabilities: a.probabilities };
    } else {
      answers[id] = { type: "noul", confidence: noulConfidence(a.noul), action, noul: a.noul };
    }
  }
  return { model: raw.model, answers, usage: raw.usage };
}

/**
 * Jev's `ChoiceQuestion.criteria` must be a label->description object; laya-core
 * (and web-playground's own question builder) also allows a plain label array
 * when no option has a description. Only `choice` needs this -- `score`'s tuple
 * and `noul`'s optional `{true,false}` are already shaped the same on both sides.
 */
export function toJevQuestions(questions: Questions): JevQuestions {
  const out: Record<string, unknown> = {};
  for (const [id, q] of Object.entries(questions)) {
    if (q.type === "choice" && Array.isArray(q.criteria)) {
      const criteria: Record<string, null> = Object.fromEntries(q.criteria.map((label) => [label, null]));
      out[id] = { type: "choice", instructions: q.instructions as JevEntryType, criteria } satisfies JevChoiceQuestion;
    } else {
      out[id] = q;
    }
  }
  return out as JevQuestions;
}
