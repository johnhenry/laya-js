/**
 * Real Laya predictions, with an explicit optional deterministic safety
 * shield -- same shape as snake-terminal's/flappy-terminal's `policy.ts`.
 * The `choice` question's `criteria` covers every hop in `legalHops(state)`
 * for this turn (the variable-cardinality analog of Snake's fixed
 * 4-direction criteria): illegal hops (mandatory capture violated, or the
 * wrong piece/move mid-chain) are shown too, with descriptive text, exactly
 * like Snake shows "Blocked." directions. The shield then restricts
 * execution to `compliantHops` when the model's raw top-1 isn't compliant.
 */
import type { Questions, State } from "@johnhenry/laya-core";
import { opponent, type Player } from "./board.ts";
import { applyHop, compliantHopsFor, legalHopsFor, materialCount, type GameSnapshot } from "./game.ts";
import { hopKey, type Hop } from "./moves.ts";

export const DEFAULT_MODEL = "aac6fef/laya-multilingual-mlx";

export type PromptStyle = "compact" | "detailed";

/** The subset of a `@johnhenry/laya` agent the policy needs. */
export interface PredictAgent {
  predict(state: State, questions: Questions): Promise<PredictLike> | PredictLike;
}

export interface PredictLike {
  answers: Record<string, { probabilities?: Record<string, number>; noul?: number }>;
  usage: { input_tokens: number; output_tokens?: number };
}

// A type alias (not an interface) so it is assignable to laya-core's `Questions`.
export type CheckersQuestions = {
  move: { type: "choice"; instructions: string; criteria: Record<string, string> };
  material_at_risk: { type: "noul"; instructions: string };
  captures_material: { type: "noul"; instructions: string };
};

export interface CheckersPrompt {
  state: string;
  questions: CheckersQuestions;
  legal: Hop[];
  compliant: Hop[];
}

export interface Decision {
  probabilities: Record<string, number>;
  proposed: Hop;
  executed: Hop;
  compliant_hops: Hop[];
  intervened: boolean;
  material_at_risk: number;
  captures_material: number;
  inference_ms: number;
  decision_ms: number;
  input_tokens: number;
  output_tokens: number;
  compliant_count: number;
}

/** Descriptive text for one hop, tiered exactly as designed: illegal (2 reasons) / simple / capture-ends / capture-continues / capture-kings-and-stops. Uses `applyHop` itself as a pure scratch simulation for the last three tiers -- text only, never legality. */
function classify(state: GameSnapshot, hop: Hop, compliant: Hop[], style: PromptStyle): string {
  const isCompliant = compliant.some((h) => hopKey(h) === hopKey(hop));
  if (!isCompliant) {
    if (state.forcedContinuation) {
      return style === "compact"
        ? `Illegal. Mid-capture chain with the piece on ${state.forcedContinuation.square} -- no other piece or move may act this turn.`
        : `Illegal: a capture chain is already in progress with the piece on ${state.forcedContinuation.square}. Only that piece's own further captures are legal this turn.`;
    }
    return style === "compact"
      ? "Illegal. A capture is available elsewhere -- mandatory capture forbids this simple move."
      : "Illegal: mandatory capture is in effect this turn (a capture is available with another piece), so this non-capturing move is forbidden.";
  }
  if (hop.kind === "simple") {
    return style === "compact" ? "Simple move. No captures available this turn." : "A simple move -- no capture is available anywhere on the board this turn.";
  }
  const next = applyHop(state, hop);
  const kinged = next.moveHistory[next.moveHistory.length - 1]!.kinged;
  if (kinged) {
    return style === "compact"
      ? "Capture. Reaches the back row -- crowned king, chain ends here even if another jump would be available."
      : "Capture that lands on the back row: this piece is crowned king immediately, and the capture chain stops here even though a further jump might otherwise be possible.";
  }
  if (next.forcedContinuation) {
    return style === "compact"
      ? "Capture. Must continue jumping with this same piece afterward."
      : "Capture. Another capture is available from the landing square, so this same piece must keep jumping before the turn passes.";
  }
  return style === "compact"
    ? "Capture. Removes an opponent piece; turn ends here."
    : "Capture that removes an opponent piece. No further jump is available from the landing square, so the turn passes after this move.";
}

/** Planner features -> the exact state text and questions sent to the agent. */
export function buildPrompt(state: GameSnapshot, style: PromptStyle = "compact"): CheckersPrompt {
  const legal = legalHopsFor(state);
  const compliant = compliantHopsFor(state);
  const captureCount = legal.filter((h) => h.kind === "capture").length;
  const material = materialCount(state.board);
  const mover: Player = state.toMove;
  const other = opponent(mover);
  const criteria: Record<string, string> = {};
  for (const hop of legal) criteria[hopKey(hop)] = classify(state, hop, compliant, style);

  let stateText: string;
  let move: string;
  const risk = "Would the moved piece (or any other piece) be capturable by the opponent on their next turn?";
  const captures = "Does the proposed hop capture an opponent piece?";
  if (style === "compact") {
    stateText =
      `${mover} to move. ${legal.length} hop(s) possible, ${captureCount} of them capture(s).` +
      (state.forcedContinuation ? ` Must continue capturing with the piece on ${state.forcedContinuation.square}.` : captureCount > 0 ? " Capture is mandatory." : "");
    move = "Choose the best legal hop.";
  } else {
    stateText =
      `Checkers, turn ${state.turnNumber}. ${mover} to move with ${material[mover].men} men and ${material[mover].kings} kings ` +
      `against ${material[other].men} men and ${material[other].kings} kings. ` +
      `${legal.length} hop(s) are geometrically possible, ${captureCount} of them capture(s).` +
      (state.forcedContinuation
        ? ` A capture chain is in progress: the piece on ${state.forcedContinuation.square} must continue jumping.`
        : captureCount > 0
          ? " Capture is mandatory this turn."
          : " No captures are available.");
    move = "Choose the best hop among the legal options for this turn -- captures are mandatory when available.";
  }
  return {
    state: stateText,
    questions: {
      move: { type: "choice", instructions: move, criteria },
      material_at_risk: { type: "noul", instructions: risk },
      captures_material: { type: "noul", instructions: captures },
    },
    legal,
    compliant,
  };
}

export interface PolicyOptions {
  /** Restrict execution to mandatory-capture-compliant hops (default true). */
  guarded?: boolean;
  prompt?: PromptStyle;
  now?: () => number;
}

export class LayaPolicy {
  agent: PredictAgent;
  guarded: boolean;
  readonly prompt: PromptStyle;
  readonly #now: () => number;

  constructor(agent: PredictAgent, options: PolicyOptions = {}) {
    this.agent = agent;
    this.guarded = options.guarded ?? true;
    this.prompt = options.prompt ?? "compact";
    if (this.prompt !== "compact" && this.prompt !== "detailed") throw new RangeError("prompt must be compact or detailed");
    this.#now = options.now ?? (() => performance.now());
  }

  async decide(state: GameSnapshot): Promise<Decision> {
    const started = this.#now();
    const p = buildPrompt(state, this.prompt);
    if (!p.legal.length) throw new Error("Cannot decide: the game is already over (no legal hops)");
    const inferenceStart = this.#now();
    const output = await this.agent.predict(p.state, p.questions);
    const inferenceMs = this.#now() - inferenceStart;
    return decisionFrom(output, p, this.guarded, inferenceMs, this.#now() - started);
  }
}

/** The shield + bookkeeping half of `decide` (pure; exported for tests and replays). */
export function decisionFrom(output: PredictLike, p: CheckersPrompt, guarded: boolean, inferenceMs = 0, decisionMs = 0): Decision {
  const answers = output.answers;
  const probabilities = answers.move!.probabilities as Record<string, number>;
  const scores = [...Object.values(probabilities), answers.material_at_risk!.noul!, answers.captures_material!.noul!];
  if (scores.some((v) => !Number.isFinite(v) || v < 0 || v > 1)) {
    throw new Error("Model returned an invalid probability; no hop executed");
  }
  const argmaxHop = (hops: readonly Hop[]): Hop => hops.reduce((best, h) => ((probabilities[hopKey(h)] ?? -Infinity) > (probabilities[hopKey(best)] ?? -Infinity) ? h : best), hops[0]!);
  const proposed = argmaxHop(p.legal);
  const compliantKeys = new Set(p.compliant.map(hopKey));
  let executed = proposed;
  let intervened = false;
  if (guarded && !compliantKeys.has(hopKey(proposed))) {
    // Empty `p.compliant` here is an assertion, not a fallback: by construction
    // (see moves.ts) compliant is empty iff legal is empty, i.e. the game is
    // over -- and `LayaPolicy.decide` already refuses to prompt in that case.
    if (p.compliant.length === 0) {
      throw new Error("decisionFrom invariant violated: compliant hops empty during an active decision; the caller must check for game over before prompting.");
    }
    executed = argmaxHop(p.compliant);
    intervened = true;
  }
  return {
    probabilities,
    proposed,
    executed,
    compliant_hops: p.compliant,
    intervened,
    material_at_risk: answers.material_at_risk!.noul!,
    captures_material: answers.captures_material!.noul!,
    inference_ms: inferenceMs,
    decision_ms: decisionMs,
    input_tokens: output.usage.input_tokens,
    output_tokens: output.usage.output_tokens ?? 0,
    compliant_count: p.compliant.length,
  };
}
