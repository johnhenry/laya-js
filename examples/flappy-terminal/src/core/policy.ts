/**
 * Real Laya predictions, with an explicit optional deterministic safety
 * shield -- the same shape as `snake-terminal/src/core/policy.ts`
 * (`buildPrompt`, `LayaPolicy`, `decisionFrom`), adapted to Flappy Bird's
 * two-action space instead of Snake's four directions.
 *
 * Backend-agnostic: pass anything with `predict(state, questions)` -- the
 * `@johnhenry/laya` agent on MLX, WebGPU or CPU.
 */
import type { Questions, State } from "@johnhenry/laya-core";
import { ACTIONS, type Action, type FlappyGame, type MoveInfo } from "./game.ts";

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
export type FlappyQuestions = {
  action: { type: "choice"; instructions: string; criteria: Record<Action, string> };
  risk: { type: "noul"; instructions: string };
  aligned: { type: "noul"; instructions: string };
};

export interface FlappyPrompt {
  state: string;
  questions: FlappyQuestions;
  moves: MoveInfo[];
  safe: MoveInfo[];
  preferred: Action | "NONE";
}

export interface Decision {
  probabilities: Record<Action, number>;
  proposed: Action;
  executed: Action;
  safe_actions: Action[];
  intervened: boolean;
  collision_risk: number;
  gap_aligned: number;
  inference_ms: number;
  decision_ms: number;
  input_tokens: number;
  output_tokens: number;
  safe_count: number;
  planner_best: Action | "NONE";
}

/** The gap center of the next not-yet-passed pipe, or mid-screen if none is tracked. */
function nextGapCenter(game: FlappyGame): number {
  const next = game.pipes.find((p) => !p.scored);
  return next ? next.gapY + game.gapHeight / 2 : game.height / 2;
}

/** Planner features -> the exact state text and questions sent to the agent. */
export function buildPrompt(game: FlappyGame, prompt: PromptStyle = "compact"): FlappyPrompt {
  const moves = game.moves();
  const safe = moves.filter((m) => m.safe);
  const target = nextGapCenter(game);
  const resultingY = (action: Action) => game.birdY + (action === "FLAP" ? game.flapImpulse : game.birdVy + game.gravity);
  let preferred: Action | "NONE" = "NONE";
  if (safe.length) {
    let best = safe[0]!;
    let bestDist = Math.abs(resultingY(best.action) - target);
    for (const m of safe) {
      const d = Math.abs(resultingY(m.action) - target);
      if (d < bestDist) {
        best = m;
        bestDist = d;
      }
    }
    preferred = best.action;
  }
  const next = game.pipes.find((p) => !p.scored);
  const gapDistance = next ? Math.max(0, Math.round(next.x - game.birdX)) : null;
  const criteria = {} as Record<Action, string>;
  let state: string;
  let action: string, risk: string, aligned: string;
  if (prompt === "compact") {
    state =
      `Bird at row ${game.birdY.toFixed(1)} of ${game.height}. ` +
      (next ? `Next gap ${gapDistance} columns away, centered at row ${target.toFixed(1)}. ` : "") +
      `${safe.length ? "A safe action exists." : "No action avoids a collision."}`;
    action = "Should the bird flap now?";
    risk = "Is the bird in danger of hitting the ground, ceiling or a pipe soon?";
    aligned = "Is the bird vertically aligned with the gap of the upcoming pipe?";
    for (const m of moves) {
      criteria[m.action] = !m.safe ? "Unsafe. Leads to a collision soon." : m.action === preferred ? "Safe. Best alignment with the gap. Best." : "Safe. Keeps the bird alive.";
    }
  } else if (prompt === "detailed") {
    state =
      `Flappy Bird. Bird row ${game.birdY.toFixed(2)} of ${game.height} (0 = ceiling, ${game.height - 1} = ground), ` +
      `vertical speed ${game.birdVy.toFixed(2)}. ` +
      (next ? `Upcoming pipe gap spans rows ${next.gapY}-${next.gapY + game.gapHeight - 1}, ${gapDistance} columns ahead. ` : "") +
      `${safe.length ? "A safe action exists for the next several ticks." : "Every action leads to a collision soon -- this may be unavoidable."}`;
    action = "Choose the action that best keeps the bird alive and aligned with the gap.";
    risk = "Is there real danger of a collision with the ground, ceiling, or a pipe soon?";
    aligned = "Is the bird's current row inside the vertical span of the upcoming pipe's gap?";
    for (const m of moves) {
      criteria[m.action] = !m.safe ? `Unsafe: ${m.reason}.` : m.action === preferred ? "Safe. Best progress toward the gap center. Best move." : "Safe, but less aligned with the gap center.";
    }
  } else {
    throw new RangeError("prompt must be compact or detailed");
  }
  return {
    state,
    questions: {
      action: { type: "choice", instructions: action, criteria },
      risk: { type: "noul", instructions: risk },
      aligned: { type: "noul", instructions: aligned },
    },
    moves,
    safe,
    preferred,
  };
}

export interface PolicyOptions {
  /** Restrict execution to lookahead-safe actions (default true). */
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

  async decide(game: FlappyGame): Promise<Decision> {
    const started = this.#now();
    const p = buildPrompt(game, this.prompt);
    // No invariant throw here, unlike Snake: Snake's Hamiltonian-cycle planner
    // guarantees a safe move always exists from a correctly-played state, so
    // an empty safe set there is a real bug. Flappy Bird has no such
    // guarantee -- stochastic pipe geometry and a bad flap history can make
    // death genuinely unavoidable -- so an empty safe set is a real,
    // expected outcome, handled in `decisionFrom` instead of thrown here.
    const inferenceStart = this.#now();
    const output = await this.agent.predict(p.state, p.questions);
    const inferenceMs = this.#now() - inferenceStart;
    return decisionFrom(output, p, this.guarded, inferenceMs, this.#now() - started);
  }
}

/** The shield + bookkeeping half of `decide` (pure; exported for tests and replays). */
export function decisionFrom(output: PredictLike, p: FlappyPrompt, guarded: boolean, inferenceMs = 0, decisionMs = 0): Decision {
  const answers = output.answers;
  const probabilities = answers.action!.probabilities as Record<Action, number>;
  const scores = [...Object.values(probabilities), answers.risk!.noul!, answers.aligned!.noul!];
  if (scores.some((v) => !Number.isFinite(v) || v < 0 || v > 1)) {
    throw new Error("Model returned an invalid probability; no action executed");
  }
  const argmax = (actions: readonly Action[]) => actions.reduce((best, a) => (probabilities[a] > probabilities[best] ? a : best), actions[0]!);
  const proposed = argmax(ACTIONS);
  const allowed = p.safe.map((m) => m.action);
  // Empty `allowed` (both actions unsafe within the lookahead): execute the
  // model's raw choice rather than overriding -- see the comment in
  // `LayaPolicy.decide` for why this differs from Snake's invariant throw.
  const executed = guarded && allowed.length > 0 && !allowed.includes(proposed) ? argmax(allowed) : proposed;
  return {
    probabilities,
    proposed,
    executed,
    safe_actions: allowed,
    intervened: proposed !== executed,
    collision_risk: answers.risk!.noul!,
    gap_aligned: answers.aligned!.noul!,
    inference_ms: inferenceMs,
    decision_ms: decisionMs,
    input_tokens: output.usage.input_tokens,
    output_tokens: output.usage.output_tokens ?? 0,
    safe_count: p.safe.length,
    planner_best: p.preferred,
  };
}
