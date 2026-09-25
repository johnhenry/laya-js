/**
 * Real Laya predictions, with an explicit optional deterministic safety
 * shield -- same shape as the other three tick-based games' `policy.ts`.
 * Richer than Flappy Bird's binary FLAP/NOFLAP: three actions here
 * (JUMP/DUCK/RUN), closer to Snake's fixed 4-way choice.
 */
import type { Questions, State } from "@johnhenry/laya-core";
import { ACTIONS, type Action, type DinoGame, type MoveInfo } from "./game.ts";

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
export type DinoQuestions = {
  action: { type: "choice"; instructions: string; criteria: Record<Action, string> };
  risk: { type: "noul"; instructions: string };
  low_obstacle: { type: "noul"; instructions: string };
};

export interface DinoPrompt {
  state: string;
  questions: DinoQuestions;
  moves: MoveInfo[];
  safe: MoveInfo[];
}

export interface Decision {
  probabilities: Record<Action, number>;
  proposed: Action;
  executed: Action;
  safe_actions: Action[];
  intervened: boolean;
  risk: number;
  low_obstacle: number;
  inference_ms: number;
  decision_ms: number;
  input_tokens: number;
  output_tokens: number;
  safe_count: number;
}

/** Planner features -> the exact state text and questions sent to the agent. */
export function buildPrompt(game: DinoGame, prompt: PromptStyle = "compact"): DinoPrompt {
  const moves = game.moves();
  const safe = moves.filter((m) => m.safe);
  const next = game.obstacles.find((o) => !o.scored);
  const distance = next ? Math.max(0, Math.round(next.x - game.dinoX)) : null;
  const criteria = {} as Record<Action, string>;
  let state: string;
  let action: string, risk: string, lowObstacle: string;
  if (prompt === "compact") {
    state =
      `Running. Speed ${game.speed.toFixed(2)}. ` +
      (next ? `Next obstacle: ${next.kind}, ${distance} units ahead. ` : "") +
      `${safe.length ? "A safe action exists." : "No action avoids a collision."}`;
    action = "Should the dino jump, duck, or keep running?";
    risk = "Is there danger of colliding with an obstacle soon without the right action?";
    lowObstacle = "Is the nearest obstacle a low one requiring a duck rather than a jump?";
    for (const m of moves) {
      criteria[m.action] = !m.safe ? "Unsafe. Leads to a collision soon." : "Safe.";
    }
  } else if (prompt === "detailed") {
    state =
      `Chrome dino runner. Speed ${game.speed.toFixed(2)}, score ${game.score}. ` +
      (next ? `Next obstacle is a ${next.kind}, ${distance} units ahead (jump clears a cactus, duck clears a low pterodactyl, a high pterodactyl is only dangerous if you jump into it). ` : "No obstacle is close. ") +
      `${safe.length ? "A safe action exists for the next several ticks." : "Every action leads to a collision soon -- this may be unavoidable."}`;
    action = "Choose the action that correctly dodges the nearest obstacle, or keep running if none is close.";
    risk = "Is there real danger of a collision soon without taking the right action?";
    lowObstacle = "Is the nearest obstacle specifically a low-flying pterodactyl (needs a duck, not a jump)?";
    for (const m of moves) {
      criteria[m.action] = !m.safe ? `Unsafe: ${m.reason}.` : "Safe.";
    }
  } else {
    throw new RangeError("prompt must be compact or detailed");
  }
  return {
    state,
    questions: {
      action: { type: "choice", instructions: action, criteria },
      risk: { type: "noul", instructions: risk },
      low_obstacle: { type: "noul", instructions: lowObstacle },
    },
    moves,
    safe,
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

  /** Returns null while airborne: no decision is meaningful mid-jump, so no prediction is requested (see game.ts). */
  async decide(game: DinoGame): Promise<Decision | null> {
    if (game.airborne) return null;
    const started = this.#now();
    const p = buildPrompt(game, this.prompt);
    const inferenceStart = this.#now();
    const output = await this.agent.predict(p.state, p.questions);
    const inferenceMs = this.#now() - inferenceStart;
    return decisionFrom(output, p, this.guarded, inferenceMs, this.#now() - started);
  }
}

/** The shield + bookkeeping half of `decide` (pure; exported for tests and replays). */
export function decisionFrom(output: PredictLike, p: DinoPrompt, guarded: boolean, inferenceMs = 0, decisionMs = 0): Decision {
  const answers = output.answers;
  const probabilities = answers.action!.probabilities as Record<Action, number>;
  const scores = [...Object.values(probabilities), answers.risk!.noul!, answers.low_obstacle!.noul!];
  if (scores.some((v) => !Number.isFinite(v) || v < 0 || v > 1)) {
    throw new Error("Model returned an invalid probability; no action executed");
  }
  const argmax = (actions: readonly Action[]) => actions.reduce((best, a) => (probabilities[a] > probabilities[best] ? a : best), actions[0]!);
  const proposed = argmax(ACTIONS);
  const allowed = p.safe.map((m) => m.action);
  // Empty `allowed` (every action collides within the lookahead): execute the model's raw
  // choice rather than overriding -- same precedent as Flappy Bird, for the same reason: a
  // badly timed obstacle pair can make every action genuinely unsafe, with no total-safety
  // guarantee to fall back on.
  const executed = guarded && allowed.length > 0 && !allowed.includes(proposed) ? argmax(allowed) : proposed;
  return {
    probabilities,
    proposed,
    executed,
    safe_actions: allowed,
    intervened: proposed !== executed,
    risk: answers.risk!.noul!,
    low_obstacle: answers.low_obstacle!.noul!,
    inference_ms: inferenceMs,
    decision_ms: decisionMs,
    input_tokens: output.usage.input_tokens,
    output_tokens: output.usage.output_tokens ?? 0,
    safe_count: allowed.length,
  };
}
