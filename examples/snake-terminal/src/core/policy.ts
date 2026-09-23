/**
 * Real Laya predictions, with an explicit optional deterministic safety shield.
 * Port of laya-mlx `laya_mlx/snake/policy.py` (`LayaPolicy.decide`): the same
 * state text, the same three questions (`move` choice + `risk` / `food` noul),
 * compact and detailed prompts, and the same shield rule.
 *
 * Backend-agnostic: pass anything with `predict(state, questions)` — the
 * `@johnhenry/laya` agent on MLX, WebGPU or CPU.
 */
import type { Questions, State } from "@johnhenry/laya-core";
import { DIRECTIONS, type Direction, type MoveInfo, type SnakeGame } from "./game.ts";

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
export type SnakeQuestions = {
  move: { type: "choice"; instructions: string; criteria: Record<Direction, string> };
  risk: { type: "noul"; instructions: string };
  food: { type: "noul"; instructions: string };
};

export interface SnakePrompt {
  state: string;
  questions: SnakeQuestions;
  moves: MoveInfo[];
  safe: MoveInfo[];
  preferred: Direction | "NONE";
  reachable: boolean;
  space: number;
}

export interface Decision {
  probabilities: Record<Direction, number>;
  proposed: Direction;
  executed: Direction;
  safe_directions: Direction[];
  intervened: boolean;
  dead_end_risk: number;
  food_reachable: number;
  inference_ms: number;
  decision_ms: number;
  input_tokens: number;
  output_tokens: number;
  safe_count: number;
  planner_best: Direction | "NONE";
}

/** Planner features -> the exact state text and questions Python sends. */
export function buildPrompt(game: SnakeGame, prompt: PromptStyle = "compact"): SnakePrompt {
  const moves = game.moves();
  const safe = moves.filter((m) => m.safe);
  let preferred: Direction | "NONE" = "NONE";
  if (safe.length) {
    let best = safe[0]!;
    for (const m of safe) if (m.advance > best.advance) best = m; // Python max(): first wins ties
    preferred = best.direction;
  }
  const [reachable, space] = game.foodReachability();
  const yn = (b: boolean) => (b ? "yes" : "no");
  const criteria = {} as Record<Direction, string>;
  let state: string;
  let move: string, risk: string, food: string;
  if (prompt === "compact") {
    state = `Safe route: ${yn(safe.length > 0)}. Food reachable through empty cells: ${yn(reachable)}.`;
    move = "Choose the best safe move toward food.";
    risk = "Is a safe route available?";
    food = "Is food reachable through empty cells?";
    for (const m of moves) {
      criteria[m.direction] = !m.legal
        ? "Blocked. Collision."
        : !m.safe
          ? "Unsafe. Traps the snake."
          : m.eats
            ? "Safe. Eat food now. Best."
            : m.direction === preferred
              ? "Safe. Best route to food."
              : "Safe. Slower route.";
    }
  } else if (prompt === "detailed") {
    state =
      `Snake game. ${safe.length} safe directions available. ` +
      `Food reachable through empty cells: ${yn(reachable)}. ` +
      `Open cells: ${space}. Snake length: ${game.body.length}. ` +
      `${safe.length ? "There is a safe route forward." : "The snake is trapped."}`;
    move = "Select the safest move with best progress toward food. Avoid collisions.";
    risk = "Is there a safe route forward for the snake?";
    food = "Is food reachable through the currently empty cells?";
    for (const m of moves) {
      criteria[m.direction] = !m.legal
        ? `Collision: ${m.reason}. Unsafe.`
        : !m.safe
          ? "Unsafe route. Risk of trapping the snake."
          : m.eats
            ? "Safe. Eat the food immediately. Best move."
            : m.direction === preferred
              ? "Safe. Best progress toward food."
              : "Safe but less progress toward food.";
    }
  } else {
    throw new RangeError("prompt must be compact or detailed");
  }
  return {
    state,
    questions: {
      move: { type: "choice", instructions: move, criteria },
      risk: { type: "noul", instructions: risk },
      food: { type: "noul", instructions: food },
    },
    moves,
    safe,
    preferred,
    reachable,
    space,
  };
}

export interface PolicyOptions {
  /** Restrict execution to cycle-safe moves (default true). */
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

  async decide(game: SnakeGame): Promise<Decision> {
    const started = this.#now();
    const p = buildPrompt(game, this.prompt);
    if (!p.safe.length && this.guarded) throw new Error("Cycle safety invariant violated: no safe action");
    const inferenceStart = this.#now();
    const output = await this.agent.predict(p.state, p.questions);
    const inferenceMs = this.#now() - inferenceStart;
    return decisionFrom(output, p, this.guarded, inferenceMs, this.#now() - started);
  }
}

/** The shield + bookkeeping half of `decide` (pure; exported for tests and replays). */
export function decisionFrom(
  output: PredictLike,
  p: SnakePrompt,
  guarded: boolean,
  inferenceMs = 0,
  decisionMs = 0,
): Decision {
  const answers = output.answers;
  const probabilities = answers.move!.probabilities as Record<Direction, number>;
  const scores = [...Object.values(probabilities), answers.risk!.noul!, answers.food!.noul!];
  if (scores.some((v) => !Number.isFinite(v) || v < 0 || v > 1)) {
    throw new Error("Model returned an invalid probability; no move executed");
  }
  const argmax = (dirs: readonly Direction[]) =>
    dirs.reduce((best, d) => (probabilities[d] > probabilities[best] ? d : best), dirs[0]!);
  const proposed = argmax(DIRECTIONS);
  const allowed = p.safe.map((m) => m.direction);
  const executed = guarded && !allowed.includes(proposed) ? argmax(allowed) : proposed;
  return {
    probabilities,
    proposed,
    executed,
    safe_directions: allowed,
    intervened: proposed !== executed,
    dead_end_risk: 1 - answers.risk!.noul!,
    food_reachable: answers.food!.noul!,
    inference_ms: inferenceMs,
    decision_ms: decisionMs,
    input_tokens: output.usage.input_tokens,
    output_tokens: output.usage.output_tokens ?? 0,
    safe_count: p.safe.length,
    planner_best: p.preferred,
  };
}
