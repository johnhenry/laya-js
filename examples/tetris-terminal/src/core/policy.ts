/**
 * Real Laya predictions, with an explicit optional deterministic safety
 * shield -- same shape as the other three games' `policy.ts`. `criteria`
 * covers the FULL legal placement set (there's no separate "illegal" tier
 * to also show, unlike Snake/Checkers, since `legalPlacements` already IS
 * the complete reachable set). `decisionFrom`'s empty-safe-set handling
 * follows Flappy Bird's precedent, not Checkers' -- see the comment there.
 */
import type { Questions, State } from "@johnhenry/laya-core";
import { BOARD_HEIGHT } from "./pieces.ts";
import { placementKey, stackHeight, TOP_MARGIN, type Placement, type PlacementInfo, type TetrisGame } from "./game.ts";

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
export type TetrisQuestions = {
  move: { type: "choice"; instructions: string; criteria: Record<string, string> };
  risk: { type: "noul"; instructions: string };
  clears: { type: "noul"; instructions: string };
};

export interface TetrisPrompt {
  state: string;
  questions: TetrisQuestions;
  moves: PlacementInfo[];
  safe: PlacementInfo[];
}

export interface Decision {
  probabilities: Record<string, number>;
  proposed: Placement;
  executed: Placement;
  safe_placements: Placement[];
  intervened: boolean;
  risk: number;
  clears_signal: number;
  inference_ms: number;
  decision_ms: number;
  input_tokens: number;
  output_tokens: number;
  safe_count: number;
}

/** Planner features -> the exact state text and questions sent to the agent. */
export function buildPrompt(game: TetrisGame, prompt: PromptStyle = "compact"): TetrisPrompt {
  const moves = game.moves();
  const safe = moves.filter((m) => m.safe);
  const criteria: Record<string, string> = {};
  let state: string;
  let move: string;
  if (prompt === "compact") {
    state = `Piece ${game.active}. Stack height ${stackHeight(game.board)}. ${safe.length} of ${moves.length} placements are safe.`;
    move = "Choose the best placement for this piece.";
    for (const m of moves) {
      criteria[placementKey(m.placement)] = !m.safe
        ? `Unsafe: stacks to row ${BOARD_HEIGHT - m.heightAfter}, breaches the safety margin.`
        : m.clears === 4
          ? "Safe. Tetris! Clears 4 lines."
          : m.clears > 0
            ? `Safe. Clears ${m.clears} line(s).`
            : "Safe.";
    }
  } else if (prompt === "detailed") {
    const next3 = game.queue.slice(0, 3).join(", ");
    state =
      `Tetris. Active piece ${game.active}, next up: ${next3}. Stack height ${stackHeight(game.board)} of ${BOARD_HEIGHT}. ` +
      `Level ${game.level}, ${game.linesCleared} lines cleared, score ${game.score}. ` +
      `${safe.length} of ${moves.length} reachable placements stay within the ${TOP_MARGIN}-row safety margin.`;
    move = "Choose the placement that best balances clearing lines and staying below the safety margin.";
    for (const m of moves) {
      criteria[placementKey(m.placement)] = !m.safe
        ? `Unsafe: locking here leaves the stack at row ${BOARD_HEIGHT - m.heightAfter} from the top, breaching the ${TOP_MARGIN}-row safety margin.`
        : m.clears === 4
          ? "Safe. A Tetris -- clears all 4 lines. Best."
          : m.clears > 0
            ? `Safe, and clears ${m.clears} line(s). Reduces stack height.`
            : "Safe, but clears no lines.";
    }
  } else {
    throw new RangeError("prompt must be compact or detailed");
  }
  return {
    state,
    questions: {
      move: { type: "choice", instructions: move, criteria },
      risk: { type: "noul", instructions: "On a scale of 0 (safe) to 1 (critical), how close is the stack to topping out?" },
      clears: { type: "noul", instructions: "Does this placement clear at least one line?" },
    },
    moves,
    safe,
  };
}

export interface PolicyOptions {
  /** Restrict execution to placements within the safety margin (default true). */
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

  async decide(game: TetrisGame): Promise<Decision> {
    const started = this.#now();
    const p = buildPrompt(game, this.prompt);
    if (!p.moves.length) throw new Error("Cannot decide: the game is already over (no legal placements)");
    const inferenceStart = this.#now();
    const output = await this.agent.predict(p.state, p.questions);
    const inferenceMs = this.#now() - inferenceStart;
    return decisionFrom(output, p, this.guarded, inferenceMs, this.#now() - started);
  }
}

function argmaxPlacement(infos: readonly PlacementInfo[], probabilities: Record<string, number>): Placement {
  let best = infos[0]!;
  for (const m of infos) {
    if ((probabilities[placementKey(m.placement)] ?? -Infinity) > (probabilities[placementKey(best.placement)] ?? -Infinity)) best = m;
  }
  return best.placement;
}

/** The shield + bookkeeping half of `decide` (pure; exported for tests and replays). */
export function decisionFrom(output: PredictLike, p: TetrisPrompt, guarded: boolean, inferenceMs = 0, decisionMs = 0): Decision {
  const answers = output.answers;
  const probabilities = answers.move!.probabilities as Record<string, number>;
  const scores = [...Object.values(probabilities), answers.risk!.noul!, answers.clears!.noul!];
  if (scores.some((v) => !Number.isFinite(v) || v < 0 || v > 1)) {
    throw new Error("Model returned an invalid probability; no placement executed");
  }
  const proposed = argmaxPlacement(p.moves, probabilities);
  const allowed = p.safe;
  // Empty `allowed` here does NOT mean game over (unlike Checkers' compliant-empty,
  // which always coincides with the loss condition) -- every reachable placement can
  // breach the safety margin while the board still has open cells elsewhere. That's
  // "in real trouble," not "already lost," so execute the model's raw choice rather
  // than throwing -- the same precedent Flappy Bird's empty-safe-set case follows.
  const executed = guarded && allowed.length > 0 && !allowed.some((m) => placementKey(m.placement) === placementKey(proposed)) ? argmaxPlacement(allowed, probabilities) : proposed;
  return {
    probabilities,
    proposed,
    executed,
    safe_placements: allowed.map((m) => m.placement),
    intervened: placementKey(proposed) !== placementKey(executed),
    risk: answers.risk!.noul!,
    clears_signal: answers.clears!.noul!,
    inference_ms: inferenceMs,
    decision_ms: decisionMs,
    input_tokens: output.usage.input_tokens,
    output_tokens: output.usage.output_tokens ?? 0,
    safe_count: allowed.length,
  };
}
