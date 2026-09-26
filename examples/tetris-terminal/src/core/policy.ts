/**
 * Real Laya predictions, with an explicit optional deterministic safety
 * shield -- same shape as the other three games' `policy.ts`, but the
 * decision itself is now per gravity STEP, not per piece: at each step the
 * model picks a rotation (0/90/180/270 degrees), a horizontal direction
 * (none/left/right) and a distance (0-9 columns, clamped to whatever's
 * free) -- three small independent `choice` questions, not one joint
 * lookup into a placement table. When the piece can no longer descend, it
 * gets exactly one more such decision (now also asking `risk`/`clears`,
 * the same two signals the old per-piece design always asked) before that
 * position locks -- a bounded "lock delay" / "extended placement" (see
 * Tetris Wiki / Hard Drop wiki), not an infinitely-resettable timer.
 *
 * The shield only ever guards this final lock decision, and its evaluation
 * set (`optionsAtRow`, game.ts) is LOCAL: everything reachable via
 * rotate+shift alone from the piece's current position, not a global
 * re-search from spawn. That's a real, weaker guarantee than the old
 * per-piece shield's -- it can fix a bad final orientation/column, not
 * rescue a trajectory that steered too high already -- and it's the honest
 * cost of "you get real per-step control," not a bug.
 */
import type { Questions, State } from "@johnhenry/laya-core";
import { resolveRotation, shapeOf, type Board, type PieceKind, type RotationLabel } from "./pieces.ts";
import { collides, optionsAtRow, stackHeight, type Placement, type PlacementInfo, type TetrisGame } from "./game.ts";

export const DEFAULT_MODEL = "aac6fef/laya-multilingual-mlx";

export type PromptStyle = "compact" | "detailed";
export type Direction = "none" | "left" | "right";
const DIRECTIONS: readonly Direction[] = ["none", "left", "right"];
const ROTATION_LABELS: readonly RotationLabel[] = ["0", "R", "2", "L"];
const ROTATION_DEGREES: Record<RotationLabel, string> = { "0": "0°", R: "90° clockwise", "2": "180°", L: "270° clockwise (90° counter-clockwise)" };
const MAX_DISTANCE = 9;

/** The subset of a `@johnhenry/laya` agent the policy needs. */
export interface PredictAgent {
  predict(state: State, questions: Questions): Promise<PredictLike> | PredictLike;
}

export interface PredictLike {
  answers: Record<string, { probabilities?: Record<string, number>; noul?: number }>;
  usage: { input_tokens: number; output_tokens?: number };
}

/** Where the active piece is right now: not yet locked, just in flight. */
export interface StepPosition {
  row: number;
  rotation: RotationLabel;
  col: number;
}

// Type aliases (not interfaces) so they are assignable to laya-core's `Questions`.
export type StepQuestions = {
  rotation: { type: "choice"; instructions: string; criteria: Record<string, string> };
  direction: { type: "choice"; instructions: string; criteria: Record<string, string> };
  distance: { type: "choice"; instructions: string; criteria: Record<string, string> };
};
export type LockQuestions = StepQuestions & {
  risk: { type: "noul"; instructions: string };
  clears: { type: "noul"; instructions: string };
};

export interface StepPrompt {
  state: string;
  questions: StepQuestions | LockQuestions;
  kind: PieceKind;
  position: StepPosition;
  isLockChance: boolean;
}

export interface StepDecision {
  /** The resolved position after applying this step's rotation+shift (same row unless it then descends). */
  position: StepPosition;
  /** Whether one more row of gravity is still possible from `position` -- tells the caller whether the NEXT decision is a normal step or the lock chance. */
  canDescend: boolean;
  /** True only on the lock-chance step: the piece has been locked (and cleared) into the board already. */
  locked: boolean;
  /** The model's own choice before the shield, set only when locked. */
  proposed?: Placement;
  executed?: Placement;
  intervened?: boolean;
  safe_count?: number;
  risk?: number;
  clears_signal?: number;
  rotation_probabilities: Record<string, number>;
  direction_probabilities: Record<string, number>;
  distance_probabilities: Record<string, number>;
  inference_ms: number;
  decision_ms: number;
  input_tokens: number;
  output_tokens: number;
}

/** Planner features -> the exact state text and questions sent to the agent for one step. */
export function buildStepPrompt(game: TetrisGame, position: StepPosition, isLockChance: boolean, prompt: PromptStyle = "compact"): StepPrompt {
  const kind = game.active;
  let state: string;
  if (prompt === "compact") {
    state =
      `Piece ${kind}, row ${position.row} of ${game.board.length}, rotation ${position.rotation}, column ${position.col}. Stack height ${stackHeight(game.board)}.` +
      (isLockChance ? " It cannot move down further -- this is the last chance to adjust before it locks." : " Choose how it moves this step.");
  } else if (prompt === "detailed") {
    const next3 = game.queue.slice(0, 3).join(", ");
    state =
      `Tetris. Active piece ${kind} (next up: ${next3}), row ${position.row} of ${game.board.length}, rotation ${position.rotation}, column ${position.col}. ` +
      `Stack height ${stackHeight(game.board)}. Level ${game.level}, ${game.linesCleared} lines cleared, score ${game.score}.` +
      (isLockChance
        ? " It cannot descend further -- this is the last chance to rotate or shift it sideways before it locks in place."
        : " Choose the rotation, horizontal direction and distance for this step.");
  } else {
    throw new RangeError("prompt must be compact or detailed");
  }
  const questions: StepQuestions = {
    rotation: { type: "choice", instructions: "Which rotation should the piece be in?", criteria: Object.fromEntries(ROTATION_LABELS.map((r) => [r, ROTATION_DEGREES[r]])) },
    direction: { type: "choice", instructions: "Which horizontal direction should it move this step, if any?", criteria: { none: "stay in this column", left: "move left", right: "move right" } },
    distance: {
      type: "choice",
      instructions: "How many columns to move in that direction this step (0 if none)? Clamped to whatever is actually free.",
      criteria: Object.fromEntries(Array.from({ length: MAX_DISTANCE + 1 }, (_, n) => [String(n), String(n)])),
    },
  };
  if (!isLockChance) return { state, questions, kind, position, isLockChance };
  return {
    state,
    questions: {
      ...questions,
      risk: { type: "noul", instructions: "On a scale of 0 (safe) to 1 (critical), how close would locking here leave the stack to topping out?" },
      clears: { type: "noul", instructions: "Would locking here clear at least one line?" },
    },
    kind,
    position,
    isLockChance,
  };
}

export interface PolicyOptions {
  /** Restrict the final lock to placements within the safety margin, when a safe local alternative exists (default true). */
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

  /** Decide one gravity step for the active piece, currently at `position`. `isLockChance` must be true exactly when the piece cannot descend from `position`. */
  async decideStep(game: TetrisGame, position: StepPosition, isLockChance: boolean): Promise<StepDecision> {
    const started = this.#now();
    const p = buildStepPrompt(game, position, isLockChance, this.prompt);
    const inferenceStart = this.#now();
    const output = await this.agent.predict(p.state, p.questions);
    const inferenceMs = this.#now() - inferenceStart;
    return decisionFromStep(output, p, game.board, this.guarded, inferenceMs, this.#now() - started);
  }
}

function argmaxKey(probabilities: Record<string, number>): string {
  let best: string | undefined;
  for (const [k, v] of Object.entries(probabilities)) {
    if (best === undefined || v > probabilities[best]!) best = k;
  }
  if (best === undefined) throw new Error("No probabilities to choose from");
  return best;
}

/** Prefers clearing more lines, then a lower resulting stack, then the option closest to what the model itself proposed. */
function bestLocalOption(options: readonly PlacementInfo[], nearCol: number): PlacementInfo {
  return [...options].sort((a, b) => b.clears - a.clears || a.heightAfter - b.heightAfter || Math.abs(a.placement.col - nearCol) - Math.abs(b.placement.col - nearCol))[0]!;
}

/** The shield + bookkeeping half of `decideStep` (pure; exported for tests and replays). */
export function decisionFromStep(output: PredictLike, p: StepPrompt, board: Board, guarded: boolean, inferenceMs = 0, decisionMs = 0): StepDecision {
  const { kind, position, isLockChance } = p;
  const answers = output.answers;
  const rotationProbs = answers.rotation!.probabilities as Record<string, number>;
  const directionProbs = answers.direction!.probabilities as Record<string, number>;
  const distanceProbs = answers.distance!.probabilities as Record<string, number>;
  const checks = [...Object.values(rotationProbs), ...Object.values(directionProbs), ...Object.values(distanceProbs)];
  if (isLockChance) checks.push(answers.risk!.noul!, answers.clears!.noul!);
  if (checks.some((v) => !Number.isFinite(v) || v < 0 || v > 1)) {
    throw new Error("Model returned an invalid probability; no step executed");
  }

  // Resolve rotation: no wall-kick attempt -- illegal in place at the CURRENT column keeps the current rotation instead.
  const chosenRotation = resolveRotation(kind, argmaxKey(rotationProbs) as RotationLabel);
  const rotation = shapeFits(board, kind, chosenRotation, position.row, position.col) ? chosenRotation : position.rotation;

  // Apply the lateral shift, clamped to whatever's actually free.
  const direction = argmaxKey(directionProbs) as Direction;
  const distance = Number(argmaxKey(distanceProbs));
  const step = direction === "left" ? -1 : direction === "right" ? 1 : 0;
  let col = position.col;
  if (step !== 0) {
    for (let i = 0; i < distance; i++) {
      if (!shapeFits(board, kind, rotation, position.row, col + step)) break;
      col += step;
    }
  }

  const canDescend = shapeFits(board, kind, rotation, position.row + 1, col);
  const resolved: StepPosition = { row: position.row, rotation, col };

  if (!isLockChance) {
    return {
      position: resolved,
      canDescend,
      locked: false,
      rotation_probabilities: rotationProbs,
      direction_probabilities: directionProbs,
      distance_probabilities: distanceProbs,
      inference_ms: inferenceMs,
      decision_ms: decisionMs,
      input_tokens: output.usage.input_tokens,
      output_tokens: output.usage.output_tokens ?? 0,
    };
  }

  const proposed: Placement = { kind, rotation, col, restRow: position.row };
  const local = optionsAtRow(board, kind, position.row, col);
  const proposedInfo = local.find((o) => o.placement.rotation === rotation && o.placement.col === col);
  const safeLocal = local.filter((o) => o.safe);
  // Empty `safeLocal` is routine near the top of a real game, not rare-and-terminal like
  // Snake's shield treats it -- trust the model's own choice rather than substituting a
  // different unshielded one it didn't ask for (same reasoning the old per-piece shield
  // used for its own empty-safe-set case).
  const executeOverride = guarded && safeLocal.length > 0 && !(proposedInfo?.safe ?? false);
  const executed = executeOverride ? bestLocalOption(safeLocal, col).placement : proposed;
  return {
    position: resolved,
    canDescend,
    locked: true,
    proposed,
    executed,
    intervened: executed !== proposed,
    safe_count: safeLocal.length,
    risk: answers.risk!.noul!,
    clears_signal: answers.clears!.noul!,
    rotation_probabilities: rotationProbs,
    direction_probabilities: directionProbs,
    distance_probabilities: distanceProbs,
    inference_ms: inferenceMs,
    decision_ms: decisionMs,
    input_tokens: output.usage.input_tokens,
    output_tokens: output.usage.output_tokens ?? 0,
  };
}

function shapeFits(board: Board, kind: PieceKind, rotation: RotationLabel, row: number, col: number): boolean {
  return !collides(board, shapeOf(kind, rotation), row, col);
}
