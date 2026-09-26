/**
 * The play loop shared by the terminal and browser demos: one real
 * prediction per gravity STEP (not per piece -- see policy.ts), shield
 * bookkeeping, rolling step/piece rates, automatic next round (seed +
 * round - 1) after a block-out. Same shape as snake-terminal's/
 * flappy-terminal's `session.ts`, but the family's plain `decide()`/
 * `advance()` pair becomes `stepDecide()`/`stepAdvance()` here (called once
 * per step, not once per piece) plus a thin `pieceSteps()` generator for
 * callers that just want to iterate a whole piece's steps.
 */
import { spawnColFor } from "./pieces.ts";
import { SPAWN_ROW, TetrisGame, type GameSnapshot } from "./game.ts";
import type { LayaPolicy, StepDecision, StepPosition } from "./policy.ts";

export interface SessionStats {
  hardware: string;
  engine: string;
  guarded: boolean;
  interventions: number;
  best: number;
  round: number;
  paused: boolean;
  elapsed: number;
  /** Raw decisions/sec -- the new, much higher rate (one per gravity step). */
  steps_per_second: number;
  /** Pieces locked/sec -- what this field always meant, now naturally lower. */
  pieces_per_second: number;
  pieces: number;
  lines: number;
  deaths: number;
  inference_ms_total: number;
}

export interface StepAdvance {
  /** True once this step's decision has been locked (and cleared) into the game. */
  locked: boolean;
  cleared?: number;
  /** Set when the piece just locked ended the round (the session already started the next one, unless `stop`). */
  roundEnd?: GameSnapshot;
  stop?: boolean;
}

export interface PieceStepEvent {
  /** Board shown with this decision (before the announced position/placement). */
  board: GameSnapshot;
  /** Where the piece was when this decision was asked. */
  position: StepPosition;
  isLockChance: boolean;
  step: StepDecision;
  advance: StepAdvance;
}

export class TetrisSession {
  readonly policy: LayaPolicy;
  readonly seed: number;
  game: TetrisGame;
  stats: SessionStats;
  /** Where the active piece is right now (spawn position at the start of each piece). */
  position: StepPosition;
  /** Start the next round after a block-out (Python-style demos stop in --unassisted mode; mirrored here too). */
  autoNextRound: boolean;
  #lockChancePending = false;
  #started: number;
  #stepStamps: number[] = [];
  #pieceStamps: number[] = [];
  readonly #now: () => number;

  constructor(policy: LayaPolicy, seed = 7, options: { hardware?: string; engine?: string; now?: () => number; autoNextRound?: boolean } = {}) {
    this.policy = policy;
    this.seed = seed;
    this.#now = options.now ?? (() => performance.now());
    this.game = new TetrisGame(seed);
    this.position = this.#spawnPosition();
    this.autoNextRound = options.autoNextRound ?? policy.guarded;
    this.#started = this.#now();
    this.stats = {
      hardware: options.hardware ?? "Local",
      engine: options.engine ?? "FP16",
      guarded: policy.guarded,
      interventions: 0,
      best: 0,
      round: 1,
      paused: false,
      elapsed: 0,
      steps_per_second: 0,
      pieces_per_second: 0,
      pieces: 0,
      lines: 0,
      deaths: 0,
      inference_ms_total: 0,
    };
  }

  #spawnPosition(): StepPosition {
    return { row: SPAWN_ROW, rotation: "0", col: spawnColFor(this.game.active) };
  }

  /** R key: start the next round with the next seed. */
  reset(): void {
    this.stats.round += 1;
    this.game = new TetrisGame(this.seed + this.stats.round - 1);
    this.position = this.#spawnPosition();
    this.#lockChancePending = false;
  }

  restartClock(): void {
    this.#started = this.#now();
    this.#stepStamps = [];
    this.#pieceStamps = [];
  }

  get elapsedMs(): number {
    return this.#now() - this.#started;
  }

  /** Decide one gravity step for the active piece at its current position. Call `stepAdvance` with the result to commit it. */
  async stepDecide(): Promise<{ board: GameSnapshot; position: StepPosition; isLockChance: boolean; step: StepDecision }> {
    const board = this.game.snapshot();
    const position = this.position;
    const isLockChance = this.#lockChancePending;
    const step = await this.policy.decideStep(this.game, position, isLockChance);
    const s = this.stats;
    s.inference_ms_total += step.inference_ms;
    if (step.locked) s.interventions += step.intervened ? 1 : 0;
    const shown = this.#now();
    this.#stepStamps.push(shown);
    if (this.#stepStamps.length > 60) this.#stepStamps.shift();
    s.elapsed = (shown - this.#started) / 1000;
    const n = this.#stepStamps.length;
    s.steps_per_second = n > 1 ? ((n - 1) * 1000) / (this.#stepStamps[n - 1]! - this.#stepStamps[0]!) : 0;
    return { board, position, isLockChance, step };
  }

  /** Commits a step's result: advances the in-flight position, or on the lock-chance step, locks the placement into the game. */
  stepAdvance(result: { position: StepPosition; isLockChance: boolean; step: StepDecision }): StepAdvance {
    const { step } = result;
    if (!result.isLockChance) {
      this.position = step.canDescend ? { ...step.position, row: step.position.row + 1 } : step.position;
      this.#lockChancePending = !step.canDescend;
      return { locked: false };
    }
    const cleared = this.game.applyPlacement(step.executed!);
    const s = this.stats;
    s.pieces += 1;
    s.lines += cleared;
    s.best = Math.max(s.best, this.game.score);
    this.#pieceStamps.push(this.#now());
    if (this.#pieceStamps.length > 60) this.#pieceStamps.shift();
    const n = this.#pieceStamps.length;
    s.pieces_per_second = n > 1 ? ((n - 1) * 1000) / (this.#pieceStamps[n - 1]! - this.#pieceStamps[0]!) : 0;
    this.#lockChancePending = false;
    if (!this.game.alive) {
      const roundEnd = this.game.snapshot();
      s.deaths += 1;
      if (!this.autoNextRound) return { locked: true, cleared, roundEnd, stop: true };
      s.round += 1;
      this.game = new TetrisGame(this.seed + s.round - 1);
      this.position = this.#spawnPosition();
      return { locked: true, cleared, roundEnd, stop: false };
    }
    this.position = this.#spawnPosition();
    return { locked: true, cleared, stop: false };
  }

  /** Convenience: iterate one piece's steps (a full `for await` loop over one piece's real decisions), yielding after each is committed. */
  async *pieceSteps(): AsyncGenerator<PieceStepEvent, void> {
    while (true) {
      const result = await this.stepDecide();
      const advance = this.stepAdvance(result);
      yield { ...result, advance };
      if (advance.locked) return;
    }
  }
}
