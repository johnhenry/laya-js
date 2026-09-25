/**
 * The play loop shared by the terminal and browser demos: one real
 * prediction per tick, shield bookkeeping, rolling decision rate, automatic
 * next round (seed + round - 1) after a death. Same shape as
 * `snake-terminal/src/core/session.ts`.
 */
import { FlappyGame, type GameSnapshot } from "./game.ts";
import type { Decision, LayaPolicy } from "./policy.ts";

export interface BoardOptions {
  width?: number;
  height?: number;
  seed?: number;
  gravity?: number;
  flapImpulse?: number;
  pipeSpeed?: number;
  gapHeight?: number;
  pipeWidth?: number;
  pipeSpacing?: number;
}

export interface SessionStats {
  hardware: string;
  engine: string;
  guarded: boolean;
  interventions: number;
  best: number;
  round: number;
  paused: boolean;
  elapsed: number;
  steps_per_second: number;
  steps: number;
  deaths: number;
  inference_ms_total: number;
}

export interface TickResult {
  /** Board shown with `decision` (before the announced action). */
  board: GameSnapshot;
  decision: Decision;
  scored: boolean;
  /** Set when the tick ended the round (the session already started the next one). */
  roundEnd?: GameSnapshot;
}

export class FlappySession {
  readonly policy: LayaPolicy;
  readonly board: Required<BoardOptions>;
  game: FlappyGame;
  stats: SessionStats;
  /** Start the next round after a death (Python-style demos stop in --unassisted mode; mirrored here too). */
  autoNextRound: boolean;
  #started: number;
  #stamps: number[] = [];
  readonly #now: () => number;

  constructor(
    policy: LayaPolicy,
    board: BoardOptions = {},
    options: { hardware?: string; engine?: string; now?: () => number; autoNextRound?: boolean } = {},
  ) {
    this.policy = policy;
    this.board = {
      width: 40,
      height: 20,
      seed: 7,
      gravity: 0.06,
      flapImpulse: -0.9,
      pipeSpeed: 0.5,
      gapHeight: 6,
      pipeWidth: 2,
      pipeSpacing: 14,
      ...board,
    };
    this.#now = options.now ?? (() => performance.now());
    this.game = this.#newGame(1);
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
      steps: 0,
      deaths: 0,
      inference_ms_total: 0,
    };
  }

  #newGame(round: number): FlappyGame {
    const b = this.board;
    return new FlappyGame({ ...b, seed: b.seed + round - 1 });
  }

  /** R key: start the next round with the next seed. */
  reset(): void {
    this.stats.round += 1;
    this.game = this.#newGame(this.stats.round);
  }

  /** Restart the clock (e.g. after warmup). */
  restartClock(): void {
    this.#started = this.#now();
    this.#stamps = [];
  }

  get elapsedMs(): number {
    return this.#now() - this.#started;
  }

  /** Decide on the current board. Call `advance` with the result to move. */
  async decide(): Promise<{ board: GameSnapshot; decision: Decision }> {
    const board = this.game.snapshot();
    const decision = await this.policy.decide(this.game);
    const s = this.stats;
    s.interventions += decision.intervened ? 1 : 0;
    s.inference_ms_total += decision.inference_ms;
    const shown = this.#now();
    this.#stamps.push(shown);
    if (this.#stamps.length > 60) this.#stamps.shift();
    s.elapsed = (shown - this.#started) / 1000;
    const n = this.#stamps.length;
    s.steps_per_second = n > 1 ? ((n - 1) * 1000) / (this.#stamps[n - 1]! - this.#stamps[0]!) : 0;
    s.best = Math.max(s.best, this.game.score);
    return { board, decision };
  }

  advance(decision: Decision): { scored: boolean; roundEnd?: GameSnapshot; stop: boolean } {
    const scored = this.game.step(decision.executed);
    const s = this.stats;
    s.steps += 1;
    s.best = Math.max(s.best, this.game.score);
    if (!this.game.alive) {
      const roundEnd = this.game.snapshot();
      s.deaths += 1;
      if (!this.autoNextRound) return { scored, roundEnd, stop: true };
      s.round += 1;
      this.game = this.#newGame(s.round);
      return { scored, roundEnd, stop: false };
    }
    return { scored, stop: false };
  }

  async tick(): Promise<TickResult & { stop: boolean }> {
    const { board, decision } = await this.decide();
    const r = this.advance(decision);
    return { board, decision, ...r };
  }
}
