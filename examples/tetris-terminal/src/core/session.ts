/**
 * The play loop shared by the terminal and browser demos: one real
 * prediction per PIECE (not per tick), shield bookkeeping, rolling
 * decision rate, automatic next round (seed + round - 1) after a
 * block-out. Same shape as snake-terminal's/flappy-terminal's `session.ts`.
 */
import { TetrisGame, type GameSnapshot } from "./game.ts";
import type { Decision, LayaPolicy } from "./policy.ts";

export interface SessionStats {
  hardware: string;
  engine: string;
  guarded: boolean;
  interventions: number;
  best: number;
  round: number;
  paused: boolean;
  elapsed: number;
  pieces_per_second: number;
  pieces: number;
  lines: number;
  deaths: number;
  inference_ms_total: number;
}

export interface TickResult {
  /** Board shown with `decision` (before the announced placement). */
  board: GameSnapshot;
  decision: Decision;
  cleared: number;
  /** Set when the piece ended the round (the session already started the next one). */
  roundEnd?: GameSnapshot;
}

export class TetrisSession {
  readonly policy: LayaPolicy;
  readonly seed: number;
  game: TetrisGame;
  stats: SessionStats;
  /** Start the next round after a block-out (Python-style demos stop in --unassisted mode; mirrored here too). */
  autoNextRound: boolean;
  #started: number;
  #stamps: number[] = [];
  readonly #now: () => number;

  constructor(policy: LayaPolicy, seed = 7, options: { hardware?: string; engine?: string; now?: () => number; autoNextRound?: boolean } = {}) {
    this.policy = policy;
    this.seed = seed;
    this.#now = options.now ?? (() => performance.now());
    this.game = new TetrisGame(seed);
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
      pieces_per_second: 0,
      pieces: 0,
      lines: 0,
      deaths: 0,
      inference_ms_total: 0,
    };
  }

  /** R key: start the next round with the next seed. */
  reset(): void {
    this.stats.round += 1;
    this.game = new TetrisGame(this.seed + this.stats.round - 1);
  }

  restartClock(): void {
    this.#started = this.#now();
    this.#stamps = [];
  }

  get elapsedMs(): number {
    return this.#now() - this.#started;
  }

  /** Decide on the current piece. Call `advance` with the result to apply it. */
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
    s.pieces_per_second = n > 1 ? ((n - 1) * 1000) / (this.#stamps[n - 1]! - this.#stamps[0]!) : 0;
    s.best = Math.max(s.best, this.game.score);
    return { board, decision };
  }

  advance(decision: Decision): { cleared: number; roundEnd?: GameSnapshot; stop: boolean } {
    const cleared = this.game.applyPlacement(decision.executed);
    const s = this.stats;
    s.pieces += 1;
    s.lines += cleared;
    s.best = Math.max(s.best, this.game.score);
    if (!this.game.alive) {
      const roundEnd = this.game.snapshot();
      s.deaths += 1;
      if (!this.autoNextRound) return { cleared, roundEnd, stop: true };
      s.round += 1;
      this.game = new TetrisGame(this.seed + s.round - 1);
      return { cleared, roundEnd, stop: false };
    }
    return { cleared, stop: false };
  }

  async tick(): Promise<TickResult & { stop: boolean }> {
    const { board, decision } = await this.decide();
    const r = this.advance(decision);
    return { board, decision, ...r };
  }
}
