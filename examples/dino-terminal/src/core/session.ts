/**
 * The play loop shared by the terminal and browser demos. Same shape as
 * flappy-terminal's/tetris-terminal's `session.ts`, with one addition:
 * while airborne, `decide()` returns a null `Decision` (no `predict()`
 * call is made -- see policy.ts) and `advance()` just continues the jump.
 */
import { DinoGame, type GameSnapshot } from "./game.ts";
import type { Decision, LayaPolicy } from "./policy.ts";

export interface BoardOptions {
  width?: number;
  seed?: number;
  baseSpeed?: number;
  maxSpeed?: number;
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
  /** Null on an airborne auto-advance tick -- no prediction was requested. */
  decision: Decision | null;
  scored: boolean;
  roundEnd?: GameSnapshot;
}

export class DinoSession {
  readonly policy: LayaPolicy;
  readonly board: Required<BoardOptions>;
  game: DinoGame;
  stats: SessionStats;
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
    this.board = { width: 60, seed: 7, baseSpeed: 0.6, maxSpeed: 1.6, ...board };
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

  #newGame(round: number): DinoGame {
    const b = this.board;
    return new DinoGame({ ...b, seed: b.seed + round - 1 });
  }

  reset(): void {
    this.stats.round += 1;
    this.game = this.#newGame(this.stats.round);
  }

  restartClock(): void {
    this.#started = this.#now();
    this.#stamps = [];
  }

  get elapsedMs(): number {
    return this.#now() - this.#started;
  }

  /** Decide on the current tick. Null while airborne -- no prediction is requested mid-jump. */
  async decide(): Promise<{ board: GameSnapshot; decision: Decision | null }> {
    const board = this.game.snapshot();
    if (this.game.airborne) return { board, decision: null };
    const decision = await this.policy.decide(this.game);
    const s = this.stats;
    s.interventions += decision?.intervened ? 1 : 0;
    s.inference_ms_total += decision?.inference_ms ?? 0;
    const shown = this.#now();
    this.#stamps.push(shown);
    if (this.#stamps.length > 60) this.#stamps.shift();
    s.elapsed = (shown - this.#started) / 1000;
    const n = this.#stamps.length;
    s.steps_per_second = n > 1 ? ((n - 1) * 1000) / (this.#stamps[n - 1]! - this.#stamps[0]!) : 0;
    s.best = Math.max(s.best, this.game.score);
    return { board, decision };
  }

  advance(decision: Decision | null): { scored: boolean; roundEnd?: GameSnapshot; stop: boolean } {
    const scored = this.game.step(decision?.executed ?? "RUN");
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
