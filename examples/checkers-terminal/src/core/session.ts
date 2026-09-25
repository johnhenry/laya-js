/**
 * The turn loop shared by the terminal and browser demos. Deliberately
 * chain-unaware (see `game.ts`'s header comment): every call to `tick()`
 * runs the identical decide -> applyHop cycle, for whichever actor
 * (`laya`/`bot`/`human`) is seated on `state.toMove`, and whether the same
 * player is asked again (a forced continuation) or the turn passes falls
 * entirely out of `GameSnapshot.forcedContinuation` persisting across
 * `applyHop` -- no imperative "if chain, loop again" branch exists here.
 */
import { newGame, type GameSnapshot } from "./game.ts";
import { applyHop } from "./game.ts";
import type { Hop } from "./moves.ts";
import type { Player } from "./board.ts";
import type { Actor } from "./actors.ts";
import type { Decision } from "./policy.ts";

export interface SessionStats {
  hardware: string;
  engine: string;
  interventions: number;
  round: number;
  paused: boolean;
  elapsed: number;
  turns_per_second: number;
  turns: number;
  red_wins: number;
  black_wins: number;
  inference_ms_total: number;
}

export interface TickResult {
  /** Board shown before this hop was applied. */
  board: GameSnapshot;
  hop: Hop;
  decision?: Decision;
  actor: Actor["kind"];
  /** Set when the hop ended the game (the session already started the next round). */
  roundEnd?: GameSnapshot;
}

export class CheckersSession {
  readonly actors: Record<Player, Actor>;
  game: GameSnapshot;
  stats: SessionStats;
  /** Start the next round (a fresh game) automatically after a win. */
  autoNextRound: boolean;
  #started: number;
  #stamps: number[] = [];
  readonly #now: () => number;

  constructor(actors: Record<Player, Actor>, options: { hardware?: string; engine?: string; now?: () => number; autoNextRound?: boolean } = {}) {
    this.actors = actors;
    this.#now = options.now ?? (() => performance.now());
    this.game = newGame();
    this.autoNextRound = options.autoNextRound ?? true;
    this.#started = this.#now();
    this.stats = {
      hardware: options.hardware ?? "Local",
      engine: options.engine ?? "FP16",
      interventions: 0,
      round: 1,
      paused: false,
      elapsed: 0,
      turns_per_second: 0,
      turns: 0,
      red_wins: 0,
      black_wins: 0,
      inference_ms_total: 0,
    };
  }

  reset(): void {
    this.stats.round += 1;
    this.game = newGame();
  }

  restartClock(): void {
    this.#started = this.#now();
    this.#stamps = [];
  }

  get elapsedMs(): number {
    return this.#now() - this.#started;
  }

  async tick(): Promise<TickResult & { stop: boolean }> {
    const board = this.game;
    const actor = this.actors[this.game.toMove];
    const result = await actor.act(this.game);
    const s = this.stats;
    if (result.decision) {
      s.interventions += result.decision.intervened ? 1 : 0;
      s.inference_ms_total += result.decision.inference_ms;
    }
    const shown = this.#now();
    this.#stamps.push(shown);
    if (this.#stamps.length > 60) this.#stamps.shift();
    s.elapsed = (shown - this.#started) / 1000;
    const n = this.#stamps.length;
    s.turns_per_second = n > 1 ? ((n - 1) * 1000) / (this.#stamps[n - 1]! - this.#stamps[0]!) : 0;

    this.game = applyHop(this.game, result.hop);
    s.turns += 1;
    let roundEnd: GameSnapshot | undefined;
    let stop = false;
    if (this.game.status !== "in_progress") {
      roundEnd = this.game;
      if (this.game.status === "red_wins") s.red_wins += 1;
      else s.black_wins += 1;
      if (!this.autoNextRound) stop = true;
      else {
        s.round += 1;
        this.game = newGame();
      }
    }
    return { board, hop: result.hop, decision: result.decision, actor: actor.kind, roundEnd, stop };
  }
}
