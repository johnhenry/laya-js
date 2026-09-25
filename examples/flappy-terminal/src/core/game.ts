/**
 * Deterministic Flappy Bird rules and a lookahead safety classifier, in the
 * shape of `snake-terminal/src/core/game.ts`: a `step(action)` that decides
 * terminal states purely from game rules (no model involved), and a
 * `moves()` that tags every possible action `legal`/`safe` for a given tick
 * -- Checkers and Snake's `legal`/`safe` analog, here always exactly the
 * two actions `FLAP`/`NOFLAP` (both always legal; `safe` comes from a
 * multi-tick lookahead, not a single step, since one tick alone is almost
 * always survivable and wouldn't be a meaningful signal).
 *
 * This game is JS-original -- there is no Python reference to port from or
 * prove parity against (see `rng.ts`).
 */
import { FlappyRng } from "./rng.ts";

export const ACTIONS = ["FLAP", "NOFLAP"] as const;
export type Action = (typeof ACTIONS)[number];

export type DeathReason = "ground" | "ceiling" | "pipe";

export interface Pipe {
  /** Column center, decreasing over time; the gap is `[gapY, gapY + gapHeight)`. */
  x: number;
  gapY: number;
  scored: boolean;
}

export interface MoveInfo {
  action: Action;
  legal: boolean;
  safe: boolean;
  reason: string;
}

export interface GameSnapshot {
  width: number;
  height: number;
  seed: number;
  bird_x: number;
  bird_y: number;
  bird_vy: number;
  pipes: { x: number; gap_y: number; scored: boolean }[];
  score: number;
  ticks: number;
  alive: boolean;
  death_reason: DeathReason | null;
  gravity: number;
  flap_impulse: number;
  pipe_speed: number;
  gap_height: number;
  pipe_width: number;
  pipe_spacing: number;
}

export interface FlappyOptions {
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

/** How many ticks ahead `moves()` simulates to classify an action `safe`. */
const SAFETY_HORIZON_TICKS = 20;
/** How close to the ground (in rows) the lookahead's reactive coast starts flapping to arrest a fall. */
const REACT_MARGIN = 4;

function collidesAt(
  birdY: number,
  pipes: readonly { x: number; gapY: number }[],
  birdX: number,
  height: number,
  gapHeight: number,
  pipeWidth: number,
): DeathReason | null {
  if (birdY < 0) return "ceiling";
  if (birdY >= height - 1) return "ground";
  for (const p of pipes) {
    if (Math.abs(p.x - birdX) < pipeWidth / 2 + 0.5) {
      if (birdY < p.gapY || birdY >= p.gapY + gapHeight) return "pipe";
    }
  }
  return null;
}

export class FlappyGame {
  readonly width: number;
  readonly height: number;
  readonly seed: number;
  readonly birdX: number;
  readonly gravity: number;
  readonly flapImpulse: number;
  readonly pipeSpeed: number;
  readonly gapHeight: number;
  readonly pipeWidth: number;
  readonly pipeSpacing: number;
  readonly #minGapY: number;
  readonly #maxGapY: number;

  birdY: number;
  birdVy = 0;
  pipes: Pipe[] = [];
  score = 0;
  ticks = 0;
  alive = true;
  deathReason: DeathReason | null = null;
  readonly #rng: FlappyRng;

  constructor(options: FlappyOptions = {}) {
    this.width = options.width ?? 40;
    this.height = options.height ?? 20;
    this.seed = options.seed ?? 7;
    this.gravity = options.gravity ?? 0.06;
    this.flapImpulse = options.flapImpulse ?? -0.9;
    this.pipeSpeed = options.pipeSpeed ?? 0.5;
    this.gapHeight = options.gapHeight ?? 6;
    this.pipeWidth = options.pipeWidth ?? 2;
    this.pipeSpacing = options.pipeSpacing ?? 14;
    this.#minGapY = 1;
    this.#maxGapY = this.height - 2 - this.gapHeight;
    if (this.#maxGapY < this.#minGapY) {
      throw new RangeError("Board too short for the requested gap height (need height >= gapHeight + 3)");
    }
    this.birdX = Math.floor(this.width / 4);
    this.birdY = this.height / 2;
    this.#rng = new FlappyRng(this.seed);
    this.#ensurePipes();
  }

  /** Rebuild a game from a recorded snapshot (the pipe RNG restarts from `seed`, then replays spawns up to `ticks`). */
  static fromSnapshot(s: GameSnapshot): FlappyGame {
    const g = new FlappyGame({
      width: s.width,
      height: s.height,
      seed: s.seed,
      gravity: s.gravity,
      flapImpulse: s.flap_impulse,
      pipeSpeed: s.pipe_speed,
      gapHeight: s.gap_height,
      pipeWidth: s.pipe_width,
      pipeSpacing: s.pipe_spacing,
    });
    g.birdY = s.bird_y;
    g.birdVy = s.bird_vy;
    g.pipes = s.pipes.map((p) => ({ x: p.x, gapY: p.gap_y, scored: p.scored }));
    g.score = s.score;
    g.ticks = s.ticks;
    g.alive = s.alive;
    g.deathReason = s.death_reason;
    return g;
  }

  #spawnPipe(x: number): Pipe {
    return { x, gapY: this.#rng.int(this.#minGapY, this.#maxGapY), scored: false };
  }

  /** Keep at least one pipe beyond the visible+spacing boundary, so a bounded lookahead never needs a pipe that doesn't exist yet. */
  #ensurePipes(): void {
    while (this.pipes.length === 0 || this.pipes[this.pipes.length - 1]!.x <= this.width + this.pipeSpacing) {
      const x = this.pipes.length === 0 ? this.width + this.pipeSpacing : this.pipes[this.pipes.length - 1]!.x + this.pipeSpacing;
      this.pipes.push(this.#spawnPipe(x));
    }
  }

  /**
   * Simulate `action` now, then a reactive coast for `horizon` ticks: true
   * if that path ever collides.
   *
   * The coast (every tick after the first) flaps only when close to the
   * ground and otherwise doesn't -- NOT a blind "never flap again" coast.
   * A pure never-flap-again coast would (almost) always eventually hit the
   * ground regardless of what `action` was, since nothing stops gravity,
   * which made `safe` degenerate to "false for NOFLAP, basically always"
   * and caused the shield to flap on nearly every tick, overshooting into
   * the ceiling. The reactive coast models "assuming reasonable corrective
   * play continues", the same spirit as Snake's planner assuming the
   * player keeps following the safe cycle rather than assuming they stop
   * steering entirely.
   */
  #willCollideWithin(action: Action, horizon: number): boolean {
    let y = this.birdY;
    let vy = this.birdVy;
    let pipes = this.pipes.map((p) => ({ x: p.x, gapY: p.gapY }));
    for (let t = 0; t < horizon; t++) {
      const a = t === 0 ? action : y >= this.height - REACT_MARGIN ? "FLAP" : "NOFLAP";
      vy = a === "FLAP" ? this.flapImpulse : vy + this.gravity;
      y += vy;
      pipes = pipes.map((p) => ({ x: p.x - this.pipeSpeed, gapY: p.gapY }));
      if (collidesAt(y, pipes, this.birdX, this.height, this.gapHeight, this.pipeWidth)) return true;
    }
    return false;
  }

  /** Both actions are always legal (no illegal input in Flappy Bird); `safe` comes from the lookahead above. */
  moves(): MoveInfo[] {
    if (!this.alive) return [];
    return ACTIONS.map((action) => {
      const safe = !this.#willCollideWithin(action, SAFETY_HORIZON_TICKS);
      return { action, legal: true, safe, reason: safe ? "safe" : `would collide within ${SAFETY_HORIZON_TICKS} ticks` };
    });
  }

  /** Advance one tick; returns true when a pipe was passed (scored) this tick. */
  step(action: Action): boolean {
    if (!this.alive) throw new Error("Cannot step a finished game");
    if (!ACTIONS.includes(action)) throw new RangeError(`Unknown action: ${action}`);
    this.ticks += 1;
    this.birdVy = action === "FLAP" ? this.flapImpulse : this.birdVy + this.gravity;
    this.birdY += this.birdVy;
    for (const p of this.pipes) p.x -= this.pipeSpeed;
    this.#ensurePipes();
    this.pipes = this.pipes.filter((p) => p.x > this.birdX - this.pipeWidth - 5);
    let scored = false;
    for (const p of this.pipes) {
      if (!p.scored && p.x < this.birdX) {
        p.scored = true;
        this.score += 1;
        scored = true;
      }
    }
    const death = collidesAt(this.birdY, this.pipes, this.birdX, this.height, this.gapHeight, this.pipeWidth);
    if (death) {
      this.alive = false;
      this.deathReason = death;
    }
    return scored;
  }

  snapshot(): GameSnapshot {
    return {
      width: this.width,
      height: this.height,
      seed: this.seed,
      bird_x: this.birdX,
      bird_y: this.birdY,
      bird_vy: this.birdVy,
      pipes: this.pipes.map((p) => ({ x: p.x, gap_y: p.gapY, scored: p.scored })),
      score: this.score,
      ticks: this.ticks,
      alive: this.alive,
      death_reason: this.deathReason,
      gravity: this.gravity,
      flap_impulse: this.flapImpulse,
      pipe_speed: this.pipeSpeed,
      gap_height: this.gapHeight,
      pipe_width: this.pipeWidth,
      pipe_spacing: this.pipeSpacing,
    };
  }
}
