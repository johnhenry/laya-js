/**
 * Deterministic Chrome-dino-runner rules and a lookahead safety classifier,
 * in the shape of `flappy-terminal/src/core/game.ts` -- tick-based,
 * fixed-cardinality action set, `moves()` tags each action `legal`/`safe`.
 * Three actions here instead of Flappy Bird's two: JUMP clears a cactus
 * but is fatal under a low-flying pterodactyl's height band; DUCK clears a
 * low pterodactyl but not a cactus; RUN is only safe when nothing is
 * underfoot. Once airborne, the jump arc is uncontrollable for its full
 * duration (matches the real game) -- no decision is requested on airborne
 * ticks at all, since no action has any effect until landing.
 *
 * The lookahead's "coast" phase (every tick after the one being tested) is
 * REACTIVE (jump for an imminent cactus, duck for an imminent low
 * pterodactyl, run otherwise) -- deliberately NOT a blind "always RUN"
 * coast. A blind coast has exactly the shape of bug flappy-terminal shipped
 * and then fixed: if the coast never reacts to anything, every path
 * eventually runs into *some* obstacle inside a long enough horizon,
 * making every action look permanently unsafe. See
 * flappy-terminal/src/core/game.ts's `#willCollideWithin` docstring for
 * the full story; this game is built reactive from the start instead of
 * relearning that the hard way.
 *
 * This game is JS-original -- there is no Python reference to port from or
 * prove parity against (see `rng.ts`).
 */
import { DinoRng } from "./rng.ts";

export const ACTIONS = ["JUMP", "DUCK", "RUN"] as const;
export type Action = (typeof ACTIONS)[number];

export type ObstacleKind = "cactus" | "pterodactyl-low" | "pterodactyl-high";
export type DeathReason = "cactus" | "pterodactyl";

export interface Obstacle {
  x: number;
  kind: ObstacleKind;
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
  dino_x: number;
  tick: number;
  airborne_ticks_left: number;
  ducking: boolean;
  obstacles: { x: number; kind: ObstacleKind; scored: boolean }[];
  score: number;
  speed: number;
  alive: boolean;
  death_reason: DeathReason | null;
  seed: number;
}

export interface DinoOptions {
  width?: number;
  seed?: number;
  baseSpeed?: number;
  maxSpeed?: number;
}

/** Total ticks airborne after a JUMP, including the liftoff tick itself. */
const JUMP_TICKS = 10;
const OBSTACLE_WIDTH = 2;
const OBSTACLE_SPACING_MIN = 18;
const OBSTACLE_SPACING_MAX = 32;
/** How many x-units ahead the lookahead's reactive coast starts reacting to an obstacle. */
const REACTION_WINDOW = 10;
const SAFETY_HORIZON_TICKS = 30;
/** cactus / pterodactyl-low / pterodactyl-high, in that relative likelihood. */
const KIND_WEIGHTS: readonly ObstacleKind[] = ["cactus", "cactus", "cactus", "pterodactyl-low", "pterodactyl-high"];

interface DinoState {
  airborneTicksLeft: number;
  ducking: boolean;
}

/** One tick's worth of state transition. Shared by `step()` and the lookahead so they can never subtly diverge. */
function advanceState(state: DinoState, action: Action): { next: DinoState; airborne: boolean; ducking: boolean } {
  if (state.airborneTicksLeft > 0) {
    // Committed to an existing jump; the chosen action (if any) has no effect until landing.
    return { next: { airborneTicksLeft: state.airborneTicksLeft - 1, ducking: false }, airborne: true, ducking: false };
  }
  if (action === "JUMP") {
    return { next: { airborneTicksLeft: JUMP_TICKS - 1, ducking: false }, airborne: true, ducking: false };
  }
  const ducking = action === "DUCK";
  return { next: { airborneTicksLeft: 0, ducking }, airborne: false, ducking };
}

function collidesWith(obstacles: readonly { x: number; kind: ObstacleKind }[], airborne: boolean, ducking: boolean, dinoX: number): ObstacleKind | null {
  for (const o of obstacles) {
    if (Math.abs(o.x - dinoX) >= OBSTACLE_WIDTH / 2 + 0.5) continue;
    if (o.kind === "cactus" && !airborne) return "cactus";
    if (o.kind === "pterodactyl-low" && !ducking) return "pterodactyl-low";
    if (o.kind === "pterodactyl-high" && airborne) return "pterodactyl-high";
  }
  return null;
}

/**
 * JUMP for an imminent cactus, DUCK for an imminent low pterodactyl, RUN
 * otherwise -- see the file header for why this must be reactive, not
 * blind. The window extends slightly BEHIND the dino too (matching
 * `collidesWith`'s own `|x - dinoX| < OBSTACLE_WIDTH/2 + 0.5` zone, not
 * just `x >= dinoX`): an obstacle whose x has just crossed the dino's x is
 * still physically overlapping for a moment, and letting go of the
 * reaction (e.g. standing up out of a duck) exactly then is a collision,
 * not a release -- caught by writing a real multi-tick simulation and
 * checking it against actual play, not just against short, easy scenarios.
 */
function reactiveDefault(obstacles: readonly { x: number; kind: ObstacleKind }[], dinoX: number): Action {
  const behind = OBSTACLE_WIDTH / 2 + 0.5;
  const near = obstacles.find((o) => o.x - dinoX > -behind && o.x - dinoX <= REACTION_WINDOW);
  if (!near) return "RUN";
  if (near.kind === "cactus") return "JUMP";
  if (near.kind === "pterodactyl-low") return "DUCK";
  return "RUN"; // pterodactyl-high: jumping into it is the one thing that's actually unsafe
}

export class DinoGame {
  readonly width: number;
  readonly dinoX: number;
  readonly seed: number;
  readonly baseSpeed: number;
  readonly maxSpeed: number;

  tick = 0;
  airborneTicksLeft = 0;
  ducking = false;
  obstacles: Obstacle[] = [];
  score = 0;
  alive = true;
  deathReason: DeathReason | null = null;
  readonly #rng: DinoRng;

  constructor(options: DinoOptions = {}) {
    this.width = options.width ?? 60;
    this.seed = options.seed ?? 7;
    this.baseSpeed = options.baseSpeed ?? 0.6;
    this.maxSpeed = options.maxSpeed ?? 1.6;
    this.dinoX = Math.floor(this.width / 5);
    this.#rng = new DinoRng(this.seed);
    this.#ensureObstacles();
  }

  get airborne(): boolean {
    return this.airborneTicksLeft > 0;
  }

  get speed(): number {
    return Math.min(this.maxSpeed, this.baseSpeed + this.score * 0.0008);
  }

  #spawnObstacle(x: number): Obstacle {
    return { x, kind: KIND_WEIGHTS[this.#rng.int(0, KIND_WEIGHTS.length - 1)]!, scored: false };
  }

  #ensureObstacles(): void {
    while (this.obstacles.length === 0 || this.obstacles[this.obstacles.length - 1]!.x <= this.width + OBSTACLE_SPACING_MAX) {
      const last = this.obstacles[this.obstacles.length - 1];
      const x = last ? last.x + this.#rng.int(OBSTACLE_SPACING_MIN, OBSTACLE_SPACING_MAX) : this.width + OBSTACLE_SPACING_MIN;
      this.obstacles.push(this.#spawnObstacle(x));
    }
  }

  /** Simulate `action` now, then a reactive coast for `horizon` ticks; true if that path ever collides. No decision is ever requested while airborne -- see `moves()`. */
  #willCollideWithin(action: Action, horizon: number): boolean {
    let state: DinoState = { airborneTicksLeft: this.airborneTicksLeft, ducking: this.ducking };
    let obstacles = this.obstacles.map((o) => ({ x: o.x, kind: o.kind }));
    let speed = this.speed;
    let score = this.score;
    for (let t = 0; t < horizon; t++) {
      const chosen = t === 0 ? action : reactiveDefault(obstacles, this.dinoX);
      const { next, airborne, ducking } = advanceState(state, chosen);
      state = next;
      obstacles = obstacles.map((o) => ({ x: o.x - speed, kind: o.kind }));
      if (collidesWith(obstacles, airborne, ducking, this.dinoX)) return true;
      score += 1;
      speed = Math.min(this.maxSpeed, this.baseSpeed + score * 0.0008);
    }
    return false;
  }

  /** Both JUMP/DUCK/RUN are always legal (no illegal input); `safe` comes from the reactive lookahead. Empty while airborne -- no decision is meaningful mid-jump. */
  moves(): MoveInfo[] {
    if (!this.alive || this.airborne) return [];
    return ACTIONS.map((action) => {
      const safe = !this.#willCollideWithin(action, SAFETY_HORIZON_TICKS);
      return { action, legal: true, safe, reason: safe ? "safe" : `would collide within ${SAFETY_HORIZON_TICKS} ticks` };
    });
  }

  /** Advance one tick. `action` is ignored while airborne (see `moves()` -- no decision is requested for those ticks anyway). Returns true when an obstacle was passed (scored). */
  step(action: Action): boolean {
    if (!this.alive) throw new Error("Cannot step a finished game");
    const effectiveAction = this.airborne ? "RUN" : action;
    if (!this.airborne && !ACTIONS.includes(action)) throw new RangeError(`Unknown action: ${action}`);
    this.tick += 1;
    const state: DinoState = { airborneTicksLeft: this.airborneTicksLeft, ducking: this.ducking };
    const { next, airborne, ducking } = advanceState(state, effectiveAction);
    this.airborneTicksLeft = next.airborneTicksLeft;
    this.ducking = next.ducking;
    const speed = this.speed;
    for (const o of this.obstacles) o.x -= speed;
    this.#ensureObstacles();
    this.obstacles = this.obstacles.filter((o) => o.x > this.dinoX - OBSTACLE_WIDTH - 5);
    let scored = false;
    for (const o of this.obstacles) {
      if (!o.scored && o.x < this.dinoX) {
        o.scored = true;
        scored = true;
      }
    }
    this.score += 1;
    const hit = collidesWith(this.obstacles, airborne, ducking, this.dinoX);
    if (hit) {
      this.alive = false;
      this.deathReason = hit === "cactus" ? "cactus" : "pterodactyl";
    }
    return scored;
  }

  snapshot(): GameSnapshot {
    return {
      width: this.width,
      dino_x: this.dinoX,
      tick: this.tick,
      airborne_ticks_left: this.airborneTicksLeft,
      ducking: this.ducking,
      obstacles: this.obstacles.map((o) => ({ x: o.x, kind: o.kind, scored: o.scored })),
      score: this.score,
      speed: this.speed,
      alive: this.alive,
      death_reason: this.deathReason,
      seed: this.seed,
    };
  }
}
