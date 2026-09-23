/**
 * Deterministic Snake rules and a separately identified cycle safety planner.
 * Faithful port of laya-mlx `laya_mlx/snake/game.py` (same Hamiltonian cycle,
 * same safety rules, same Python-compatible food RNG for a given seed).
 */
import { PyRandom } from "./pyrandom.ts";

export const DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"] as const;
export type Direction = (typeof DIRECTIONS)[number];
export type Cell = readonly [number, number];
export const VECTORS: Readonly<Record<Direction, Cell>> = {
  UP: [0, -1],
  DOWN: [0, 1],
  LEFT: [-1, 0],
  RIGHT: [1, 0],
};

/** Visit each square once with adjacent steps, including the closing edge. */
export function hamiltonianCycle(width: number, height: number): Cell[] {
  if (Math.min(width, height) < 4 || (width % 2 && height % 2)) {
    throw new RangeError("Board dimensions must be >= 4, with at least one even dimension");
  }
  if (height % 2) return hamiltonianCycle(height, width).map(([x, y]) => [y, x] as const);
  const path: Cell[] = [[0, 0]];
  for (let y = 0; y < height; y++) {
    if (y % 2 === 0) for (let x = 1; x < width; x++) path.push([x, y]);
    else for (let x = width - 1; x > 0; x--) path.push([x, y]);
  }
  for (let y = height - 1; y > 0; y--) path.push([0, y]);
  return path;
}

export interface MoveInfo {
  direction: Direction;
  legal: boolean;
  safe: boolean;
  advance: number;
  reason: string;
  eats: boolean;
}

export type LegalReason = "wall" | "reverse" | "body" | "legal";

export interface GameSnapshot {
  width: number;
  height: number;
  seed: number;
  body: [number, number][];
  food: [number, number] | null;
  score: number;
  length: number;
  ticks: number;
  alive: boolean;
  won: boolean;
  death_reason: string | null;
}

const same = (a: Cell | null, b: Cell | null) => !!a && !!b && a[0] === b[0] && a[1] === b[1];

export class SnakeGame {
  readonly width: number;
  readonly height: number;
  readonly seed: number;
  readonly cycle: Cell[];
  readonly capacity: number;
  readonly initialLength: number;
  /** Head first (Python's deque order). */
  body: Cell[];
  food: Cell | null;
  score = 0;
  ticks = 0;
  alive = true;
  won = false;
  deathReason: string | null = null;
  readonly #index: Int32Array; // y * width + x -> cycle index
  readonly #rng: PyRandom;

  constructor(width = 24, height = 16, seed = 7, initialLength = 6) {
    this.width = width;
    this.height = height;
    this.seed = seed;
    this.cycle = hamiltonianCycle(width, height);
    this.capacity = width * height;
    this.#index = new Int32Array(this.capacity);
    this.cycle.forEach(([x, y], i) => (this.#index[y * width + x] = i));
    if (!(initialLength >= 2 && initialLength < this.capacity)) {
      throw new RangeError("Initial length must be >= 2 and smaller than the board");
    }
    this.initialLength = initialLength;
    this.#rng = new PyRandom(seed);
    const start = this.indexOf([Math.floor(width / 2), Math.floor(height / 2)]);
    this.body = Array.from({ length: initialLength }, (_, i) => this.cycle[mod(start - i, this.capacity)]!);
    this.food = this.#spawnFood();
  }

  /** Rebuild a game from a recorded snapshot (the food RNG restarts from `seed`). */
  static fromSnapshot(s: Pick<GameSnapshot, "width" | "height" | "seed" | "body" | "food"> & Partial<GameSnapshot>): SnakeGame {
    const g = new SnakeGame(s.width, s.height, s.seed, Math.max(2, Math.min(s.body.length, s.width * s.height - 1)));
    g.body = s.body.map(([x, y]) => [x, y] as const);
    g.food = s.food ? [s.food[0], s.food[1]] : null;
    g.score = s.score ?? 0;
    g.ticks = s.ticks ?? 0;
    g.alive = s.alive ?? true;
    g.won = s.won ?? false;
    g.deathReason = s.death_reason ?? null;
    return g;
  }

  get head(): Cell {
    return this.body[0]!;
  }

  inBounds([x, y]: Cell): boolean {
    return x >= 0 && x < this.width && y >= 0 && y < this.height;
  }

  indexOf(cell: Cell): number {
    return this.#index[cell[1] * this.width + cell[0]]!;
  }

  #key([x, y]: Cell): number {
    return y * this.width + x;
  }

  #spawnFood(): Cell | null {
    const occupied = new Set(this.body.map((c) => this.#key(c)));
    const empty = this.cycle.filter((c) => !occupied.has(this.#key(c)));
    return empty.length ? this.#rng.choice(empty) : null;
  }

  target(direction: Direction): Cell {
    const [dx, dy] = VECTORS[direction];
    return [this.head[0] + dx, this.head[1] + dy];
  }

  legalReason(direction: Direction): LegalReason {
    const cell = this.target(direction);
    if (!this.inBounds(cell)) return "wall";
    if (same(cell, this.body[1]!)) return "reverse";
    // The tail moves on a non-growing step.
    const body = same(cell, this.food) ? this.body : this.body.slice(0, -1);
    return body.some((c) => same(c, cell)) ? "body" : "legal";
  }

  moves(): MoveInfo[] {
    if (!this.alive || this.won) return [];
    const headIndex = this.indexOf(this.head);
    const tailDistance = mod(this.indexOf(this.body[this.body.length - 1]!) - headIndex, this.capacity);
    const foodDistance = mod(this.indexOf(this.food!) - headIndex, this.capacity);
    return DIRECTIONS.map((direction) => {
      let reason: string = this.legalReason(direction);
      const legal = reason === "legal";
      const target = this.target(direction);
      const advance = mod((this.inBounds(target) ? this.indexOf(target) : headIndex) - headIndex, this.capacity);
      const eats = same(target, this.food);
      let safe = legal;
      if (safe && (advance > tailDistance || (advance === tailDistance && eats))) {
        safe = false;
        reason = "would cross the tail";
      }
      if (safe && (advance === 0 || advance > foodDistance)) {
        safe = false;
        reason = "would skip the food on the safe route";
      }
      return { direction, legal, safe, advance, reason, eats };
    });
  }

  /** Current empty-cell connectivity; the occupied tail is not treated as empty. */
  foodReachability(): [reachable: boolean, space: number] {
    const blocked = new Set(this.body.slice(1).map((c) => this.#key(c)));
    const visited = new Set([this.#key(this.head)]);
    const queue: Cell[] = [this.head];
    for (let q = 0; q < queue.length; q++) {
      const [x, y] = queue[q]!;
      for (const [dx, dy] of Object.values(VECTORS)) {
        const cell: Cell = [x + dx, y + dy];
        const key = this.#key(cell);
        if (this.inBounds(cell) && !blocked.has(key) && !visited.has(key)) {
          visited.add(key);
          queue.push(cell);
        }
      }
    }
    return [this.food !== null && visited.has(this.#key(this.food)), visited.size];
  }

  /** Advance one tick; returns true when food was eaten. */
  step(direction: Direction): boolean {
    if (!this.alive || this.won) throw new Error("Cannot step a finished game");
    if (!DIRECTIONS.includes(direction)) throw new RangeError(`Unknown direction: ${direction}`);
    this.ticks += 1;
    const reason = this.legalReason(direction);
    if (reason !== "legal") {
      this.alive = false;
      this.deathReason = reason;
      return false;
    }
    const target = this.target(direction);
    this.body.unshift(target);
    if (same(target, this.food)) {
      this.score += 1;
      if (this.body.length === this.capacity) {
        this.won = true;
        this.food = null;
      } else {
        this.food = this.#spawnFood();
      }
      return true;
    }
    this.body.pop();
    return false;
  }

  cycleOrderValid(): boolean {
    const indices = [...this.body].reverse().map((c) => this.indexOf(c));
    let sum = 0;
    for (let i = 1; i < indices.length; i++) {
      const d = mod(indices[i]! - indices[i - 1]!, this.capacity);
      if (d <= 0) return false;
      sum += d;
    }
    return sum < this.capacity;
  }

  snapshot(): GameSnapshot {
    return {
      width: this.width,
      height: this.height,
      seed: this.seed,
      body: this.body.map(([x, y]) => [x, y]),
      food: this.food ? [this.food[0], this.food[1]] : null,
      score: this.score,
      length: this.body.length,
      ticks: this.ticks,
      alive: this.alive,
      won: this.won,
      death_reason: this.deathReason,
    };
  }
}

function mod(a: number, n: number): number {
  return ((a % n) + n) % n;
}
