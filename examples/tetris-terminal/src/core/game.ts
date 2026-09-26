/**
 * Deterministic Tetris rules. `legalPlacements` enumerates every reachable
 * (rotation, column) placement of the active piece via a drop-simulation
 * from spawn -- a standalone utility (hints, tooling, tests); gameplay no
 * longer calls it (see below).
 *
 * Decisions are made once per gravity STEP, not once per piece: at each
 * step the model picks a rotation, a horizontal direction and a distance
 * (see policy.ts), applied via `sweepColumns` (lateral movement, clamped to
 * whatever's actually free -- no wall kicks). When the piece can no longer
 * descend, it gets exactly one more such decision (a bounded "lock delay" /
 * "extended placement", see policy.ts) before `applyPlacement` locks it and
 * clears full rows.
 *
 * `optionsAtRow` is the shield's evaluation set at that final lock decision:
 * everything reachable via rotate+shift ALONE from the piece's current
 * position, not a full drop-simulation from spawn -- local to "what could
 * this step still reach," not global re-planning of the whole descent.
 * `sweepColumns` (walk outward from the current column, stop at the first
 * collision each way) is what keeps this local set from including
 * geometrically disconnected shelves the piece could never actually have
 * slid to.
 */
import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  DISTINCT_ROTATIONS,
  emptyBoard,
  shapeOf,
  spawnColFor,
  type Board,
  type Cell,
  type PieceKind,
  type RotationLabel,
  type ShapeCell,
} from "./pieces.ts";
import { TetrisRng } from "./rng.ts";

/** Spawn-buffer convention: a placement is "safe" only if the stack (after clearing) stays this far from the ceiling. */
export const TOP_MARGIN = 4;
/** Every drop-simulation (and any UI animating a piece's descent) starts here. */
export const SPAWN_ROW = 0;
const LINE_SCORES = [0, 100, 300, 500, 800];
const QUEUE_LOOKAHEAD = 5;

export interface Placement {
  kind: PieceKind;
  rotation: RotationLabel;
  col: number;
  restRow: number;
}

/** "{kind}-{rotation}-c{col}", e.g. "T-0-c3", "I-R-c6" -- unique per distinct reachable placement. */
export function placementKey(p: Placement): string {
  return `${p.kind}-${p.rotation}-c${p.col}`;
}

export interface PlacementInfo {
  placement: Placement;
  /** Always true: legalPlacements() only ever generates geometrically legal placements. */
  legal: true;
  safe: boolean;
  clears: number;
  heightAfter: number;
}

export interface GameSnapshot {
  board: Board;
  active: PieceKind;
  queue: PieceKind[];
  score: number;
  linesCleared: number;
  level: number;
  alive: boolean;
  seed: number;
}

/** Whether `shape` anchored at `(row, col)` collides with a wall, the stack, or the board's bounds. */
export function collides(board: Board, shape: readonly ShapeCell[], row: number, col: number): boolean {
  for (const [dr, dc] of shape) {
    const r = row + dr;
    const c = col + dc;
    if (c < 0 || c >= BOARD_WIDTH || r >= BOARD_HEIGHT) return true;
    if (r >= 0 && board[r]![c]) return true;
  }
  return false;
}

/** Every reachable (rotation, column) placement of `kind` on `board`, via drop-simulation from row 0. */
export function legalPlacements(board: Board, kind: PieceKind): Placement[] {
  const results: Placement[] = [];
  for (const rotation of DISTINCT_ROTATIONS[kind]) {
    const shape = shapeOf(kind, rotation);
    const cols = shape.map(([, c]) => c);
    const minCol = -Math.min(...cols) || 0; // avoid -0 (every shape's own min column offset is 0)
    const maxCol = BOARD_WIDTH - 1 - Math.max(...cols);
    for (let col = minCol; col <= maxCol; col++) {
      if (collides(board, shape, SPAWN_ROW, col)) continue; // can't even occupy the spawn row here
      let row = SPAWN_ROW;
      while (!collides(board, shape, row + 1, col)) row++;
      results.push({ kind, rotation, col, restRow: row });
    }
  }
  return results;
}

/**
 * Whether `kind` can enter play at all: does its default ("0") orientation,
 * at its fixed centered spawn column, collide right now. This is the new
 * block-out check -- narrower and more realistic than the old
 * "does ANY rotation/column combination fit on row 0 somewhere" test, since
 * the piece actually enters play in one specific orientation and column,
 * not by trying every hypothetical placement.
 */
export function canSpawn(board: Board, kind: PieceKind): boolean {
  return !collides(board, shapeOf(kind, "0"), SPAWN_ROW, spawnColFor(kind));
}

/** The stack height measured from the ceiling (0 = empty board). */
export function stackHeight(board: Board): number {
  for (let r = 0; r < BOARD_HEIGHT; r++) {
    if (board[r]!.some((c) => c !== null)) return BOARD_HEIGHT - r;
  }
  return 0;
}

/**
 * Per-column height measured from the ceiling (0 = that column is empty) --
 * the board-shape profile `stackHeight`'s single overall number doesn't
 * carry, used to give the per-step prompt real terrain to reason about
 * instead of just one scalar.
 */
export function columnHeights(board: Board): number[] {
  const heights = new Array<number>(BOARD_WIDTH).fill(0);
  for (let c = 0; c < BOARD_WIDTH; c++) {
    for (let r = 0; r < BOARD_HEIGHT; r++) {
      if (board[r]![c]) {
        heights[c] = BOARD_HEIGHT - r;
        break;
      }
    }
  }
  return heights;
}

/**
 * How many columns `shape` can move from `col` toward `step` (`-1` left,
 * `1` right) before the next one would collide -- the live shift's own
 * clamp, and the prompt's "how far is actually free" hint, share this.
 */
export function maxFreeDistance(board: Board, shape: readonly ShapeCell[], row: number, col: number, step: -1 | 1): number {
  let n = 0;
  while (!collides(board, shape, row, col + step * (n + 1))) n++;
  return n;
}

/**
 * Every column reachable from `fromCol` at `row` for `shape`, by walking
 * outward one column at a time and stopping at the first collision each
 * way (wall, stack, or out of bounds) -- clamped lateral movement, shared
 * by a live step's shift and by `optionsAtRow`'s reachability check.
 * `fromCol` is assumed legal already (true of any current position) and is
 * always included in the result.
 */
export function sweepColumns(board: Board, shape: readonly ShapeCell[], row: number, fromCol: number): number[] {
  const cols = [fromCol];
  for (let c = fromCol - 1; !collides(board, shape, row, c); c--) cols.push(c);
  for (let c = fromCol + 1; !collides(board, shape, row, c); c++) cols.push(c);
  return cols.sort((a, b) => a - b);
}

/**
 * The lock-time shield's evaluation set: every (rotation, column) reachable
 * from the piece's CURRENT `(row, col)` via rotate-in-place-then-shift
 * alone (no wall kicks -- a rotation illegal at `col` is simply skipped),
 * restricted to options that are themselves resting at `row` (this is the
 * final lock decision; only options that couldn't descend further belong
 * here). Always includes the current position itself (`sweepColumns`
 * guarantees it), so this set is never empty.
 */
export function optionsAtRow(board: Board, kind: PieceKind, row: number, col: number): PlacementInfo[] {
  const results: PlacementInfo[] = [];
  for (const rotation of DISTINCT_ROTATIONS[kind]) {
    const shape = shapeOf(kind, rotation);
    if (collides(board, shape, row, col)) continue; // illegal in place at this column -- no wall-kick attempt
    for (const c of sweepColumns(board, shape, row, col)) {
      if (!collides(board, shape, row + 1, c)) continue; // could still descend from here -- not a lock option
      const placement: Placement = { kind, rotation, col: c, restRow: row };
      const { board: nextBoard, cleared } = applyPlacement(board, placement);
      const heightAfter = stackHeight(nextBoard);
      results.push({ placement, legal: true, safe: heightAfter <= BOARD_HEIGHT - TOP_MARGIN, clears: cleared, heightAfter });
    }
  }
  return results;
}

/** Lock `placement` into `board`, then clear full rows. Pure -- returns a new board. */
export function applyPlacement(board: Board, placement: Placement): { board: Board; cleared: number } {
  const shape = shapeOf(placement.kind, placement.rotation);
  const next: Board = board.map((row) => [...row]);
  for (const [dr, dc] of shape) next[placement.restRow + dr]![placement.col + dc] = placement.kind;
  const kept = next.filter((row) => row.some((cell) => cell === null));
  const cleared = BOARD_HEIGHT - kept.length;
  const filler: Cell[][] = Array.from({ length: cleared }, () => Array<Cell>(BOARD_WIDTH).fill(null));
  return { board: [...filler, ...kept], cleared };
}

export class TetrisGame {
  board: Board;
  active: PieceKind;
  queue: PieceKind[] = [];
  score = 0;
  linesCleared = 0;
  alive: boolean;
  readonly seed: number;
  #bag: PieceKind[] = [];
  readonly #rng: TetrisRng;

  constructor(seed = 7) {
    this.seed = seed;
    this.#rng = new TetrisRng(seed);
    this.board = emptyBoard();
    this.active = this.#draw();
    while (this.queue.length < QUEUE_LOOKAHEAD) this.queue.push(this.#draw());
    this.alive = canSpawn(this.board, this.active);
  }

  #draw(): PieceKind {
    if (this.#bag.length === 0) this.#bag = this.#rng.bag();
    return this.#bag.shift()!;
  }

  get level(): number {
    return Math.floor(this.linesCleared / 10) + 1;
  }

  /** Every reachable placement of the active piece -- a standalone utility (hints, tooling, tests), not used to decide play. */
  legalPlacements(): Placement[] {
    return this.alive ? legalPlacements(this.board, this.active) : [];
  }

  /**
   * Apply a placement: lock, clear, draw the next piece, then check game
   * over. Game over here means block-out -- the new piece's fixed default
   * spawn configuration collides (`canSpawn`), the same event a real game
   * of Tetris calls block-out, mirroring Checkers' "no legal moves = loss".
   * This is unrelated to the per-step shield finding no SAFE lock option,
   * which is routine near the top of a real game, not game over (see
   * policy.ts) -- don't confuse the two.
   */
  applyPlacement(placement: Placement): number {
    if (!this.alive) throw new Error("Cannot apply a placement to a finished game");
    const { board, cleared } = applyPlacement(this.board, placement);
    this.board = board;
    this.score += LINE_SCORES[cleared]! * this.level;
    this.linesCleared += cleared;
    this.active = this.queue.shift()!;
    this.queue.push(this.#draw());
    this.alive = canSpawn(this.board, this.active);
    return cleared;
  }

  snapshot(): GameSnapshot {
    return {
      board: this.board.map((row) => [...row]),
      active: this.active,
      queue: [...this.queue],
      score: this.score,
      linesCleared: this.linesCleared,
      level: this.level,
      alive: this.alive,
      seed: this.seed,
    };
  }
}
