/**
 * Deterministic Tetris rules. `legalPlacements` enumerates every reachable
 * (rotation, column) placement of the active piece via a drop-simulation
 * (this IS the complete legal set -- there's no extra "mandatory" layer
 * the way Checkers has mandatory capture). `moves()` additionally tags
 * each placement `safe`: does the stack height, measured AFTER line-clear
 * resolution, stay within `TOP_MARGIN` of the ceiling. This is a real,
 * sometimes-binding constraint (not vacuously identical to `legal`, unlike
 * a naive "safe = legal" would be) -- the same structural shape as Snake's
 * "would trap the snake" rule and Flappy Bird's "would collide within the
 * lookahead" rule, just computed from a lock+clear simulation instead of a
 * multi-tick physics simulation.
 *
 * Decisions are made once per PIECE, not once per tick: rotate+shift+
 * hard-drop+lock+clear is a single pure `applyPlacement` transformation,
 * with no animated-movement concept, the same way Checkers computes a
 * hop's resulting board directly rather than simulating a slide.
 */
import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  DISTINCT_ROTATIONS,
  emptyBoard,
  shapeOf,
  type Board,
  type Cell,
  type PieceKind,
  type RotationLabel,
  type ShapeCell,
} from "./pieces.ts";
import { TetrisRng } from "./rng.ts";

/** Spawn-buffer convention: a placement is "safe" only if the stack (after clearing) stays this far from the ceiling. */
export const TOP_MARGIN = 4;
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

function collides(board: Board, shape: readonly ShapeCell[], row: number, col: number): boolean {
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
      if (collides(board, shape, 0, col)) continue; // can't even occupy the spawn row here
      let row = 0;
      while (!collides(board, shape, row + 1, col)) row++;
      results.push({ kind, rotation, col, restRow: row });
    }
  }
  return results;
}

/** The stack height measured from the ceiling (0 = empty board). */
export function stackHeight(board: Board): number {
  for (let r = 0; r < BOARD_HEIGHT; r++) {
    if (board[r]!.some((c) => c !== null)) return BOARD_HEIGHT - r;
  }
  return 0;
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
    this.alive = legalPlacements(this.board, this.active).length > 0;
  }

  #draw(): PieceKind {
    if (this.#bag.length === 0) this.#bag = this.#rng.bag();
    return this.#bag.shift()!;
  }

  get level(): number {
    return Math.floor(this.linesCleared / 10) + 1;
  }

  legalPlacements(): Placement[] {
    return this.alive ? legalPlacements(this.board, this.active) : [];
  }

  moves(): PlacementInfo[] {
    return this.legalPlacements().map((placement) => {
      const { board: nextBoard, cleared } = applyPlacement(this.board, placement);
      const heightAfter = stackHeight(nextBoard);
      return { placement, legal: true, safe: heightAfter <= BOARD_HEIGHT - TOP_MARGIN, clears: cleared, heightAfter };
    });
  }

  /**
   * Apply a placement: lock, clear, draw the next piece, then check game
   * over. Game over here means block-out (the new piece's spawn cells
   * collide) -- `legalPlacements()` returning empty and game-over are the
   * SAME event, mirroring Checkers' "no legal moves = loss". This is a
   * different condition from `moves()` finding no SAFE placement, which
   * does NOT mean game over (see policy.ts's decisionFrom) -- don't
   * confuse the two empty-set cases.
   */
  applyPlacement(placement: Placement): number {
    if (!this.alive) throw new Error("Cannot apply a placement to a finished game");
    const { board, cleared } = applyPlacement(this.board, placement);
    this.board = board;
    this.score += LINE_SCORES[cleared]! * this.level;
    this.linesCleared += cleared;
    this.active = this.queue.shift()!;
    this.queue.push(this.#draw());
    this.alive = legalPlacements(this.board, this.active).length > 0;
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
