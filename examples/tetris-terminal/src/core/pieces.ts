/**
 * The 7 standard tetrominoes, as cell offsets (row, col) per rotation
 * state. Only geometrically DISTINCT rotations are listed per piece
 * (`DISTINCT_ROTATIONS`) -- O has one shape, I/S/Z have two, T/J/L have
 * four -- so `legalPlacements` in game.ts never even generates a
 * duplicate-shape placement in the first place, rather than generating
 * then deduplicating.
 */
export type PieceKind = "I" | "O" | "T" | "S" | "Z" | "J" | "L";
export const PIECE_KINDS: readonly PieceKind[] = ["I", "O", "T", "S", "Z", "J", "L"];

export type RotationLabel = "0" | "R" | "2" | "L";
const ROTATION_ORDER: readonly RotationLabel[] = ["0", "R", "2", "L"];

export type Cell = PieceKind | null;
/** 20 rows x 10 cols, row 0 = top. No hidden buffer rows (see game.ts's game-over comment for why none is needed). */
export type Board = Cell[][];
export const BOARD_WIDTH = 10;
export const BOARD_HEIGHT = 20;

export type ShapeCell = readonly [row: number, col: number];

const SHAPES: Record<PieceKind, Partial<Record<RotationLabel, readonly ShapeCell[]>>> = {
  O: {
    "0": [[0, 0], [0, 1], [1, 0], [1, 1]],
  },
  I: {
    "0": [[0, 0], [0, 1], [0, 2], [0, 3]],
    R: [[0, 0], [1, 0], [2, 0], [3, 0]],
  },
  S: {
    "0": [[0, 1], [0, 2], [1, 0], [1, 1]],
    R: [[0, 0], [1, 0], [1, 1], [2, 1]],
  },
  Z: {
    "0": [[0, 0], [0, 1], [1, 1], [1, 2]],
    R: [[0, 1], [1, 0], [1, 1], [2, 0]],
  },
  T: {
    "0": [[0, 1], [1, 0], [1, 1], [1, 2]],
    R: [[0, 0], [1, 0], [1, 1], [2, 0]],
    "2": [[0, 0], [0, 1], [0, 2], [1, 1]],
    L: [[0, 1], [1, 0], [1, 1], [2, 1]],
  },
  J: {
    "0": [[0, 0], [1, 0], [1, 1], [1, 2]],
    R: [[0, 0], [0, 1], [1, 0], [2, 0]],
    "2": [[0, 0], [0, 1], [0, 2], [1, 2]],
    L: [[0, 1], [1, 1], [2, 0], [2, 1]],
  },
  L: {
    "0": [[0, 2], [1, 0], [1, 1], [1, 2]],
    R: [[0, 0], [1, 0], [2, 0], [2, 1]],
    "2": [[0, 0], [0, 1], [0, 2], [1, 0]],
    L: [[0, 0], [0, 1], [1, 1], [2, 1]],
  },
};

/** The rotation labels that produce a geometrically distinct shape for each piece, in a fixed enumeration order. */
export const DISTINCT_ROTATIONS: Record<PieceKind, readonly RotationLabel[]> = Object.fromEntries(
  PIECE_KINDS.map((kind) => [kind, ROTATION_ORDER.filter((r) => SHAPES[kind][r])]),
) as unknown as Record<PieceKind, readonly RotationLabel[]>;

export function shapeOf(kind: PieceKind, rotation: RotationLabel): readonly ShapeCell[] {
  const shape = SHAPES[kind][rotation];
  if (!shape) throw new RangeError(`${kind} has no distinct rotation ${rotation}`);
  return shape;
}

/**
 * Any of the 4 absolute labels resolves to a real shape, even for pieces
 * that don't have a geometrically distinct one at that label (O has only
 * "0"; I/S/Z only "0"/"R"). Works because each kind's distinct-rotation
 * count (1, 2, or 4) always evenly divides 4: `ROTATION_ORDER.slice(0, n)`
 * is exactly `DISTINCT_ROTATIONS[kind]` for all 7 standard kinds, so
 * indexing modulo that count always lands on a label `SHAPES` actually has.
 * This is what lets the per-step decision model freely choose "0"/"R"/"2"/"L"
 * (0/90/180/270 degrees) regardless of which piece is active.
 */
export function resolveRotation(kind: PieceKind, label: RotationLabel): RotationLabel {
  const period = DISTINCT_ROTATIONS[kind].length;
  return ROTATION_ORDER[ROTATION_ORDER.indexOf(label) % period]!;
}

/**
 * Whether `kind` at `rotation` stays within the board's columns when placed
 * with its bounding box starting at `col`.
 */
export function fitsAtColumn(kind: PieceKind, rotation: RotationLabel, col: number): boolean {
  const cols = shapeOf(kind, rotation).map(([, c]) => c);
  return col + Math.min(...cols) >= 0 && col + Math.max(...cols) < BOARD_WIDTH;
}

/** The centered spawn column for `kind`'s default ("0") orientation -- where a new piece actually enters play. */
export function spawnColFor(kind: PieceKind): number {
  const cols = shapeOf(kind, "0").map(([, c]) => c);
  const width = Math.max(...cols) - Math.min(...cols) + 1;
  return Math.floor((BOARD_WIDTH - width) / 2) - Math.min(...cols);
}

export function emptyBoard(): Board {
  return Array.from({ length: BOARD_HEIGHT }, () => Array<Cell>(BOARD_WIDTH).fill(null));
}
