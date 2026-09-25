/**
 * Board representation and square/direction math for American-style
 * checkers (8x8, single-square kings, no flying kings). Dark squares only
 * are ever occupied: a square is playable when `fileIndex + rank` is odd
 * (so a1 is a playable/dark square, matching the standard board), and a
 * diagonal step always preserves that parity.
 */
export const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"] as const;
export type File = (typeof FILES)[number];
export type Rank = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;
export type Square = `${File}${Rank}`;

export type Player = "red" | "black";
export type PieceKind = "man" | "king";

export interface Piece {
  player: Player;
  kind: PieceKind;
}

/** Sparse: only dark squares are ever keyed. */
export type Board = Partial<Record<Square, Piece>>;

export function opponent(player: Player): Player {
  return player === "red" ? "black" : "red";
}

export function fileIndex(f: File): number {
  return FILES.indexOf(f);
}

export function isPlayable(fi: number, rank: number): boolean {
  return (fi + rank) % 2 === 1;
}

export function makeSquare(fi: number, rank: number): Square | null {
  if (fi < 0 || fi > 7 || rank < 1 || rank > 8 || !isPlayable(fi, rank)) return null;
  return `${FILES[fi]}${rank}` as Square;
}

export function parseSquare(sq: Square): { fi: number; rank: number } {
  return { fi: fileIndex(sq[0] as File), rank: Number(sq.slice(1)) };
}

export function step(sq: Square, df: number, dr: number, n = 1): Square | null {
  const { fi, rank } = parseSquare(sq);
  return makeSquare(fi + df * n, rank + dr * n);
}

export interface Direction {
  name: "NE" | "NW" | "SE" | "SW";
  df: number;
  dr: number;
}

export const DIRECTIONS: Readonly<Record<Direction["name"], Direction>> = {
  NE: { name: "NE", df: 1, dr: 1 },
  NW: { name: "NW", df: -1, dr: 1 },
  SE: { name: "SE", df: 1, dr: -1 },
  SW: { name: "SW", df: -1, dr: -1 },
};

export const ALL_FOUR_DIAGONALS: readonly Direction[] = [DIRECTIONS.NE, DIRECTIONS.NW, DIRECTIONS.SE, DIRECTIONS.SW];

/** Red moves toward increasing rank; black moves toward decreasing rank. */
export function forwardDiagonalsFor(player: Player): readonly Direction[] {
  return player === "red" ? [DIRECTIONS.NE, DIRECTIONS.NW] : [DIRECTIONS.SE, DIRECTIONS.SW];
}

export function isBackRowFor(player: Player, sq: Square): boolean {
  const { rank } = parseSquare(sq);
  return player === "red" ? rank === 8 : rank === 1;
}

/** The standard 8x8 setup: red on ranks 1-3, black on ranks 6-8, all as men. */
export function initialBoard(): Board {
  const board: Board = {};
  for (let rank = 1 as Rank; rank <= 3; rank++) {
    for (let fi = 0; fi < 8; fi++) {
      const sq = makeSquare(fi, rank);
      if (sq) board[sq] = { player: "red", kind: "man" };
    }
  }
  for (let rank = 6 as Rank; rank <= 8; rank++) {
    for (let fi = 0; fi < 8; fi++) {
      const sq = makeSquare(fi, rank);
      if (sq) board[sq] = { player: "black", kind: "man" };
    }
  }
  return board;
}

export function ownPieces(board: Board, player: Player): [Square, Piece][] {
  return Object.entries(board).filter(([, p]) => p?.player === player) as [Square, Piece][];
}
