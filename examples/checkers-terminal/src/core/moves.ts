/**
 * Move generation: `legalHops` (every geometrically legal single hop, a
 * simple step or one capture jump, ignoring mandatory capture) and
 * `compliantHops` (what mandatory capture / an in-progress multi-jump chain
 * actually permits this turn). One hop is one decision point -- a full
 * multi-jump chain is never collapsed into a single option; the chain state
 * machine lives in `game.ts`.
 *
 * `compliantHops` is empty if and only if `legalHops` is empty (game over):
 * never empty mid-turn, since `forcedContinuation` is only ever set (in
 * `game.ts`) after confirming a further capture exists.
 */
import { ALL_FOUR_DIAGONALS, forwardDiagonalsFor, ownPieces, step, type Board, type Player, type Square } from "./board.ts";

export type HopKind = "simple" | "capture";

export interface Hop {
  kind: HopKind;
  from: Square;
  to: Square;
  /** Set iff `kind === "capture"` -- the jumped-over square. */
  captured?: Square;
}

export interface ForcedContinuation {
  square: Square;
}

/** Notation for `choice` criteria keys: "c3-d4" (simple) / "c3xd4" (capture). Unambiguous: a (from, to) pair has exactly one geometric path. */
export function hopKey(h: Hop): string {
  return h.kind === "simple" ? `${h.from}-${h.to}` : `${h.from}x${h.to}`;
}

export function legalHops(board: Board, player: Player): Hop[] {
  const hops: Hop[] = [];
  for (const [sq, piece] of ownPieces(board, player)) {
    const dirs = piece.kind === "king" ? ALL_FOUR_DIAGONALS : forwardDiagonalsFor(piece.player);
    for (const dir of dirs) {
      const adj = step(sq, dir.df, dir.dr, 1);
      if (!adj) continue;
      const occupant = board[adj];
      if (!occupant) {
        hops.push({ kind: "simple", from: sq, to: adj });
      } else if (occupant.player !== piece.player) {
        const landing = step(sq, dir.df, dir.dr, 2);
        if (landing && !board[landing]) hops.push({ kind: "capture", from: sq, to: landing, captured: adj });
      }
    }
  }
  return hops;
}

export function compliantHops(board: Board, player: Player, forced: ForcedContinuation | null): Hop[] {
  const legal = legalHops(board, player);
  if (forced) return legal.filter((h) => h.kind === "capture" && h.from === forced.square);
  const captures = legal.filter((h) => h.kind === "capture");
  return captures.length > 0 ? captures : legal;
}
