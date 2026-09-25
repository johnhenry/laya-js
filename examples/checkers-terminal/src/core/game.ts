/**
 * The chain/kinging state machine: `applyHop` is where the mandatory
 * multi-jump rules actually live. Order within one hop's application:
 *
 *   1. Move the piece; remove any captured piece.
 *   2. Kinging check FIRST -- a man landing on the back row is promoted
 *      immediately, before anything else is decided.
 *   3. Chain-continuation lookahead, but ONLY if the hop was a capture AND
 *      the piece did NOT just king this hop -- a man that crowns mid-chain
 *      stops immediately, even if a further jump would otherwise be
 *      available (the standard rule, not a simplification here).
 *   4. If a continuation exists, `forcedContinuation` is set and the turn
 *      does not pass; otherwise it's cleared and the turn passes.
 *   5. After applying, if the new mover has zero legal hops, the game ends.
 *
 * The turn loop in session.ts is deliberately chain-unaware: it always runs
 * the same compliantHops -> decide -> applyHop cycle, and whether the same
 * player is asked again falls entirely out of `forcedContinuation`
 * persisting (or not) across this function -- no imperative "if chain, loop
 * again" branch exists anywhere.
 */
import { initialBoard, isBackRowFor, opponent, type Board, type Player, type Square } from "./board.ts";
import { compliantHops, hopKey, legalHops, type ForcedContinuation, type Hop } from "./moves.ts";

export type GameStatus = "in_progress" | "red_wins" | "black_wins";

export interface HopRecord {
  hop: Hop;
  player: Player;
  turnNumber: number;
  kinged: boolean;
}

export interface GameSnapshot {
  board: Board;
  toMove: Player;
  turnNumber: number;
  forcedContinuation: ForcedContinuation | null;
  status: GameStatus;
  moveHistory: HopRecord[];
}

export function newGame(): GameSnapshot {
  return {
    board: initialBoard(),
    toMove: "red",
    turnNumber: 1,
    forcedContinuation: null,
    status: "in_progress",
    moveHistory: [],
  };
}

export function legalHopsFor(state: GameSnapshot): Hop[] {
  return legalHops(state.board, state.toMove);
}

export function compliantHopsFor(state: GameSnapshot): Hop[] {
  return compliantHops(state.board, state.toMove, state.forcedContinuation);
}

export function isCompliant(state: GameSnapshot, hop: Hop): boolean {
  const key = hopKey(hop);
  return compliantHopsFor(state).some((h) => hopKey(h) === key);
}

export function applyHop(state: GameSnapshot, hop: Hop): GameSnapshot {
  if (state.status !== "in_progress") throw new Error("Cannot apply a hop to a finished game");
  if (!isCompliant(state, hop)) throw new Error(`Hop ${hopKey(hop)} is not compliant with the current turn's rules`);

  const board: Board = { ...state.board };
  const piece = { ...board[hop.from]! };
  delete board[hop.from];
  if (hop.kind === "capture") delete board[hop.captured!];
  board[hop.to] = piece;

  let kingedThisHop = false;
  if (piece.kind === "man" && isBackRowFor(piece.player, hop.to)) {
    piece.kind = "king";
    kingedThisHop = true;
  }

  let forcedContinuation: ForcedContinuation | null = null;
  let turnEnds = true;
  if (hop.kind === "capture" && !kingedThisHop) {
    const further = legalHops(board, state.toMove).filter((h) => h.kind === "capture" && h.from === hop.to);
    if (further.length > 0) {
      forcedContinuation = { square: hop.to };
      turnEnds = false;
    }
  }

  const nextToMove = turnEnds ? opponent(state.toMove) : state.toMove;
  const moveHistory = [...state.moveHistory, { hop, player: state.toMove, turnNumber: state.turnNumber, kinged: kingedThisHop }];

  let next: GameSnapshot = {
    board,
    toMove: nextToMove,
    turnNumber: turnEnds ? state.turnNumber + 1 : state.turnNumber,
    forcedContinuation: turnEnds ? null : forcedContinuation,
    status: "in_progress",
    moveHistory,
  };

  if (legalHops(next.board, next.toMove).length === 0) {
    next = { ...next, status: next.toMove === "red" ? "black_wins" : "red_wins" };
  }
  return next;
}

export function materialCount(board: Board): Record<Player, { men: number; kings: number }> {
  const out = { red: { men: 0, kings: 0 }, black: { men: 0, kings: 0 } };
  for (const piece of Object.values(board)) {
    if (!piece) continue;
    if (piece.kind === "man") out[piece.player].men++;
    else out[piece.player].kings++;
  }
  return out;
}
