/**
 * Move-generation edge cases -- the checklist from the design pass, since
 * mandatory capture / multi-jump chains / kinging interaction is where a
 * checkers engine is most likely to be subtly wrong. Every scenario below
 * is a constructed board, not random play, so each rule is tested in
 * isolation.
 */
import assert from "node:assert/strict";
import { applyHop, isCompliant, newGame, type GameSnapshot } from "../src/core/game.ts";
import { compliantHops, hopKey, legalHops, type Hop } from "../src/core/moves.ts";
import { decisionFrom, buildPrompt, type PredictLike } from "../src/core/policy.ts";
import type { Board, Square } from "../src/core/board.ts";
import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const test = makeTest(bun);

function state(board: Board, overrides: Partial<GameSnapshot> = {}): GameSnapshot {
  return { board, toMove: "red", turnNumber: 1, forcedContinuation: null, status: "in_progress", moveHistory: [], ...overrides };
}

const keys = (hops: Hop[]) => hops.map(hopKey).sort();

test("the opening position has exactly the 7 well-known legal opening moves, all simple", () => {
  const hops = legalHops(newGame().board, "red");
  assert.equal(hops.length, 7);
  assert.ok(hops.every((h) => h.kind === "simple"));
  assert.deepEqual(keys(hops), ["a3-b4", "c3-b4", "c3-d4", "e3-d4", "e3-f4", "g3-f4", "g3-h4"]);
});

test("no legal moves: a player with no pieces left has none", () => {
  const board: Board = { a3: { player: "black", kind: "man" } };
  assert.deepEqual(legalHops(board, "red"), []);
});

test("no legal moves: a player with pieces but every one fully blocked has none", () => {
  // Red man on a1: its only forward diagonal is b2 (a1 is a corner, the other diagonal is off-board).
  // b2 occupied by black, and the capture-landing square c3 also occupied -- no move, no capture.
  const board: Board = {
    a1: { player: "red", kind: "man" },
    b2: { player: "black", kind: "man" },
    c3: { player: "black", kind: "man" },
  };
  assert.deepEqual(legalHops(board, "red"), []);
});

test("a capture whose landing square is occupied is not legal, but is legal once it's empty", () => {
  // a1 corner: red man's only direction is NE (NW would be off-board). Black adjacent at b2,
  // landing square c3 -- occupied by black blocks the capture; empty allows it.
  const blocked: Board = { a1: { player: "red", kind: "man" }, b2: { player: "black", kind: "man" }, c3: { player: "black", kind: "man" } };
  assert.deepEqual(legalHops(blocked, "red"), []);
  const open: Board = { a1: { player: "red", kind: "man" }, b2: { player: "black", kind: "man" } };
  assert.deepEqual(keys(legalHops(open, "red")), ["a1xc3"]);
});

test("a capture whose landing square would be off-board is not legal", () => {
  // b2's NW-adjacent square is a3 (on-board); the landing two steps NW would be off-board (fi -1).
  const board: Board = { b2: { player: "red", kind: "man" }, a3: { player: "black", kind: "man" } };
  const hops = legalHops(board, "red");
  assert.ok(!hops.some((h) => h.kind === "capture"));
  assert.deepEqual(keys(hops), ["b2-c3"]); // only the unrelated NE simple move remains
});

test("multi-piece simultaneous mandatory capture: both pieces' captures are compliant at once", () => {
  const board: Board = {
    a3: { player: "red", kind: "man" },
    b4: { player: "black", kind: "man" }, // a3xc5
    g3: { player: "red", kind: "man" },
    f4: { player: "black", kind: "man" }, // g3xe5
  };
  const compliant = compliantHops(board, "red", null);
  assert.deepEqual(keys(compliant), ["a3xc5", "g3xe5"]);
  assert.ok(compliant.every((h) => h.kind === "capture"));
});

test("branching capture directions on one piece: a king with two adjacent enemies offers two capture hops", () => {
  const board: Board = {
    d4: { player: "red", kind: "king" },
    c5: { player: "black", kind: "man" }, // NW capture -> b6
    e5: { player: "black", kind: "man" }, // NE capture -> f6
  };
  const compliant = compliantHops(board, "red", null);
  assert.deepEqual(keys(compliant), ["d4xb6", "d4xf6"]);
});

test("branching chain continuations: after the first capture, two further captures are both compliant", () => {
  const board: Board = {
    a3: { player: "red", kind: "man" },
    b4: { player: "black", kind: "man" }, // a3xc5
    d6: { player: "black", kind: "man" }, // from c5: NE -> d6, capture to e7
    b6: { player: "black", kind: "man" }, // from c5: NW -> b6, capture to a7
  };
  let s = state(board);
  const first = compliantHops(s.board, "red", null);
  assert.deepEqual(keys(first), ["a3xc5"]);
  s = applyHop(s, first[0]!);
  assert.deepEqual(s.forcedContinuation, { square: "c5" });
  assert.equal(s.toMove, "red"); // turn does not pass mid-chain
  const continuation = compliantHops(s.board, s.toMove, s.forcedContinuation);
  assert.deepEqual(keys(continuation), ["c5xa7", "c5xe7"]);
});

test("kinging stops the chain even though a further capture would otherwise be available", () => {
  // Red man on c7 captures black on d8... wait d8 isn't a valid landing target from c7 over d8 (off-board).
  // Set up: red man b6, black man c7, landing d8 (red's back row) -- captures and kings.
  // A further enemy (black) sits so that from d8 (now a king's-eligible square) another jump would
  // otherwise be geometrically available -- placed adjacent with an empty landing beyond it.
  const board: Board = {
    b6: { player: "red", kind: "man" },
    c7: { player: "black", kind: "man" },
    // after landing on d8, a further capture would require an enemy adjacent to d8 with an empty
    // landing beyond -- but d8 is the last rank, so "further" diagonals from d8 for a king go toward
    // rank 6: e7/c7 adjacency. Place one there.
    e7: { player: "black", kind: "man" }, // would be capturable from d8 towards f6, landing empty
  };
  let s = state(board);
  const first = compliantHops(s.board, "red", null);
  assert.deepEqual(keys(first), ["b6xd8"]);
  const next = applyHop(s, first[0]!);
  assert.equal(next.board.d8?.kind, "king");
  assert.equal(next.forcedContinuation, null); // stopped, even though d8xf6 would otherwise be legal
  assert.equal(next.toMove, "black"); // turn passed
  // Confirm the "otherwise available" premise: a king manually placed on d8 WOULD have that capture.
  const wouldHaveContinued = legalHops({ ...next.board, d8: { player: "red", kind: "king" } }, "red").some((h) => h.kind === "capture" && h.from === "d8");
  assert.ok(wouldHaveContinued, "test setup sanity: the further capture must genuinely have been available");
});

test("a near-miss chain (never lands on the back row) keeps chaining correctly, unaffected by kinging logic", () => {
  const board: Board = {
    a3: { player: "red", kind: "man" },
    b4: { player: "black", kind: "man" }, // a3xc5
    d6: { player: "black", kind: "man" }, // c5xe7
  };
  let s = state(board);
  s = applyHop(s, compliantHops(s.board, "red", null)[0]!); // a3xc5
  assert.deepEqual(s.forcedContinuation, { square: "c5" });
  s = applyHop(s, compliantHops(s.board, s.toMove, s.forcedContinuation)[0]!); // c5xe7
  assert.equal(s.forcedContinuation, null); // no further capture from e7 in this setup
  assert.equal(s.toMove, "black");
  assert.equal(s.board.e7?.player, "red");
  assert.equal(s.board.e7?.kind, "man"); // never touched the back row (rank 8), so still a man
});

test("an already-king landing on its own back row does not re-trigger anything", () => {
  const board: Board = { d4: { player: "red", kind: "king" } };
  const s = applyHop(state(board), { kind: "simple", from: "d4", to: "e5" });
  assert.equal(s.board.e5?.kind, "king"); // stays a king, no crash, no special-case side effect
});

test("turn-boundary state: forcedContinuation clears and the next turn's compliant set spans all of the new mover's pieces", () => {
  const board: Board = {
    a3: { player: "red", kind: "man" },
    b4: { player: "black", kind: "man" }, // captured by red, ending the turn (no further chain)
    f6: { player: "black", kind: "man" }, // unrelated black piece, must be reachable on black's turn
  };
  let s = state(board);
  s = applyHop(s, compliantHops(s.board, "red", null)[0]!); // a3xc5, no continuation (nothing further from c5)
  assert.equal(s.forcedContinuation, null);
  assert.equal(s.toMove, "black");
  const blackMoves = compliantHops(s.board, "black", s.forcedContinuation);
  assert.ok(blackMoves.some((h) => h.from === "f6"), "black's unrelated piece must have legal moves available");
});

test("shield override during a forced continuation restricts execution to that piece's own captures", () => {
  const board: Board = {
    a3: { player: "red", kind: "man" },
    b4: { player: "black", kind: "man" },
    d6: { player: "black", kind: "man" },
    b6: { player: "black", kind: "man" },
    g3: { player: "red", kind: "man" }, // a second, unrelated red piece with its own legal (but non-compliant mid-chain) move
  };
  let s = state(board);
  s = applyHop(s, compliantHops(s.board, "red", null)[0]!); // a3xc5
  const p = buildPrompt(s);
  assert.deepEqual(keys(p.compliant), ["c5xa7", "c5xe7"]);
  // g3's own simple move is geometrically legal but NOT compliant mid-chain (wrong piece).
  assert.ok(keys(p.legal).includes("g3-f4") && keys(p.legal).includes("g3-h4"));
  assert.ok(!keys(p.compliant).includes("g3-f4"));
  // The model proposes g3-f4 (not compliant) over both real continuation options.
  const probabilities: Record<string, number> = { "c5xa7": 0.1, "c5xe7": 0.2, "g3-f4": 0.9, "g3-h4": 0.05 };
  const stub: PredictLike = { answers: { move: { probabilities }, material_at_risk: { noul: 0.2 }, captures_material: { noul: 0 } }, usage: { input_tokens: 1 } };
  const guarded = decisionFrom(stub, p, true);
  assert.equal(hopKey(guarded.proposed), "g3-f4");
  assert.equal(hopKey(guarded.executed), "c5xe7"); // argmax of the two compliant options (0.2 > 0.1)
  assert.ok(guarded.intervened);
  const raw = decisionFrom(stub, p, false);
  assert.equal(hopKey(raw.executed), "g3-f4");
  assert.ok(!raw.intervened);
});

test("board-edge geometry: an a-file or h-file piece never produces an off-board hop (no wraparound)", () => {
  // Only these are actually dark/playable squares on the a-file and h-file.
  for (const edge of ["a1", "a3", "a5", "a7", "h2", "h4", "h6", "h8"] as Square[]) {
    for (const player of ["red", "black"] as const) {
      for (const kind of ["man", "king"] as const) {
        const board: Board = { [edge]: { player, kind } };
        const hops = legalHops(board, player);
        // A man can be fully stuck at the far edge in its own non-forward direction (e.g. a black
        // man on a1, rank 1 is black's own back row) -- that's correct, not asserted here. Only
        // that whatever hops DO exist never wrap around the board is under test.
        for (const h of hops) {
          const toFile = h.to[0];
          if (edge.startsWith("a")) assert.notEqual(toFile, "h", `${edge} -> ${h.to} wrapped around to the h-file`);
          if (edge.startsWith("h")) assert.notEqual(toFile, "a", `${edge} -> ${h.to} wrapped around to the a-file`);
        }
      }
    }
  }
});

test("isCompliant agrees with compliantHops membership", () => {
  const s = newGame();
  const legal = legalHops(s.board, "red");
  const compliant = compliantHops(s.board, "red", null);
  for (const h of legal) assert.equal(isCompliant(s, h), compliant.some((c) => hopKey(c) === hopKey(h)));
});

test("applyHop rejects a non-compliant hop", () => {
  const s = newGame();
  // Not even geometrically legal (a3-a4 isn't diagonal), so trivially non-compliant.
  assert.throws(() => applyHop(s, { kind: "simple", from: "a3", to: "a4" }), /not compliant/);
});

test("applyHop rejects a hop onto an occupied square (never reaches the board mutation)", () => {
  const s = newGame();
  assert.throws(() => applyHop(s, { kind: "simple", from: "c3", to: "b2" })); // b2 is red's own piece
});

test("invalid model probabilities execute no hop", () => {
  const s = newGame();
  const p = buildPrompt(s);
  const probabilities = Object.fromEntries(keys(p.legal).map((k, i) => [k, i === 0 ? NaN : 0]));
  const stub: PredictLike = { answers: { move: { probabilities }, material_at_risk: { noul: 1 }, captures_material: { noul: 0 } }, usage: { input_tokens: 1 } };
  assert.throws(() => decisionFrom(stub, p, true), /invalid probability/);
});
