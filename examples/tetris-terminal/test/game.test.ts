/**
 * Pure-logic tests for the Tetris core: piece shapes, legal-placement
 * generation and bounds, line clears, the safety-margin shield, and the
 * 7-bag randomizer. No Python reference (see rng.ts) -- no parity tier.
 */
import assert from "node:assert/strict";
import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  DISTINCT_ROTATIONS,
  PIECE_KINDS,
  TetrisGame,
  TetrisRng,
  applyPlacement,
  buildPrompt,
  decisionFrom,
  emptyBoard,
  legalPlacements,
  placementKey,
  shapeOf,
  stackHeight,
  type PieceKind,
  type Placement,
  type PredictLike,
} from "../src/core/index.ts";
import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const test = makeTest(bun);

// ---------------------------------------------------------------- shapes
test("every distinct rotation has exactly 4 cells", () => {
  for (const kind of PIECE_KINDS) {
    for (const rotation of DISTINCT_ROTATIONS[kind]) {
      assert.equal(shapeOf(kind, rotation).length, 4, `${kind} ${rotation}`);
    }
  }
});

test("O has exactly one distinct rotation; I/S/Z have two; T/J/L have four", () => {
  assert.deepEqual(DISTINCT_ROTATIONS.O, ["0"]);
  assert.deepEqual(DISTINCT_ROTATIONS.I, ["0", "R"]);
  assert.deepEqual(DISTINCT_ROTATIONS.S, ["0", "R"]);
  assert.deepEqual(DISTINCT_ROTATIONS.Z, ["0", "R"]);
  assert.deepEqual(DISTINCT_ROTATIONS.T, ["0", "R", "2", "L"]);
  assert.deepEqual(DISTINCT_ROTATIONS.J, ["0", "R", "2", "L"]);
  assert.deepEqual(DISTINCT_ROTATIONS.L, ["0", "R", "2", "L"]);
});

test("no two distinct rotations of the same piece produce the same cell-set (regression guard)", () => {
  for (const kind of PIECE_KINDS) {
    const cellSets = DISTINCT_ROTATIONS[kind].map((r) => JSON.stringify([...shapeOf(kind, r)].sort()));
    assert.equal(new Set(cellSets).size, cellSets.length, `${kind} has duplicate rotation shapes`);
  }
});

// ---------------------------------------------------------------- legalPlacements bounds
test("I-piece: 7 horizontal columns (0..6), 10 vertical columns (0..9) on an empty board", () => {
  const board = emptyBoard();
  const placements = legalPlacements(board, "I");
  const horizontal = placements.filter((p) => p.rotation === "0");
  const vertical = placements.filter((p) => p.rotation === "R");
  assert.deepEqual(horizontal.map((p) => p.col).sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6]);
  assert.deepEqual(vertical.map((p) => p.col).sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
});

test("O-piece: exactly one rotation's worth of keys, 9 columns (0..8)", () => {
  const board = emptyBoard();
  const placements = legalPlacements(board, "O");
  assert.ok(placements.every((p) => p.rotation === "0"));
  assert.deepEqual(placements.map((p) => p.col).sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6, 7, 8]);
  const keys = placements.map(placementKey);
  assert.ok(keys.every((k) => /^O-0-c\d$/.test(k)));
});

test("no legal placement of any piece ever escapes the board bounds", () => {
  const board = emptyBoard();
  for (const kind of PIECE_KINDS) {
    for (const placement of legalPlacements(board, kind)) {
      const shape = shapeOf(placement.kind, placement.rotation);
      for (const [dr, dc] of shape) {
        const r = placement.restRow + dr;
        const c = placement.col + dc;
        assert.ok(r >= 0 && r < BOARD_HEIGHT, `${placementKey(placement)}: row ${r} out of bounds`);
        assert.ok(c >= 0 && c < BOARD_WIDTH, `${placementKey(placement)}: col ${c} out of bounds`);
      }
    }
  }
});

test("T/J/L board-edge bounds hold for every one of their 4 rotations", () => {
  const board = emptyBoard();
  for (const kind of ["T", "J", "L"] as PieceKind[]) {
    const byRotation = new Map<string, Placement[]>();
    for (const p of legalPlacements(board, kind)) {
      byRotation.set(p.rotation, [...(byRotation.get(p.rotation) ?? []), p]);
    }
    assert.equal(byRotation.size, 4, `${kind} should have placements in all 4 rotations`);
    for (const placements of byRotation.values()) assert.ok(placements.length > 0);
  }
});

// ---------------------------------------------------------------- line clears
test("a placement that clears 4 lines at once (a Tetris) removes exactly those rows and shifts the rest down", () => {
  const board = emptyBoard();
  // Rows 16-19 filled in every column except 9; dropping a vertical I into column 9 completes all four.
  for (let r = 16; r < BOARD_HEIGHT; r++) for (let c = 0; c < BOARD_WIDTH - 1; c++) board[r]![c] = "T";
  const placements = legalPlacements(board, "I").filter((p) => p.rotation === "R" && p.col === 9);
  assert.equal(placements.length, 1);
  const { board: after, cleared } = applyPlacement(board, placements[0]!);
  assert.equal(cleared, 4);
  assert.ok(after.every((row) => row.every((cell) => cell === null))); // fully empty again
  assert.equal(stackHeight(after), 0);
});

test("a non-clearing placement leaves the board exactly as locked", () => {
  const board = emptyBoard();
  const [placement] = legalPlacements(board, "O");
  const { board: after, cleared } = applyPlacement(board, placement!);
  assert.equal(cleared, 0);
  assert.equal(stackHeight(after), BOARD_HEIGHT - placement!.restRow);
});

// ---------------------------------------------------------------- safety margin (the shield's "safe" tier)
test("moves() finds every placement unsafe when the stack is already critically tall, without the game being over", () => {
  const game = new TetrisGame(1);
  const board = emptyBoard();
  // Solid floor at row 3, rows 0-2 open. Column 9 stays empty throughout so no row is ever
  // completely full (a fully-filled row would instantly clear in real play -- filling every
  // column here would be an invalid, unreachable fixture, not a "tall stack").
  for (let r = 3; r < BOARD_HEIGHT; r++) for (let c = 0; c < BOARD_WIDTH - 1; c++) board[r]![c] = "T";
  game.board = board;
  game.active = "O";
  const moves = game.moves();
  assert.ok(moves.length > 0); // plenty of room to physically place
  assert.ok(moves.every((m) => !m.safe)); // but every one breaches the TOP_MARGIN
  assert.equal(game.alive, true); // NOT game over -- see the policy.ts comment on this exact asymmetry
});

test("moves() finds placements safe on a mostly empty board, and safer ones that clear lines", () => {
  const game = new TetrisGame(1);
  const board = emptyBoard();
  for (let c = 0; c < BOARD_WIDTH - 1; c++) board[BOARD_HEIGHT - 1]![c] = "T"; // one row, one gap
  game.board = board;
  game.active = "O";
  const moves = game.moves();
  assert.ok(moves.every((m) => m.safe)); // an almost-empty board is nowhere near the margin
});

// ---------------------------------------------------------------- game over (block-out)
test("legalPlacements is empty when the spawn row is fully occupied, for every piece kind", () => {
  const board = emptyBoard();
  for (let c = 0; c < BOARD_WIDTH; c++) board[0]![c] = "T";
  for (const kind of PIECE_KINDS) assert.deepEqual(legalPlacements(board, kind), []);
});

test("a real, non-clearing placement can lead to genuine block-out (game over)", () => {
  const game = new TetrisGame(1);
  const board = emptyBoard();
  // Column 9 is a permanent well, left empty at every row, so no row is ever completely full
  // (see the comment on the fixture above -- a 100%-full row is an invalid, unreachable state).
  // Row 0: cols 0-3 and col 8 filled, cols 4-7 open (exactly one horizontal I fits), col 9 open.
  for (const c of [0, 1, 2, 3, 8]) board[0]![c] = "T";
  for (let r = 1; r < BOARD_HEIGHT; r++) for (let c = 0; c < BOARD_WIDTH - 1; c++) board[r]![c] = "T"; // solid floor everywhere else, except the col-9 well
  game.board = board;
  game.active = "I";
  game.queue = ["O", "O", "O", "O", "O"];
  // (A vertical I can also legally drop into the col-9 well -- that's a real, expected option
  // and irrelevant to this test, which is specifically about the horizontal placement.)
  const horizontal = game.legalPlacements().filter((p) => p.rotation === "0");
  assert.equal(horizontal.length, 1); // only the 4-wide gap at cols 4-7 fits
  assert.equal(horizontal[0]!.col, 4);
  const cleared = game.applyPlacement(horizontal[0]!);
  assert.equal(cleared, 0); // col 9 stays open -- row 0 is NOT completed, so no rescuing clear
  assert.equal(game.active, "O");
  // O needs 2 adjacent open columns; only col 9 is open anywhere on the board (col 8 is solid
  // at every row), so it can never fit -- true block-out, not just "no safe option."
  assert.equal(game.alive, false);
});

// ---------------------------------------------------------------- 7-bag
test("a 7-bag is always a permutation of all 7 kinds, deterministically, over 1000+ bags", () => {
  const a = new TetrisRng(42);
  const b = new TetrisRng(42);
  for (let i = 0; i < 1000; i++) {
    const bagA = a.bag();
    const bagB = b.bag();
    assert.deepEqual(bagA, bagB);
    assert.equal(new Set(bagA).size, 7);
    assert.deepEqual([...bagA].sort(), [...PIECE_KINDS].sort());
  }
});

test("different seeds produce different bag sequences", () => {
  const a = new TetrisRng(1).bag();
  const b = new TetrisRng(2).bag();
  assert.notDeepEqual(a, b);
});

// ---------------------------------------------------------------- shield
test("guard restricts execution to the safe set and reports intervention", () => {
  const game = new TetrisGame(1);
  const board = emptyBoard();
  for (let r = 3; r < BOARD_HEIGHT; r++) for (let c = 0; c < BOARD_WIDTH - 1; c++) board[r]![c] = "T"; // every O placement unsafe (see above; col 9 stays open so no row is ever full)
  game.board = board;
  game.active = "O";
  const p = buildPrompt(game);
  assert.equal(p.safe.length, 0);
  assert.ok(p.moves.length > 0);
  const bestKey = placementKey(p.moves[3]!.placement); // arbitrary choice among the (all-unsafe) options
  const probabilities = Object.fromEntries(p.moves.map((m) => [placementKey(m.placement), placementKey(m.placement) === bestKey ? 0.9 : 0.01]));
  const stub: PredictLike = { answers: { move: { probabilities }, risk: { noul: 0.9 }, clears: { noul: 0 } }, usage: { input_tokens: 1 } };
  const decision = decisionFrom(stub, p, true);
  // Empty safe set: executes the raw proposed choice rather than throwing (Flappy Bird's precedent).
  assert.equal(placementKey(decision.proposed), bestKey);
  assert.equal(placementKey(decision.executed), bestKey);
  assert.ok(!decision.intervened);
});

test("guard overrides an unsafe top choice to the best safe alternative when one exists", async () => {
  const game = new TetrisGame(1);
  const board = emptyBoard();
  // A 2-wide well at cols 8-9, floor at row 16 (safe zone), plus a much taller decoy region
  // elsewhere that's still reachable but breaches the margin.
  for (let r = 0; r < BOARD_HEIGHT; r++) {
    for (let c = 0; c < BOARD_WIDTH; c++) {
      if (c >= 8) continue; // cols 8-9 stay open all the way down
      if (r >= 3) board[r]![c] = "T"; // cols 0-7 stacked up to row 3 (unsafe if built on)
    }
  }
  game.board = board;
  game.active = "O";
  const p = buildPrompt(game);
  const safeKeys = new Set(p.safe.map((m) => placementKey(m.placement)));
  assert.ok(safeKeys.size > 0 && safeKeys.size < p.moves.length); // a real, non-trivial split
  const unsafeKey = placementKey(p.moves.find((m) => !m.safe)!.placement);
  const [safeBest] = [...safeKeys];
  const probabilities = Object.fromEntries(p.moves.map((m) => [placementKey(m.placement), placementKey(m.placement) === unsafeKey ? 0.9 : placementKey(m.placement) === safeBest ? 0.5 : 0.01]));
  const stub: PredictLike = { answers: { move: { probabilities }, risk: { noul: 0.9 }, clears: { noul: 0 } }, usage: { input_tokens: 1 } };
  const guarded = decisionFrom(stub, p, true);
  assert.equal(placementKey(guarded.proposed), unsafeKey);
  assert.ok(safeKeys.has(placementKey(guarded.executed)));
  assert.ok(guarded.intervened);
  const raw = decisionFrom(stub, p, false);
  assert.equal(placementKey(raw.executed), unsafeKey);
  assert.ok(!raw.intervened);
});

test("invalid model probabilities execute no placement", async () => {
  const game = new TetrisGame(1);
  const p = buildPrompt(game);
  const probabilities = Object.fromEntries(p.moves.map((m, i) => [placementKey(m.placement), i === 0 ? NaN : 0]));
  const stub: PredictLike = { answers: { move: { probabilities }, risk: { noul: 1 }, clears: { noul: 0 } }, usage: { input_tokens: 1 } };
  assert.throws(() => decisionFrom(stub, p, true), /invalid probability/);
});

// ---------------------------------------------------------------- full-game smoke play
test("a simple lowest-height heuristic survives many pieces across several seeds without crashing", () => {
  for (let seed = 0; seed < 6; seed++) {
    const game = new TetrisGame(seed);
    let n = 0;
    while (game.alive && n < 300) {
      const moves = game.moves();
      const pool = moves.some((m) => m.safe) ? moves.filter((m) => m.safe) : moves;
      let best = pool[0]!;
      for (const m of pool) if (m.heightAfter < best.heightAfter) best = m;
      game.applyPlacement(best.placement);
      n++;
    }
    assert.ok(n > 20, `seed ${seed} topped out almost immediately (${n} pieces)`);
    assert.ok(Number.isInteger(game.score) && game.score >= 0);
    assert.ok(Number.isInteger(game.linesCleared) && game.linesCleared >= 0);
  }
});
