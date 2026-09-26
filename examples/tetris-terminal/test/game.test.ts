/**
 * Pure-logic tests for the Tetris core: piece shapes, legal-placement
 * generation and bounds, line clears, the safety shield (now scoped to
 * the lock decision, local to the current row), rotation resolution,
 * spawn legality, and the 7-bag randomizer. No Python reference (see
 * rng.ts) -- no parity tier.
 */
import assert from "node:assert/strict";
import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  DISTINCT_ROTATIONS,
  PIECE_KINDS,
  SPAWN_ROW,
  TetrisGame,
  TetrisRng,
  applyPlacement,
  buildStepPrompt,
  canSpawn,
  decisionFromStep,
  emptyBoard,
  fitsAtColumn,
  legalPlacements,
  optionsAtRow,
  placementKey,
  resolveRotation,
  shapeOf,
  spawnColFor,
  stackHeight,
  sweepColumns,
  type PieceKind,
  type Placement,
  type PredictLike,
  type RotationLabel,
  type StepPosition,
} from "../src/core/index.ts";
import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const test = makeTest(bun);

function oneHot(keys: string[], winner: string): Record<string, number> {
  return Object.fromEntries(keys.map((k) => [k, k === winner ? 0.9 : 0.1 / Math.max(1, keys.length - 1)]));
}

/** A stubbed model answer for one gravity-step decision (isLockChance adds risk/clears). */
function stubStep(choice: { rotation: RotationLabel; direction: "none" | "left" | "right"; distance: number; risk?: number; clears?: number }): PredictLike {
  const answers: PredictLike["answers"] = {
    rotation: { probabilities: oneHot(["0", "R", "2", "L"], choice.rotation) },
    direction: { probabilities: oneHot(["none", "left", "right"], choice.direction) },
    distance: { probabilities: oneHot(Array.from({ length: 10 }, (_, n) => String(n)), String(choice.distance)) },
  };
  if (choice.risk !== undefined) answers.risk = { noul: choice.risk };
  if (choice.clears !== undefined) answers.clears = { noul: choice.clears };
  return { answers, usage: { input_tokens: 1 } };
}

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

test("resolveRotation always resolves to a member of DISTINCT_ROTATIONS, and geometrically-equivalent labels map onto each other", () => {
  for (const kind of PIECE_KINDS) {
    for (const label of ["0", "R", "2", "L"] as RotationLabel[]) {
      assert.ok(DISTINCT_ROTATIONS[kind].includes(resolveRotation(kind, label)), `${kind} ${label}`);
    }
  }
  // O: 1-fold symmetry -- every label resolves to "0".
  assert.equal(resolveRotation("O", "R"), "0");
  assert.equal(resolveRotation("O", "2"), "0");
  assert.equal(resolveRotation("O", "L"), "0");
  // I/S/Z: 2-fold symmetry -- 180 degrees maps back to 0, 270 maps back to 90.
  assert.equal(resolveRotation("I", "2"), "0");
  assert.equal(resolveRotation("I", "L"), "R");
  assert.equal(resolveRotation("S", "2"), "0");
  assert.equal(resolveRotation("Z", "L"), "R");
  // T/J/L: all 4 distinct -- identity.
  assert.equal(resolveRotation("T", "L"), "L");
});

test("spawnColFor centers each kind's default orientation on the board", () => {
  assert.equal(spawnColFor("I"), 3); // occupies cols 3-6, matching real Tetris' own centered I spawn
  assert.equal(spawnColFor("O"), 4); // occupies cols 4-5
  for (const kind of PIECE_KINDS) assert.ok(fitsAtColumn(kind, "0", spawnColFor(kind)), `${kind}'s spawn column must fit on the board`);
});

// ---------------------------------------------------------------- legalPlacements bounds (still a standalone, tested utility -- gameplay no longer calls it)
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

test("fitsAtColumn agrees with the board bounds a wider/narrower rotation would actually need", () => {
  assert.equal(fitsAtColumn("I", "0", 9), false); // horizontal I is 4 wide, can't start at the last column
  assert.equal(fitsAtColumn("I", "0", 6), true); // its rightmost valid horizontal start
  assert.equal(fitsAtColumn("I", "R", 9), true); // vertical I is 1 wide, fits anywhere
  assert.equal(fitsAtColumn("O", "0", 8), true); // O's rightmost valid column
  assert.equal(fitsAtColumn("O", "0", 9), false); // O is 2 wide
  assert.equal(fitsAtColumn("T", "0", -1), false); // never valid off the left edge either
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

// ---------------------------------------------------------------- optionsAtRow (the lock-time shield's local evaluation set)
test("optionsAtRow finds every option unsafe when the stack is already critically tall, without the game being over", () => {
  const board = emptyBoard();
  // Solid floor at row 3, rows 0-2 open. Column 9 stays empty throughout so no row is ever
  // completely full (a fully-filled row would instantly clear in real play).
  for (let r = 3; r < BOARD_HEIGHT; r++) for (let c = 0; c < BOARD_WIDTH - 1; c++) board[r]![c] = "T";
  const [placement] = legalPlacements(board, "O"); // where a real drop from spawn would land
  const options = optionsAtRow(board, "O", placement!.restRow, placement!.col);
  assert.ok(options.length > 0); // plenty of room to physically rest
  assert.ok(options.every((o) => !o.safe)); // but every one breaches TOP_MARGIN
  assert.equal(canSpawn(board, "O"), true); // NOT game over -- see policy.ts's comment on this exact asymmetry
});

test("optionsAtRow finds every option safe on a mostly empty board", () => {
  const board = emptyBoard();
  for (let c = 0; c < BOARD_WIDTH - 1; c++) board[BOARD_HEIGHT - 1]![c] = "T"; // one row, one gap
  const [placement] = legalPlacements(board, "O");
  const options = optionsAtRow(board, "O", placement!.restRow, placement!.col);
  assert.ok(options.every((o) => o.safe)); // an almost-empty board is nowhere near the margin
});

test("optionsAtRow only includes columns actually reachable by sliding from the current column, not any resting spot on the same row (regression: a naive full-row scan would jump a wall to an unreachable shelf)", () => {
  const board = emptyBoard();
  // A full-height wall at column 4 splits the board; a floor a few rows down on both sides
  // gives the left shelf (cols 0-3) and the right shelf (cols 5-9) the SAME resting row.
  for (let r = 0; r < BOARD_HEIGHT; r++) board[r]![4] = "T";
  const floorStart = BOARD_HEIGHT - 3;
  for (let c = 0; c < BOARD_WIDTH; c++) if (c !== 4) for (let r = floorStart; r < BOARD_HEIGHT; r++) board[r]![c] = "T";
  const restRow = floorStart - 2;
  const fromLeft = optionsAtRow(board, "O", restRow, 0).map((o) => o.placement.col).sort((a, b) => a - b);
  assert.deepEqual(fromLeft, [0, 1, 2]); // never jumps the wall to reach the right shelf (5-8)
  // The right shelf genuinely is legal-and-resting at this same row -- just unreachable from col 0.
  const fromRight = optionsAtRow(board, "O", restRow, 5).map((o) => o.placement.col).sort((a, b) => a - b);
  assert.deepEqual(fromRight, [5, 6, 7, 8]);
});

test("sweepColumns always includes the starting column and stops at the first collision each way", () => {
  const board = emptyBoard();
  for (let r = 0; r < BOARD_HEIGHT; r++) board[r]![3] = "T"; // wall at col 3
  const cols = sweepColumns(board, shapeOf("O", "0"), 0, 5);
  assert.ok(cols.includes(5));
  assert.ok(!cols.some((c) => c <= 2)); // O is 2 wide -- col 2 would occupy col 3 (the wall)
});

// ---------------------------------------------------------------- game over (block-out)
test("legalPlacements is empty when the spawn row is fully occupied, for every piece kind (still true of this standalone utility)", () => {
  const board = emptyBoard();
  for (let c = 0; c < BOARD_WIDTH; c++) board[0]![c] = "T";
  for (const kind of PIECE_KINDS) assert.deepEqual(legalPlacements(board, kind), []);
});

test("canSpawn is false exactly when the piece's fixed default spawn configuration collides", () => {
  const board = emptyBoard();
  assert.equal(canSpawn(board, "O"), true);
  for (const [dr, dc] of shapeOf("O", "0")) board[dr]![spawnColFor("O") + dc] = "T";
  assert.equal(canSpawn(board, "O"), false);
});

test("a real, non-clearing placement can lead to genuine block-out (game over)", () => {
  const game = new TetrisGame(1);
  const board = emptyBoard();
  // Column 9 is a permanent well, left empty at every row, so no row is ever completely full
  // (a 100%-full row is an invalid, unreachable state). Row 0: cols 0-3 and col 8 filled, cols
  // 4-7 open (exactly one horizontal I fits), col 9 open.
  for (const c of [0, 1, 2, 3, 8]) board[0]![c] = "T";
  for (let r = 1; r < BOARD_HEIGHT; r++) for (let c = 0; c < BOARD_WIDTH - 1; c++) board[r]![c] = "T"; // solid floor everywhere else, except the col-9 well
  game.board = board;
  game.active = "I";
  game.queue = ["O", "O", "O", "O", "O"];
  const horizontal = game.legalPlacements().filter((p) => p.rotation === "0");
  assert.equal(horizontal.length, 1); // only the 4-wide gap at cols 4-7 fits
  assert.equal(horizontal[0]!.col, 4);
  const cleared = game.applyPlacement(horizontal[0]!);
  assert.equal(cleared, 0); // col 9 stays open -- row 0 is NOT completed, so no rescuing clear
  assert.equal(game.active, "O");
  // Locking the I fills row 0's cols 4-7 too, so O's default spawn footprint (row 0-1, cols 4-5)
  // now collides at row 0 -- real block-out via the literal spawn configuration, not just
  // "no rotation/column fits anywhere" (col 9 alone could never fit an O regardless).
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

// ---------------------------------------------------------------- per-step decisions and the lock-time shield
test("rotation is checked against the CURRENT column before any shift -- an illegal rotation is rejected even if the shift alone would have made room (no wall-kick, a deliberate simplification)", () => {
  const game = new TetrisGame(1);
  const board = emptyBoard();
  for (let r = 0; r < BOARD_HEIGHT; r++) board[r]![2] = "T"; // full-height wall at col 2
  game.board = board;
  game.active = "I";
  const position: StepPosition = { row: 5, rotation: "R", col: 0 }; // vertical I, clear of the wall
  const p = buildStepPrompt(game, position, false);
  // Model asks to rotate to horizontal (illegal here -- the wall at col 2 is inside cols 0-3) AND shift right by 1.
  const stub = stubStep({ rotation: "0", direction: "right", distance: 1 });
  const result = decisionFromStep(stub, p, board, true);
  assert.equal(result.position.rotation, "R"); // rejected -- illegal at the CURRENT column (0)
  assert.equal(result.position.col, 1); // the shift still applies, using the KEPT rotation's shape
  assert.equal(result.locked, false);
});

test("guarded lock overrides an unsafe proposed choice to a safer local alternative reachable via rotate+shift", () => {
  const game = new TetrisGame(1);
  const board = emptyBoard();
  for (let r = 7; r < BOARD_HEIGHT; r++) for (let c = 0; c < BOARD_WIDTH - 1; c++) board[r]![c] = "T"; // deep floor, col 9 a permanent well
  for (let c = 0; c < BOARD_WIDTH; c++) if (c !== 6) board[4]![c] = "T"; // row 4 nearly full except col 6
  game.board = board;
  game.active = "I";
  const position: StepPosition = { row: 3, rotation: "0", col: 6 }; // horizontal I, resting, unsafe
  const p = buildStepPrompt(game, position, true);
  // Model proposes staying put (no rotation change, no shift) -- unsafe (doesn't complete row 4).
  const stub = stubStep({ rotation: "0", direction: "none", distance: 0, risk: 0.9, clears: 0 });
  const guarded = decisionFromStep(stub, p, board, true);
  assert.deepEqual(guarded.proposed, { kind: "I", rotation: "0", col: 6, restRow: 3 });
  // Rotating to vertical at this SAME column completes row 4 -- the shield finds and executes it.
  assert.deepEqual(guarded.executed, { kind: "I", rotation: "R", col: 6, restRow: 3 });
  assert.ok(guarded.intervened);
  const raw = decisionFromStep(stub, p, board, false);
  assert.deepEqual(raw.executed, raw.proposed);
  assert.ok(!raw.intervened);
});

test("guarded lock keeps the model's own proposed choice when no local option is safe (routine near the top of a real game, not rare-and-terminal)", () => {
  const game = new TetrisGame(1);
  const board = emptyBoard();
  for (let r = 3; r < BOARD_HEIGHT; r++) for (let c = 0; c < BOARD_WIDTH - 1; c++) board[r]![c] = "T";
  game.board = board;
  game.active = "O";
  const [placement] = legalPlacements(board, "O");
  const position: StepPosition = { row: placement!.restRow, rotation: "0", col: placement!.col };
  const p = buildStepPrompt(game, position, true);
  const stub = stubStep({ rotation: "0", direction: "none", distance: 0, risk: 0.95, clears: 0 });
  const guarded = decisionFromStep(stub, p, board, true);
  assert.deepEqual(guarded.executed, guarded.proposed);
  assert.ok(!guarded.intervened);
});

test("invalid model probabilities throw; no step executed", () => {
  const game = new TetrisGame(1);
  const position: StepPosition = { row: SPAWN_ROW, rotation: "0", col: spawnColFor(game.active) };
  const p = buildStepPrompt(game, position, false);
  const stub = stubStep({ rotation: "0", direction: "none", distance: 0 });
  stub.answers.rotation!.probabilities!["0"] = NaN;
  assert.throws(() => decisionFromStep(stub, p, game.board, true), /invalid probability/);
});

// ---------------------------------------------------------------- full-game smoke play
/** `TOP_MARGIN` isn't exported (only `optionsAtRow`/`canSpawn` need it internally now) -- this
 * reconstructs the old `moves()`'s global decoration directly from the still-available
 * `legalPlacements`/`applyPlacement`/`stackHeight`, using the same `<= BOARD_HEIGHT - 4` margin,
 * for a smoke test that (deliberately) picks among the FULL reachable set, not just one row's
 * local options -- unlike the real per-step engine, this heuristic can freely choose ANY
 * placement each piece, so it isn't a stand-in for the shield, just a cheap crash/sanity check. */
function safeGlobalMoves(board: ReturnType<TetrisGame["snapshot"]>["board"], kind: PieceKind) {
  return legalPlacements(board, kind).map((placement) => {
    const { board: nextBoard, cleared } = applyPlacement(board, placement);
    return { placement, clears: cleared, heightAfter: stackHeight(nextBoard), safe: stackHeight(nextBoard) <= BOARD_HEIGHT - 4 };
  });
}

test("a simple lowest-height heuristic survives many pieces across several seeds without crashing", () => {
  for (let seed = 0; seed < 6; seed++) {
    const game = new TetrisGame(seed);
    let n = 0;
    while (game.alive && n < 300) {
      const moves = safeGlobalMoves(game.board, game.active);
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
