/** Port of laya-mlx tests/test_snake.py (game + shield) plus a Python-RNG replay check. */
import assert from "node:assert/strict";
import { DIRECTIONS, PyRandom, SnakeGame, hamiltonianCycle, LayaPolicy, buildPrompt, type Direction, type PredictLike } from "../src/core/index.ts";
import { makeTest, readFrames } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const test = makeTest(bun);

for (const [w, h] of [[4, 4], [4, 5], [5, 4], [24, 16]] as const) {
  test(`cycle visits every cell and closes ${w}x${h}`, () => {
    const cycle = hamiltonianCycle(w, h);
    assert.equal(new Set(cycle.map(([x, y]) => `${x},${y}`)).size, w * h);
    assert.equal(cycle.length, w * h);
    assert.ok(cycle.every(([x, y]) => x >= 0 && x < w && y >= 0 && y < h));
    cycle.forEach((a, i) => {
      const b = cycle[(i + 1) % cycle.length]!;
      assert.equal(Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]), 1);
    });
  });
}

test("collision, growth, tail vacancy and board clear", () => {
  const game = new SnakeGame(4, 4, 7, 4);
  game.body = [[1, 1], [1, 2], [0, 2], [0, 1]];
  game.food = [3, 3];
  assert.equal(game.legalReason("LEFT"), "legal");
  assert.equal(game.legalReason("DOWN"), "reverse");
  game.step("LEFT");
  assert.ok(game.alive);
  assert.equal(game.body.length, 4);
  game.step("LEFT");
  assert.ok(!game.alive);
  assert.equal(game.deathReason, "wall");

  const full = new SnakeGame(4, 4, 7, 15);
  const move = full.moves().find((m) => m.safe)!;
  assert.ok(full.step(move.direction));
  assert.ok(full.won && full.food === null && full.body.length === 16 && full.score === 1);
  assert.deepEqual(full.moves(), []);
});

test("arbitrary shielded choices complete the board without starving (12 seeds)", () => {
  for (let seed = 0; seed < 12; seed++) {
    const game = new SnakeGame(6, 6, seed);
    const rng = new PyRandom(seed + 100);
    let lastFood = 0;
    for (let i = 0; i < game.capacity * (game.capacity - game.initialLength); i++) {
      const allowed = game.moves().filter((m) => m.safe).map((m) => m.direction);
      assert.ok(allowed.length);
      const ate = game.step(rng.choice(allowed));
      assert.ok(game.alive && game.cycleOrderValid());
      assert.equal(new Set(game.body.map(([x, y]) => `${x},${y}`)).size, game.body.length);
      assert.equal(game.body.length, game.initialLength + game.score);
      assert.ok(game.ticks - lastFood <= game.capacity);
      if (ate) lastFood = game.ticks;
      if (game.won) break;
    }
    assert.ok(game.won, `seed ${seed}`);
  }
});

test("seed reproduces foods and actions", () => {
  const first = new SnakeGame(24, 16, 71);
  const second = new SnakeGame(24, 16, 71);
  for (let i = 0; i < 100; i++) {
    const d = first.moves().filter((m) => m.safe).reduce((b, m) => (m.advance > b.advance ? m : b)).direction;
    first.step(d);
    second.step(d);
    assert.deepEqual(first.snapshot(), second.snapshot());
  }
});

test("guard preserves raw probabilities and reports intervention", async () => {
  const game = new SnakeGame();
  const safe = game.moves().filter((m) => m.safe).map((m) => m.direction);
  const unsafe = DIRECTIONS.find((d) => !safe.includes(d))!;
  const probabilities = Object.fromEntries(DIRECTIONS.map((d) => [d, d === unsafe ? 0.9 : 0.1 / 3])) as Record<Direction, number>;
  const stub = {
    predict: (): PredictLike => ({
      answers: { move: { probabilities }, risk: { noul: 0.8 }, food: { noul: 0.7 } },
      usage: { input_tokens: 1, output_tokens: 0 },
    }),
  };
  const guarded = await new LayaPolicy(stub).decide(game);
  assert.equal(guarded.proposed, unsafe);
  assert.ok(safe.includes(guarded.executed));
  assert.ok(guarded.intervened);
  assert.deepEqual(guarded.probabilities, probabilities);
  assert.ok(Math.abs(guarded.dead_end_risk - 0.2) < 1e-12);
  const raw = await new LayaPolicy(stub, { guarded: false }).decide(game);
  assert.equal(raw.executed, unsafe);
  assert.ok(!raw.intervened);
});

test("invalid model probabilities execute no move", async () => {
  const game = new SnakeGame();
  const stub = {
    predict: (): PredictLike => ({
      answers: { move: { probabilities: { UP: NaN, DOWN: 0, LEFT: 0, RIGHT: 1 } }, risk: { noul: 1 }, food: { noul: 1 } },
      usage: { input_tokens: 1 },
    }),
  };
  await assert.rejects(new LayaPolicy(stub).decide(game), /invalid probability/);
});

test("replaying Python's recorded moves reproduces every recorded board (game + Python RNG)", async () => {
  const { meta, frames } = await readFrames();
  const s = (meta as { settings: { width: number; height: number; seed: number; initial_length: number } }).settings;
  const game = new SnakeGame(s.width, s.height, s.seed, s.initial_length);
  for (const f of frames) {
    assert.deepEqual(game.snapshot(), f.game, `tick ${f.game.ticks}`);
    game.step(f.decision.executed);
  }
});

test("planner features match Python on recorded frames (safe set, planner best)", async () => {
  const { frames } = await readFrames();
  for (const f of frames) {
    const p = buildPrompt(SnakeGame.fromSnapshot(f.game));
    assert.deepEqual(p.safe.map((m) => m.direction), f.decision.safe_directions);
    assert.equal(p.preferred, f.decision.planner_best);
    assert.equal(p.safe.length, f.decision.safe_count);
  }
});
