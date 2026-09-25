/** Pure-logic tests for the Flappy Bird core: physics, collision, the lookahead safety classifier, and the shield. No Python reference (see rng.ts) -- no parity tier. */
import assert from "node:assert/strict";
import { ACTIONS, FlappyGame, FlappyRng, LayaPolicy, buildPrompt, type Action, type PredictLike } from "../src/core/index.ts";
import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const test = makeTest(bun);

test("FlappyRng is deterministic for a given seed and varies across seeds", () => {
  const a = new FlappyRng(7);
  const b = new FlappyRng(7);
  const seqA = Array.from({ length: 20 }, () => a.int(0, 100));
  const seqB = Array.from({ length: 20 }, () => b.int(0, 100));
  assert.deepEqual(seqA, seqB);
  const c = new FlappyRng(8);
  const seqC = Array.from({ length: 20 }, () => c.int(0, 100));
  assert.notDeepEqual(seqA, seqC);
});

test("construction seeds pipes beyond the visible board and rejects too-short boards", () => {
  const g = new FlappyGame({ seed: 1 });
  assert.ok(g.pipes.length >= 1);
  assert.ok(g.pipes[g.pipes.length - 1]!.x > g.width);
  assert.ok(g.alive);
  assert.equal(g.birdY, g.height / 2);
  assert.throws(() => new FlappyGame({ height: 5, gapHeight: 6 }), RangeError);
});

test("falling with no flaps eventually hits the ground", () => {
  const g = new FlappyGame({ seed: 1 });
  let n = 0;
  while (g.alive && n < 100) {
    g.step("NOFLAP");
    n++;
  }
  assert.ok(!g.alive);
  assert.equal(g.deathReason, "ground");
});

test("continuous flapping eventually hits the ceiling", () => {
  const g = new FlappyGame({ seed: 1 });
  let n = 0;
  while (g.alive && n < 100) {
    g.step("FLAP");
    n++;
  }
  assert.ok(!g.alive);
  assert.equal(g.deathReason, "ceiling");
});

test("a pipe directly at the bird's column with the bird outside the gap kills on the next step", () => {
  const g = new FlappyGame({ seed: 1 });
  g.birdY = 5;
  g.birdVy = 0;
  g.pipes = [{ x: g.birdX, gapY: 15, scored: false }];
  g.step("NOFLAP");
  assert.ok(!g.alive);
  assert.equal(g.deathReason, "pipe");
});

test("a pipe with the bird inside the gap does not kill", () => {
  const g = new FlappyGame({ seed: 1 });
  g.birdY = 10;
  g.birdVy = 0;
  g.pipes = [{ x: g.birdX, gapY: 7, scored: false }]; // gap [7, 7+gapHeight) contains 10 for the default gapHeight (6)
  g.step("NOFLAP");
  assert.ok(g.alive);
});

test("moves() flags NOFLAP unsafe when it collides on the very next tick, even though the lookahead's reactive coast could otherwise recover from a milder fall", () => {
  const g = new FlappyGame({ seed: 1, height: 20 });
  // Right at the edge of the ground with real downward speed: NOFLAP crosses the ground boundary
  // on tick 0 itself, before the lookahead's reactive coast (which only corrects from tick 1
  // onward) ever gets a chance to flap. A milder fall one or two rows higher recovers fine --
  // that's the fix (see core/game.ts's #willCollideWithin comment), not tested here.
  g.birdY = g.height - 1.1;
  g.birdVy = 0.5;
  const m = g.moves();
  assert.equal(m.find((x) => x.action === "NOFLAP")!.safe, false);
  assert.equal(m.find((x) => x.action === "FLAP")!.safe, true);
});

test("moves() reports NOFLAP safe from a mild fall well above the ground (the lookahead's reactive coast recovers in time)", () => {
  const g = new FlappyGame({ seed: 1, height: 20 });
  g.birdY = g.height - 2;
  g.birdVy = 0.5;
  assert.equal(g.moves().find((x) => x.action === "NOFLAP")!.safe, true);
});

test("moves() flags FLAP unsafe when the bird is already at the ceiling", () => {
  const g = new FlappyGame({ seed: 1 });
  g.birdY = 0.3;
  g.birdVy = 0;
  const m = g.moves();
  assert.equal(m.find((x) => x.action === "FLAP")!.safe, false);
});

test("moves() can find both actions unsafe (no throw) -- Flappy Bird has no total-safety guarantee", () => {
  const g = new FlappyGame({ seed: 1 });
  g.birdY = 5;
  g.birdVy = 0;
  // Neither FLAP (-> y=4.1) nor NOFLAP (-> y=5.06) lands inside gap [15,17).
  g.pipes = [{ x: g.birdX, gapY: 15, scored: false }];
  const m = g.moves();
  assert.ok(m.every((x) => !x.safe));
});

test("guard restricts execution to the safe set and reports intervention", async () => {
  const g = new FlappyGame({ seed: 1, height: 20 });
  g.birdY = g.height - 1.1;
  g.birdVy = 0.5; // NOFLAP unsafe, FLAP safe (see the "collides on the very next tick" test above)
  const p = buildPrompt(g);
  assert.deepEqual(p.safe.map((m) => m.action).sort(), ["FLAP"]);
  const probabilities: Record<Action, number> = { FLAP: 0.1, NOFLAP: 0.9 };
  const stub = {
    predict: (): PredictLike => ({
      answers: { action: { probabilities }, risk: { noul: 0.8 }, aligned: { noul: 0.3 } },
      usage: { input_tokens: 1, output_tokens: 0 },
    }),
  };
  const guarded = await new LayaPolicy(stub).decide(g);
  assert.equal(guarded.proposed, "NOFLAP");
  assert.equal(guarded.executed, "FLAP");
  assert.ok(guarded.intervened);
  const raw = await new LayaPolicy(stub, { guarded: false }).decide(g);
  assert.equal(raw.executed, "NOFLAP");
  assert.ok(!raw.intervened);
});

test("an empty safe set executes the model's raw choice instead of throwing (documented Snake divergence)", async () => {
  const g = new FlappyGame({ seed: 1 });
  g.birdY = 5;
  g.birdVy = 0;
  g.pipes = [{ x: g.birdX, gapY: 15, scored: false }]; // both actions unsafe, see the moves() test above
  const p = buildPrompt(g);
  assert.equal(p.safe.length, 0);
  const probabilities: Record<Action, number> = { FLAP: 0.7, NOFLAP: 0.3 };
  const stub = {
    predict: (): PredictLike => ({
      answers: { action: { probabilities }, risk: { noul: 0.9 }, aligned: { noul: 0.1 } },
      usage: { input_tokens: 1 },
    }),
  };
  const decision = await new LayaPolicy(stub).decide(g);
  assert.equal(decision.proposed, "FLAP");
  assert.equal(decision.executed, "FLAP");
  assert.ok(!decision.intervened);
});

test("invalid model probabilities execute no action", async () => {
  const g = new FlappyGame({ seed: 1 });
  const stub = {
    predict: (): PredictLike => ({
      answers: { action: { probabilities: { FLAP: NaN, NOFLAP: 1 } }, risk: { noul: 1 }, aligned: { noul: 1 } },
      usage: { input_tokens: 1 },
    }),
  };
  await assert.rejects(new LayaPolicy(stub).decide(g), /invalid probability/);
});

test("seed reproduces pipe placement and gameplay given the same action sequence", () => {
  const a = new FlappyGame({ seed: 42 });
  const b = new FlappyGame({ seed: 42 });
  const actions: Action[] = [];
  for (let i = 0; i < 150; i++) actions.push(i % 7 === 0 ? "FLAP" : "NOFLAP");
  for (const action of actions) {
    if (!a.alive) break;
    a.step(action);
    b.step(action);
    assert.deepEqual(a.snapshot(), b.snapshot());
  }
});

test("a simple gap-tracking heuristic survives many ticks across several seeds without crashing", () => {
  for (let seed = 0; seed < 6; seed++) {
    const g = new FlappyGame({ seed });
    let n = 0;
    while (g.alive && n < 2000) {
      const next = g.pipes.find((p) => !p.scored);
      const target = next ? next.gapY + g.gapHeight / 2 : g.height / 2;
      g.step(g.birdY > target ? "FLAP" : "NOFLAP");
      n++;
    }
    assert.ok(n > 5, `seed ${seed} died almost immediately (${n} ticks)`);
    assert.ok(Number.isInteger(g.score) && g.score >= 0);
  }
});

test("ACTIONS is exactly FLAP and NOFLAP", () => {
  assert.deepEqual([...ACTIONS].sort(), ["FLAP", "NOFLAP"]);
});
