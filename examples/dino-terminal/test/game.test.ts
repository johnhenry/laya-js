/**
 * Pure-logic tests for the Dino core: physics, collision per obstacle
 * type, the reactive lookahead safety classifier, the airborne
 * no-decision-requested rule, and the shield. No Python reference (see
 * rng.ts) -- no parity tier.
 *
 * The reactive-coast fix this suite exists to guard against was found by
 * empirically simulating real play BEFORE writing these tests (not after):
 * a heuristic policy died to a pterodactyl on every single seed, with zero
 * ducks ever chosen, which is how the coast's off-by-one collision-zone
 * bug (see game.ts) was actually caught -- these tests encode that
 * finding, not just the isolated unit scenarios.
 */
import assert from "node:assert/strict";
import { ACTIONS, DinoGame, LayaPolicy, buildPrompt, decisionFrom, type Action, type Obstacle, type PredictLike } from "../src/core/index.ts";
import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const test = makeTest(bun);

function obstacleAtDino(game: DinoGame, kind: Obstacle["kind"]): Obstacle[] {
  return [{ x: game.dinoX, kind, scored: false }];
}

test("a cactus requires airborne: RUN and DUCK are fatal, JUMP is safe", () => {
  const game = new DinoGame({ seed: 1 });
  game.obstacles = obstacleAtDino(game, "cactus");
  const moves = game.moves();
  assert.equal(moves.find((m) => m.action === "JUMP")!.safe, true);
  assert.equal(moves.find((m) => m.action === "DUCK")!.safe, false);
  assert.equal(moves.find((m) => m.action === "RUN")!.safe, false);
});

test("a low pterodactyl requires ducking: RUN and JUMP are fatal, DUCK is safe", () => {
  const game = new DinoGame({ seed: 1 });
  game.obstacles = obstacleAtDino(game, "pterodactyl-low");
  const moves = game.moves();
  assert.equal(moves.find((m) => m.action === "DUCK")!.safe, true);
  assert.equal(moves.find((m) => m.action === "JUMP")!.safe, false);
  assert.equal(moves.find((m) => m.action === "RUN")!.safe, false);
});

test("a high pterodactyl punishes jumping specifically: JUMP is fatal, RUN and DUCK are safe", () => {
  const game = new DinoGame({ seed: 1 });
  game.obstacles = obstacleAtDino(game, "pterodactyl-high");
  const moves = game.moves();
  assert.equal(moves.find((m) => m.action === "JUMP")!.safe, false);
  assert.equal(moves.find((m) => m.action === "RUN")!.safe, true);
  assert.equal(moves.find((m) => m.action === "DUCK")!.safe, true);
});

test("moves() is empty while airborne -- no decision is meaningful mid-jump", () => {
  const game = new DinoGame({ seed: 1 });
  game.airborneTicksLeft = 5;
  assert.deepEqual(game.moves(), []);
});

test("LayaPolicy.decide() never calls predict() while airborne", async () => {
  const game = new DinoGame({ seed: 1 });
  game.airborneTicksLeft = 5;
  const stub = { predict: () => { throw new Error("predict() should not be called while airborne"); } };
  const decision = await new LayaPolicy(stub).decide(game);
  assert.equal(decision, null);
});

test("moves() finds RUN safe from a comfortably clear position (the reactive-coast regression test)", () => {
  // Direct regression test for the bug this suite is named after: a blind or off-by-one coast
  // makes RUN look unsafe far more often than it should. Default construction spawns obstacles
  // well beyond the reaction window, so RUN must be safe here.
  const game = new DinoGame({ seed: 1 });
  assert.equal(game.moves().find((m) => m.action === "RUN")!.safe, true);
});

test("step() applies gravity-free jump physics: airborne for the full jump duration, then lands", () => {
  const game = new DinoGame({ seed: 1 });
  game.obstacles = [];
  game.step("JUMP");
  assert.ok(game.airborne);
  const ticksAirborne = game.airborneTicksLeft + 1;
  for (let i = 0; i < ticksAirborne - 1; i++) game.step("RUN"); // action ignored while airborne
  assert.ok(!game.airborne);
});

test("step() rejects an unknown action when grounded", () => {
  const game = new DinoGame({ seed: 1 });
  assert.throws(() => game.step("FLAP" as Action), RangeError);
});

test("seed reproduces obstacle placement given the same action sequence", () => {
  const a = new DinoGame({ seed: 42 });
  const b = new DinoGame({ seed: 42 });
  const actions: Action[] = [];
  for (let i = 0; i < 150; i++) actions.push(ACTIONS[i % 3]!);
  for (const action of actions) {
    if (!a.alive) break;
    a.step(a.airborne ? "RUN" : action);
    b.step(b.airborne ? "RUN" : action);
    assert.deepEqual(a.snapshot(), b.snapshot());
  }
});

test("a shield-following heuristic (prefer safe RUN, else any safe action) survives 5000 ticks across many seeds", () => {
  for (let seed = 0; seed < 6; seed++) {
    const game = new DinoGame({ seed });
    let n = 0;
    let jumps = 0;
    let ducks = 0;
    while (game.alive && n < 5000) {
      let action: Action;
      if (game.airborne) {
        action = "RUN";
      } else {
        const moves = game.moves();
        const safe = moves.filter((m) => m.safe);
        const pool = safe.length ? safe : moves;
        const run = pool.find((m) => m.action === "RUN");
        action = (run ?? pool[0]!).action;
      }
      if (action === "JUMP") jumps++;
      if (action === "DUCK") ducks++;
      game.step(action);
      n++;
    }
    assert.equal(n, 5000, `seed ${seed} died at tick ${n} (${game.deathReason})`);
    // Both JUMP and DUCK must genuinely get used -- if either is ~0 across 5000 ticks and
    // several obstacle kinds, that's the exact shape of the coast bug this suite guards against.
    assert.ok(jumps > 20, `seed ${seed}: only ${jumps} jumps in 5000 ticks`);
    assert.ok(ducks > 20, `seed ${seed}: only ${ducks} ducks in 5000 ticks`);
  }
});

// ---------------------------------------------------------------- shield
test("guard restricts execution to the safe set and reports intervention", () => {
  const game = new DinoGame({ seed: 1 });
  game.obstacles = obstacleAtDino(game, "pterodactyl-low"); // DUCK safe; JUMP and RUN unsafe
  const p = buildPrompt(game);
  assert.deepEqual(p.safe.map((m) => m.action).sort(), ["DUCK"]);
  const probabilities: Record<Action, number> = { JUMP: 0.1, DUCK: 0.1, RUN: 0.8 };
  const stub: PredictLike = { answers: { action: { probabilities }, risk: { noul: 0.9 }, low_obstacle: { noul: 1 } }, usage: { input_tokens: 1, output_tokens: 0 } };
  const guarded = decisionFrom(stub, p, true);
  assert.equal(guarded.proposed, "RUN");
  assert.equal(guarded.executed, "DUCK");
  assert.ok(guarded.intervened);
  const raw = decisionFrom(stub, p, false);
  assert.equal(raw.executed, "RUN");
  assert.ok(!raw.intervened);
});

test("an empty safe set executes the model's raw choice instead of throwing (same precedent as Flappy Bird)", () => {
  const game = new DinoGame({ seed: 1 });
  // A cactus and a low pterodactyl stacked on the same spot: no single action clears both.
  game.obstacles = [
    { x: game.dinoX, kind: "cactus", scored: false },
    { x: game.dinoX, kind: "pterodactyl-low", scored: false },
  ];
  const p = buildPrompt(game);
  assert.equal(p.safe.length, 0);
  const probabilities: Record<Action, number> = { JUMP: 0.7, DUCK: 0.2, RUN: 0.1 };
  const stub: PredictLike = { answers: { action: { probabilities }, risk: { noul: 1 }, low_obstacle: { noul: 1 } }, usage: { input_tokens: 1 } };
  const decision = decisionFrom(stub, p, true);
  assert.equal(decision.proposed, "JUMP");
  assert.equal(decision.executed, "JUMP");
  assert.ok(!decision.intervened);
});

test("invalid model probabilities execute no action", () => {
  const game = new DinoGame({ seed: 1 });
  const p = buildPrompt(game);
  const stub: PredictLike = { answers: { action: { probabilities: { JUMP: NaN, DUCK: 0, RUN: 1 } }, risk: { noul: 1 }, low_obstacle: { noul: 0 } }, usage: { input_tokens: 1 } };
  assert.throws(() => decisionFrom(stub, p, true), /invalid probability/);
});
