/** Actors, the session turn loop, and full-game smoke play. Move-generation edge cases live in moves.test.ts. */
import assert from "node:assert/strict";
import { BotActor, HumanActor, LayaActor, type ActorResult } from "../src/core/actors.ts";
import { materialCount, newGame } from "../src/core/game.ts";
import { hopKey } from "../src/core/moves.ts";
import { LayaPolicy, buildPrompt, type PredictLike } from "../src/core/policy.ts";
import { CheckersRng } from "../src/core/rng.ts";
import { CheckersSession } from "../src/core/session.ts";
import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const test = makeTest(bun);

test("materialCount reflects the standard opening position", () => {
  const m = materialCount(newGame().board);
  assert.deepEqual(m.red, { men: 12, kings: 0 });
  assert.deepEqual(m.black, { men: 12, kings: 0 });
});

test("BotActor always picks a legal (compliant) hop and prefers a capture when one exists", async () => {
  const state = {
    board: {
      a3: { player: "red" as const, kind: "man" as const },
      b4: { player: "black" as const, kind: "man" as const },
      g3: { player: "red" as const, kind: "man" as const }, // a non-capture option that must be ignored
    },
    toMove: "red" as const,
    turnNumber: 1,
    forcedContinuation: null,
    status: "in_progress" as const,
    moveHistory: [],
  };
  const bot = new BotActor(new CheckersRng(1));
  const { hop } = await bot.act(state);
  assert.equal(hop.kind, "capture");
  assert.equal(hopKey(hop), "a3xc5");
});

test("BotActor's capture tie-break is deterministic for a given seed", async () => {
  const state = {
    board: {
      a3: { player: "red" as const, kind: "man" as const },
      b4: { player: "black" as const, kind: "man" as const },
      g3: { player: "red" as const, kind: "man" as const },
      f4: { player: "black" as const, kind: "man" as const },
    },
    toMove: "red" as const,
    turnNumber: 1,
    forcedContinuation: null,
    status: "in_progress" as const,
    moveHistory: [],
  };
  const a = await new BotActor(new CheckersRng(7)).act(state);
  const b = await new BotActor(new CheckersRng(7)).act(state);
  assert.equal(hopKey(a.hop), hopKey(b.hop));
});

test("HumanActor re-prompts on invalid input and returns the selected compliant hop", async () => {
  const responses = ["not a number", "99", "1"];
  const io = { prompt: async () => responses.shift()! };
  const actor = new HumanActor(io);
  const { hop } = await actor.act(newGame());
  assert.equal(responses.length, 0); // all three prompts were consumed
  assert.ok(hop.kind === "simple");
});

test("LayaActor applies the shield via LayaPolicy and surfaces the Decision", async () => {
  const s = {
    board: {
      a3: { player: "red" as const, kind: "man" as const },
      b4: { player: "black" as const, kind: "man" as const },
      g3: { player: "red" as const, kind: "man" as const },
    },
    toMove: "red" as const,
    turnNumber: 1,
    forcedContinuation: null,
    status: "in_progress" as const,
    moveHistory: [],
  };
  const p = buildPrompt(s);
  // g3-f4 is legal but not compliant (mandatory capture is in effect); the model prefers it anyway.
  const probabilities = Object.fromEntries(Object.keys(p.questions.move.criteria).map((k) => [k, k === "g3-f4" ? 0.9 : 0.01]));
  const stub = {
    predict: (): PredictLike => ({
      answers: { move: { probabilities }, material_at_risk: { noul: 0.4 }, captures_material: { noul: 0 } },
      usage: { input_tokens: 1, output_tokens: 0 },
    }),
  };
  const actor = new LayaActor(new LayaPolicy(stub));
  const { hop, decision } = await actor.act(s);
  assert.equal(hopKey(hop), "a3xc5");
  assert.ok(decision!.intervened);
});

test("bot vs. bot games terminate cleanly with a winner, across many seeds", async () => {
  for (let seed = 0; seed < 8; seed++) {
    const session = new CheckersSession(
      { red: new BotActor(new CheckersRng(seed)), black: new BotActor(new CheckersRng(seed + 1000)) },
      { autoNextRound: false },
    );
    let ticks = 0;
    let result: (ActorResult & { stop: boolean }) | undefined;
    for (; ticks < 2000; ticks++) {
      const r = await session.tick();
      if (r.stop) {
        result = r as unknown as ActorResult & { stop: boolean };
        break;
      }
    }
    assert.ok(result, `seed ${seed}: game did not finish within 2000 hops`);
    assert.ok(session.game.status === "red_wins" || session.game.status === "black_wins", `seed ${seed}: unexpected status ${session.game.status}`);
    assert.equal(session.stats.red_wins + session.stats.black_wins, 1);
  }
});

test("CheckersSession auto-advances to the next round after a win when autoNextRound is true", async () => {
  // A contrived near-finished position: black has one piece, red can capture it and black then has no moves.
  const session = new CheckersSession(
    { red: new BotActor(new CheckersRng(1)), black: new BotActor(new CheckersRng(2)) },
    { autoNextRound: true },
  );
  session.game = {
    board: { a3: { player: "red", kind: "man" }, b4: { player: "black", kind: "man" } },
    toMove: "red",
    turnNumber: 1,
    forcedContinuation: null,
    status: "in_progress",
    moveHistory: [],
  };
  const r = await session.tick(); // red captures black's only piece
  assert.ok(r.roundEnd);
  assert.equal(r.stop, false);
  assert.equal(session.stats.round, 2);
  assert.equal(session.game.turnNumber, 1); // a fresh game
  assert.equal(Object.keys(session.game.board).length, 24); // back to the standard opening setup
});
