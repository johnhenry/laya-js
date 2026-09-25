/**
 * The three opponent modes: a real Laya policy, a zero-inference scripted
 * bot, and a terminal-menu human. All three implement the same `Actor`
 * interface so `session.ts`'s turn loop doesn't need to know which is
 * playing which side.
 */
import { compliantHopsFor, type GameSnapshot } from "./game.ts";
import { hopKey, type Hop } from "./moves.ts";
import { LayaPolicy, type Decision } from "./policy.ts";
import type { CheckersRng } from "./rng.ts";

export interface ActorResult {
  hop: Hop;
  /** Set only for the `laya` actor -- the bot and human actors don't run a model. */
  decision?: Decision;
}

export interface Actor {
  readonly kind: "laya" | "bot" | "human";
  act(state: GameSnapshot): Promise<ActorResult>;
}

export class LayaActor implements Actor {
  readonly kind = "laya" as const;
  readonly policy: LayaPolicy;

  constructor(policy: LayaPolicy) {
    this.policy = policy;
  }

  async act(state: GameSnapshot): Promise<ActorResult> {
    const decision = await this.policy.decide(state);
    return { hop: decision.executed, decision };
  }
}

/** Zero-inference: prefers a capture (arbitrary tie-break via the seeded RNG), else a random compliant hop. */
export class BotActor implements Actor {
  readonly kind = "bot" as const;
  readonly rng: CheckersRng;

  constructor(rng: CheckersRng) {
    this.rng = rng;
  }

  async act(state: GameSnapshot): Promise<ActorResult> {
    const compliant = compliantHopsFor(state);
    if (!compliant.length) throw new Error("BotActor.act: no compliant hops -- the game is already over");
    const captures = compliant.filter((h) => h.kind === "capture");
    const pool = captures.length ? captures : compliant;
    return { hop: pool[this.rng.int(0, pool.length - 1)]! };
  }
}

export interface HumanIO {
  /** Print the numbered menu (and anything else); read a line of input. */
  prompt(lines: string[]): Promise<string>;
}

/** Terminal: print the numbered compliant-hop menu, read a line of stdin -- a plain numbered menu, not chess notation. */
export class HumanActor implements Actor {
  readonly kind = "human" as const;
  readonly io: HumanIO;

  constructor(io: HumanIO) {
    this.io = io;
  }

  async act(state: GameSnapshot): Promise<ActorResult> {
    const compliant = compliantHopsFor(state);
    if (!compliant.length) throw new Error("HumanActor.act: no compliant hops -- the game is already over");
    const lines = compliant.map((h, i) => `  ${i + 1}. ${hopKey(h)}${h.kind === "capture" ? " (capture)" : ""}`);
    for (;;) {
      const raw = await this.io.prompt(lines);
      const n = Number(raw.trim());
      if (Number.isInteger(n) && n >= 1 && n <= compliant.length) return { hop: compliant[n - 1]! };
    }
  }
}
