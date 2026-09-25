#!/usr/bin/env node
/**
 * laya-checkers (JS): the live terminal Checkers demo, driven by real Laya
 * predictions, a scripted bot, and/or a human player on either side. Same
 * flag family as snake-terminal/flappy-terminal where it applies -- but
 * turn-based rather than tick-based, and with no `--unassisted`: guarding
 * is chosen per seat via `--red`/`--black` (pick `laya` for a seat, there's
 * no separate flag to disable its shield here) rather than one global
 * toggle, since an unguarded seat rarely differs from a guarded one in
 * outcome -- illegal hops are excluded from the criteria the model even
 * sees, unlike Snake/Flappy where the raw top-1 can be any of the fixed
 * options including unsafe ones.
 */
import { execFileSync } from "node:child_process";
import { createWriteStream, mkdirSync, type WriteStream } from "node:fs";
import { dirname } from "node:path";
import { createInterface } from "node:readline/promises";
import { parseArgs } from "node:util";
import type { Player } from "./core/board.ts";
import { BotActor, HumanActor, LayaActor, type Actor, type HumanIO } from "./core/actors.ts";
import { DEFAULT_MODEL, LayaPolicy, type PromptStyle } from "./core/policy.ts";
import { CheckersRng } from "./core/rng.ts";
import { CheckersSession, type TickResult } from "./core/session.ts";
import { loadCheckersAgent, type BackendName } from "./load-agent.ts";
import { compose } from "./ui.ts";

const HELP = `laya-checkers — Checkers with real Laya predictions, a scripted bot, and/or a human player

Usage: laya-checkers [options]

  --model <id|dir>       Cached Hub id or local checkpoint (default ${DEFAULT_MODEL})
  --backend <name>       auto | mlx | webgpu | cpu (default auto)
  --dtype <f16|f32>      Weight/compute dtype (default f16)
  --prompt <style>       compact | detailed (default compact)
  --optimize             Ask the agent for 16-token buckets + prefix cache (if supported)
  --red <who>            laya | bot | human (default laya)
  --black <who>          laya | bot | human (default bot)
  --seed <n>             Seed for the bot's tie-breaking RNG (default 7); round r uses seed + r - 1
  --pace <n>             Turns per second when spectating (default 2)
  --max-speed            No pacing delay between turns
  --duration <s>         Stop after this many seconds
  --steps <n>            Stop after this many hops (per episode with --episodes)
  --record <file.jsonl>  Write turns and boards (laya-checkers-v1 JSONL)
  --headless             No terminal display
  --episodes <n>         Benchmark: n games back to back (bot/laya seats only, no human)
  --online               Allow downloading a missing checkpoint

Ctrl-C quits.`;

class UsageError extends Error {}

function hardwareName(): string {
  if (process.platform === "darwin") {
    try {
      return execFileSync("sysctl", ["-n", "machdep.cpu.brand_string"], { encoding: "utf8" }).trim().replace(/^Apple /, "");
    } catch {}
  }
  return process.arch;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function positive(name: string, v: string | undefined): number | undefined {
  if (v === undefined) return undefined;
  const n = Number(v);
  if (!(n > 0 && Number.isFinite(n))) throw new UsageError(`--${name}: expected a positive finite number`);
  return n;
}

function seatKind(name: string, v: string): "laya" | "bot" | "human" {
  if (v !== "laya" && v !== "bot" && v !== "human") throw new UsageError(`--${name} must be laya, bot or human`);
  return v;
}

class TerminalHumanIO implements HumanIO {
  #rl = createInterface({ input: process.stdin, output: process.stdout });
  async prompt(lines: string[]): Promise<string> {
    for (const line of lines) console.log(line);
    return this.#rl.question("Your move (number): ");
  }
  close(): void {
    this.#rl.close();
  }
}

function runtimeName(): string {
  const bun = (globalThis as { Bun?: { version: string } }).Bun;
  return bun ? `bun ${bun.version}` : `node ${process.versions.node}`;
}

async function main(argv: string[]): Promise<number> {
  const { values: a } = parseArgs({
    args: argv,
    options: {
      model: { type: "string" },
      backend: { type: "string", default: "auto" },
      dtype: { type: "string", default: "f16" },
      prompt: { type: "string", default: "compact" },
      optimize: { type: "boolean", default: false },
      red: { type: "string", default: "laya" },
      black: { type: "string", default: "bot" },
      seed: { type: "string", default: "7" },
      pace: { type: "string", default: "2" },
      "max-speed": { type: "boolean", default: false },
      duration: { type: "string" },
      steps: { type: "string" },
      record: { type: "string" },
      headless: { type: "boolean", default: false },
      episodes: { type: "string" },
      online: { type: "boolean", default: false },
      help: { type: "boolean", short: "h", default: false },
    },
  });
  if (a.help) {
    console.log(HELP);
    return 0;
  }
  const backend = a.backend as BackendName;
  if (!["auto", "mlx", "webgpu", "cpu"].includes(backend)) throw new UsageError("--backend must be auto, mlx, webgpu or cpu");
  if (a.dtype !== "f16" && a.dtype !== "f32") throw new UsageError("--dtype must be f16 or f32");
  if (a.prompt !== "compact" && a.prompt !== "detailed") throw new UsageError("--prompt must be compact or detailed");
  const redKind = seatKind("red", a.red);
  const blackKind = seatKind("black", a.black);
  const seed = Number(a.seed);
  let pace = positive("pace", a.pace)!;
  const duration = positive("duration", a.duration);
  const steps = a.steps === undefined ? undefined : Number(a.steps);
  if (steps !== undefined && !(Number.isInteger(steps) && steps >= 1)) throw new UsageError("--steps must be positive");
  const episodes = a.episodes === undefined ? undefined : Number(a.episodes);
  if (episodes !== undefined && !(Number.isInteger(episodes) && episodes >= 1)) throw new UsageError("--episodes must be positive");
  const headless = a.headless || episodes !== undefined;
  if (headless && (redKind === "human" || blackKind === "human")) throw new UsageError("a human seat needs an interactive display -- remove --headless/--episodes or change --red/--black");
  if (!headless && !process.stdout.isTTY) throw new UsageError("Interactive display needs a TTY. Use --headless for a non-interactive run.");

  const needsAgent = redKind === "laya" || blackKind === "laya";
  let agent: Awaited<ReturnType<typeof loadCheckersAgent>>["agent"] | undefined;
  let engine = "—";
  let policy: LayaPolicy | undefined;
  if (needsAgent) {
    const model = a.model ?? DEFAULT_MODEL;
    process.stderr.write(`Loading ${model} (${backend}, ${a.dtype})${a.online ? "" : "; offline, no network requests"}...\n`);
    const loadStart = performance.now();
    const res = await loadCheckersAgent(model, { backend, dtype: a.dtype, offline: !a.online, optimize: a.optimize });
    agent = res.agent;
    engine = res.engine;
    process.stderr.write(`Loaded in ${((performance.now() - loadStart) / 1000).toFixed(1)} s on ${engine}\n`);
    policy = new LayaPolicy(agent, { prompt: a.prompt as PromptStyle });
  }
  const hardware = hardwareName();

  function makeActor(kind: "laya" | "bot" | "human", rngSeed: number, io?: TerminalHumanIO): Actor {
    if (kind === "laya") return new LayaActor(policy!);
    if (kind === "bot") return new BotActor(new CheckersRng(rngSeed));
    return new HumanActor(io!);
  }

  try {
    if (episodes !== undefined) return await benchmark({ red: redKind, black: blackKind }, seed, episodes, steps, { hardware, engine, model: a.model ?? DEFAULT_MODEL, backend });
    return await play({ red: redKind, black: blackKind }, seed, { hardware, engine, headless, pace, maxSpeed: a["max-speed"], duration, steps, record: a.record });
  } finally {
    await agent?.dispose?.();
  }

  async function benchmark(kinds: Record<Player, "laya" | "bot" | "human">, seed0: number, n: number, maxSteps: number | undefined, meta: { hardware: string; engine: string; model: string; backend: string }): Promise<number> {
    const results: { seed: number; turns: number; winner: string; interventions: number }[] = [];
    let interventions = 0;
    for (let e = 0; e < n; e++) {
      const seed = seed0 + e;
      const actors = { red: makeActor(kinds.red, seed), black: makeActor(kinds.black, seed + 1_000_000) };
      const session = new CheckersSession(actors, { hardware: meta.hardware, engine: meta.engine, autoNextRound: false });
      let last: TickResult & { stop: boolean };
      let turns = 0;
      do {
        last = await session.tick();
        turns++;
        if (maxSteps && turns >= maxSteps) break;
      } while (!last.stop);
      const winner = session.game.status === "in_progress" ? "none (step limit reached)" : session.game.status;
      results.push({ seed, turns, winner, interventions: session.stats.interventions });
      interventions += session.stats.interventions;
      console.error(`seed=${seed} turns=${turns} winner=${winner} interventions=${session.stats.interventions}`);
    }
    console.log(JSON.stringify({ engine: meta.engine, backend: meta.backend, model: meta.model, hardware: meta.hardware, runtime: runtimeName(), red: kinds.red, black: kinds.black, episodes: n, interventions, per_game: results }, null, 2));
    return 0;
  }

  async function play(kinds: Record<Player, "laya" | "bot" | "human">, seed0: number, opts: { hardware: string; engine: string; headless: boolean; pace: number; maxSpeed: boolean; duration?: number; steps?: number; record?: string }): Promise<number> {
    const io = kinds.red === "human" || kinds.black === "human" ? new TerminalHumanIO() : undefined;
    const actors = { red: makeActor(kinds.red, seed0, io), black: makeActor(kinds.black, seed0 + 1_000_000, io) };
    const session = new CheckersSession(actors, { hardware: opts.hardware, engine: opts.engine });
    let record: WriteStream | undefined;
    if (opts.record) {
      mkdirSync(dirname(opts.record), { recursive: true });
      record = createWriteStream(opts.record, { flags: "wx" });
      record.write(JSON.stringify({ type: "metadata", format: "laya-checkers-v1", created_utc: new Date().toISOString(), red: kinds.red, black: kinds.black, hardware: opts.hardware, engine: opts.engine, runtime: runtimeName() }) + "\n");
    }
    let quit = false;
    const onSigint = () => (quit = true);
    process.on("SIGINT", onSigint);
    session.restartClock();
    let last: TickResult | undefined;
    try {
      while (!quit) {
        if ((opts.duration && session.elapsedMs >= opts.duration * 1000) || (opts.steps && session.stats.turns >= opts.steps)) break;
        if (!opts.headless) {
          const lastTurn = last ? { hop: last.hop, player: last.board.toMove, actor: last.actor, decision: last.decision } : undefined;
          process.stdout.write("\x1b[2J\x1b[H" + compose(session.game.board, session.game.toMove, session.stats.round, lastTurn, session.stats).ansi(!process.env.NO_COLOR));
        }
        const r = await session.tick();
        last = r;
        record?.write(JSON.stringify({ type: "turn", at: session.elapsedMs / 1000, board: r.board, hop: r.hop, actor: r.actor, decision: r.decision }) + "\n");
        if (r.roundEnd) {
          record?.write(JSON.stringify({ type: "round_end", at: session.elapsedMs / 1000, game: r.roundEnd }) + "\n");
          if (!opts.headless) {
            process.stdout.write("\x1b[2J\x1b[H" + compose(r.roundEnd.board, r.roundEnd.toMove, session.stats.round, { hop: r.hop, player: r.board.toMove, actor: r.actor, decision: r.decision }, session.stats).ansi(!process.env.NO_COLOR));
          }
          if (r.stop) break;
          await sleep(opts.headless ? 0 : 1500);
          last = undefined;
        } else if (!opts.maxSpeed && !opts.headless) {
          await sleep(1000 / opts.pace);
        }
      }
    } finally {
      process.off("SIGINT", onSigint);
      io?.close();
      if (record) {
        record.write(JSON.stringify({ type: "end", stats: session.stats, game: session.game }) + "\n");
        await new Promise((r) => record!.end(r));
      }
    }
    console.log(JSON.stringify({ turns: session.stats.turns, red_wins: session.stats.red_wins, black_wins: session.stats.black_wins, interventions: session.stats.interventions, engine: opts.engine }, null, 2));
    return 0;
  }
}

main(process.argv.slice(2)).then(
  (code) => process.exit(code),
  (error) => {
    if (error instanceof UsageError || (error as { code?: string }).code === "ERR_PARSE_ARGS_UNKNOWN_OPTION") {
      console.error(`laya-checkers: ${error.message}\n\n${HELP}`);
      process.exit(2);
    }
    console.error(error);
    process.exit(1);
  },
);
