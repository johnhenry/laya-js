#!/usr/bin/env node
/**
 * laya-snake (JS): the live terminal Snake demo driven by real Laya predictions
 * on MLX, WebGPU or CPU. Port of laya-mlx `laya_mlx/snake/cli.py` + the
 * uncapped part of `benchmark.py` (`--episodes N --headless`).
 */
import { execFileSync } from "node:child_process";
import { createWriteStream, mkdirSync, type WriteStream } from "node:fs";
import { dirname } from "node:path";
import { parseArgs } from "node:util";
import { SnakeGame } from "./core/game.ts";
import { DEFAULT_MODEL, LayaPolicy, type PromptStyle } from "./core/policy.ts";
import { SnakeSession } from "./core/session.ts";
import { Keyboard } from "./keyboard.ts";
import { loadSnakeAgent, type BackendName } from "./load-agent.ts";
import { compose, layoutSize } from "./ui.ts";

const HELP = `laya-snake — Snake driven by real Laya predictions (MLX / WebGPU / CPU)

Usage: laya-snake [options]

  --model <id|dir>       Cached Hub id or local checkpoint (default ${DEFAULT_MODEL})
  --backend <name>       auto | mlx | webgpu | cpu (default auto)
  --dtype <f16|f32>      Weight/compute dtype (default f16)
  --prompt <style>       compact | detailed (default compact)
  --optimize             Ask the agent for 16-token buckets + prefix cache (if supported)
  --width/--height <n>   Board size (default 24 x 16)
  --seed <n>             Seed (default 7); round r uses seed + r - 1
  --initial-length <n>   Initial snake length (default 6)
  --fps <n>              Paced decisions per second (default 12)
  --max-speed            One move per completed inference, no pacing
  --duration <s>         Stop after this many seconds (excluding warmup)
  --steps <n>            Stop after this many game steps (per episode with --episodes)
  --unassisted           Execute raw Laya top-1; disable the safety shield
  --record <file.jsonl>  Write decisions and boards (laya-snake-v1 JSONL)
  --headless             No terminal display
  --episodes <n>         Benchmark: n uncapped episodes (seeds seed..seed+n-1), prints moves/s
  --no-alt-screen        Keep the final frame in scrollback
  --online               Allow downloading a missing checkpoint

Keys: SPACE pause · ↑/↓ (or +/-) speed · R reset · Q quit`;

function hardwareName(): string {
  if (process.platform === "darwin") {
    try {
      return execFileSync("sysctl", ["-n", "machdep.cpu.brand_string"], { encoding: "utf8" }).trim().replace(/^Apple /, "");
    } catch {}
  }
  return process.arch;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function percentile(values: number[], q: number): number {
  // numpy.percentile, linear interpolation
  const s = [...values].sort((a, b) => a - b);
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return s[lo]! + (s[hi]! - s[lo]!) * (pos - lo);
}

function positive(name: string, v: string | undefined): number | undefined {
  if (v === undefined) return undefined;
  const n = Number(v);
  if (!(n > 0 && Number.isFinite(n))) throw new UsageError(`--${name}: expected a positive finite number`);
  return n;
}

class UsageError extends Error {}

async function main(argv: string[]): Promise<number> {
  const { values: a } = parseArgs({
    args: argv,
    options: {
      model: { type: "string" },
      backend: { type: "string", default: "auto" },
      dtype: { type: "string", default: "f16" },
      prompt: { type: "string", default: "compact" },
      optimize: { type: "boolean", default: false },
      width: { type: "string", default: "24" },
      height: { type: "string", default: "16" },
      seed: { type: "string", default: "7" },
      "initial-length": { type: "string", default: "6" },
      fps: { type: "string", default: "12" },
      "max-speed": { type: "boolean", default: false },
      duration: { type: "string" },
      steps: { type: "string" },
      unassisted: { type: "boolean", default: false },
      record: { type: "string" },
      headless: { type: "boolean", default: false },
      episodes: { type: "string" },
      "no-alt-screen": { type: "boolean", default: false },
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
  const board = {
    width: Number(a.width),
    height: Number(a.height),
    seed: Number(a.seed),
    initialLength: Number(a["initial-length"]),
  };
  let fps = positive("fps", a.fps)!;
  const duration = positive("duration", a.duration);
  const steps = a.steps === undefined ? undefined : Number(a.steps);
  if (steps !== undefined && !(Number.isInteger(steps) && steps >= 1)) throw new UsageError("--steps must be positive");
  const episodes = a.episodes === undefined ? undefined : Number(a.episodes);
  if (episodes !== undefined && !(Number.isInteger(episodes) && episodes >= 1)) throw new UsageError("--episodes must be positive");
  try {
    new SnakeGame(board.width, board.height, board.seed, board.initialLength);
  } catch (e) {
    throw new UsageError((e as Error).message);
  }
  const interactive = !a.headless && episodes === undefined;
  if (interactive && !process.stdout.isTTY) {
    throw new UsageError("Interactive display needs a TTY. Use --headless for a non-interactive run.");
  }

  const model = a.model ?? DEFAULT_MODEL;
  process.stderr.write(`Loading ${model} (${backend}, ${a.dtype})${a.online ? "" : "; offline, no network requests"}...\n`);
  const loadStart = performance.now();
  const { agent, engine } = await loadSnakeAgent(model, {
    backend,
    dtype: a.dtype,
    offline: !a.online,
    optimize: a.optimize,
  });
  process.stderr.write(`Loaded in ${((performance.now() - loadStart) / 1000).toFixed(1)} s on ${engine}\n`);
  const hardware = hardwareName();
  const policy = new LayaPolicy(agent, { guarded: !a.unassisted, prompt: a.prompt as PromptStyle });

  // Warmup, like Python (6 decisions on seed + 10000).
  const warm = new SnakeGame(board.width, board.height, board.seed + 10000, board.initialLength);
  for (let i = 0; i < 6 && warm.alive; i++) warm.step((await policy.decide(warm)).executed);

  try {
    if (episodes !== undefined) return await benchmark(policy, board, episodes, steps ?? 600, { hardware, engine, model, backend });
    return await play();
  } finally {
    await agent.dispose?.();
  }

  async function play(): Promise<number> {
    const session = new SnakeSession(policy, board, { hardware, engine });
    const stats = session.stats;
    let record: WriteStream | undefined;
    if (a.record) {
      mkdirSync(dirname(a.record), { recursive: true });
      record = createWriteStream(a.record, { flags: "wx" });
      record.write(
        JSON.stringify({
          type: "metadata",
          format: "laya-snake-v1",
          created_utc: new Date().toISOString(),
          model: { name: model, hardware, engine, runtime: runtimeName(), prompt: a.prompt, network: a.online ? "online" : "offline" },
          settings: a,
          note: "Real synchronized inference; board is shown before the announced action. Risk is 1 - P(safe route).",
        }) + "\n",
      );
    }
    const out = process.stdout;
    const keys = interactive ? new Keyboard() : undefined;
    const altScreen = interactive && !a["no-alt-screen"];
    const noColor = !!process.env.NO_COLOR;
    if (interactive) out.write((altScreen ? "\x1b[?1049h" : "") + "\x1b[?25l\x1b[2J");
    const draw = (text: string) => out.write("\x1b[H" + text);
    let displayed: { board: ReturnType<SnakeGame["snapshot"]>; decision: object } = { board: session.game.snapshot(), decision: {} };
    const inference: number[] = [];
    let calls = 0;
    let quit = false;
    const onSigint = () => (quit = true);
    process.on("SIGINT", onSigint);
    session.restartClock();
    try {
      while (!quit) {
        const now = performance.now();
        if ((duration && session.elapsedMs >= duration * 1000) || (steps && stats.steps >= steps)) break;
        const pressed = keys?.read().toLowerCase() ?? "";
        if (pressed.includes("q") || pressed.includes("\x03")) break;
        if (pressed.includes(" ")) stats.paused = !stats.paused;
        if (pressed.includes("\x1b[a") || pressed.includes("+")) fps = Math.min(240, fps + 2);
        if (pressed.includes("\x1b[b") || pressed.includes("-")) fps = Math.max(1, fps - 2);
        if (pressed.includes("r")) {
          session.reset();
          displayed = { board: session.game.snapshot(), decision: {} };
        }
        if (stats.paused) {
          stats.elapsed = session.elapsedMs / 1000;
          if (interactive) draw(compose(displayed.board, displayed.decision, stats).ansi(!noColor));
          await sleep(30);
          continue;
        }
        if (interactive) {
          const [mw, mh] = layoutSize(board.width, board.height);
          if ((out.columns ?? 0) < mw || (out.rows ?? 0) < mh) {
            out.write(`\x1b[2J\x1b[HResize terminal to at least ${mw} columns × ${mh} rows.\nThe game is waiting. Q quits.`);
            await sleep(100);
            continue;
          }
        }
        const { board: shown, decision } = await session.decide();
        calls++;
        inference.push(decision.inference_ms);
        displayed = { board: shown, decision };
        if (interactive) draw(compose(shown, decision, stats).ansi(!noColor));
        record?.write(JSON.stringify({ type: "frame", at: session.elapsedMs / 1000, game: shown, decision, stats: { ...stats } }) + "\n");
        if (!a["max-speed"]) {
          const remaining = 1000 / fps - (performance.now() - now);
          if (remaining > 0) await sleep(remaining);
        }
        const r = session.advance(decision);
        if (r.roundEnd) {
          record?.write(JSON.stringify({ type: "round_end", at: session.elapsedMs / 1000, game: r.roundEnd }) + "\n");
          if (r.stop) break;
          if (interactive) {
            draw(compose(r.roundEnd, {}, stats).ansi(!noColor));
            await sleep(1000);
          }
        }
      }
    } finally {
      process.off("SIGINT", onSigint);
      keys?.close();
      if (interactive) out.write("\x1b[0m\x1b[?25h" + (altScreen ? "\x1b[?1049l" : "\n"));
    }
    const seconds = session.elapsedMs / 1000;
    const summary = {
      steps: stats.steps,
      inference_calls: calls,
      seconds,
      steps_per_second: seconds ? stats.steps / seconds : 0,
      score: session.game.score,
      length: session.game.body.length,
      best_score: stats.best,
      interventions: stats.interventions,
      deaths: stats.deaths,
      guarded: policy.guarded,
      network: a.online ? "online" : "offline",
      engine,
      mean_inference_ms: inference.length ? inference.reduce((x, y) => x + y, 0) / inference.length : null,
    };
    if (record) {
      record.write(JSON.stringify({ type: "end", summary, game: session.game.snapshot() }) + "\n");
      await new Promise((r) => record!.end(r));
    }
    console.log(JSON.stringify(summary, null, 2));
    return 0;
  }
}

function runtimeName(): string {
  const bun = (globalThis as { Bun?: { version: string } }).Bun;
  return bun ? `bun ${bun.version}` : `node ${process.versions.node}`;
}

/** `--episodes N --headless`: uncapped episodes like Python's benchmark `uncapped_soak`. */
async function benchmark(
  policy: LayaPolicy,
  board: { width: number; height: number; seed: number; initialLength: number },
  episodes: number,
  steps: number,
  meta: { hardware: string; engine: string; model: string; backend: string },
): Promise<number> {
  const results = [];
  let totalSteps = 0;
  let totalSeconds = 0;
  let deaths = 0;
  let interventions = 0;
  const allInference: number[] = [];
  for (let e = 0; e < episodes; e++) {
    const seed = board.seed + e;
    const game = new SnakeGame(board.width, board.height, seed, board.initialLength);
    const inference: number[] = [];
    let episodeInterventions = 0;
    let lastFood = 0;
    let maxFoodGap = 0;
    const started = performance.now();
    for (let i = 0; i < steps; i++) {
      const decision = await policy.decide(game);
      episodeInterventions += decision.intervened ? 1 : 0;
      inference.push(decision.inference_ms);
      const ate = game.step(decision.executed);
      if (ate) {
        maxFoodGap = Math.max(maxFoodGap, game.ticks - lastFood);
        lastFood = game.ticks;
      }
      if (policy.guarded && game.alive && !game.cycleOrderValid()) throw new Error("Guarded game broke its cycle-order invariant");
      if (!game.alive || game.won) break;
    }
    const seconds = (performance.now() - started) / 1000;
    const r = {
      seed,
      steps: inference.length,
      seconds,
      moves_per_second: inference.length / seconds,
      score: game.score,
      length: game.body.length,
      alive: game.alive,
      won: game.won,
      death_reason: game.deathReason,
      interventions: episodeInterventions,
      max_completed_food_gap: maxFoodGap,
      inference_ms: {
        mean: inference.reduce((x, y) => x + y, 0) / inference.length,
        p50: percentile(inference, 0.5),
        p95: percentile(inference, 0.95),
        p99: percentile(inference, 0.99),
        max: Math.max(...inference),
      },
    };
    results.push(r);
    totalSteps += r.steps;
    totalSeconds += seconds;
    deaths += game.alive ? 0 : 1;
    interventions += episodeInterventions;
    allInference.push(...inference);
    console.error(
      `${policy.guarded ? "shield" : "top-1"} seed=${seed} steps=${r.steps} score=${r.score} alive=${r.alive} ` +
        `actual=${r.moves_per_second.toFixed(1)}/s p50=${r.inference_ms.p50.toFixed(1)}ms interventions=${episodeInterventions}`,
    );
  }
  const summary = {
    engine: meta.engine,
    backend: meta.backend,
    model: meta.model,
    hardware: meta.hardware,
    runtime: runtimeName(),
    prompt: policy.prompt,
    guarded: policy.guarded,
    episodes,
    steps: totalSteps,
    seconds: totalSeconds,
    moves_per_second: totalSteps / totalSeconds,
    deaths,
    interventions,
    inference_ms: {
      mean: allInference.reduce((x, y) => x + y, 0) / allInference.length,
      p50: percentile(allInference, 0.5),
      p95: percentile(allInference, 0.95),
    },
    per_episode: results,
  };
  console.log(JSON.stringify(summary, null, 2));
  return 0;
}

main(process.argv.slice(2)).then(
  (code) => process.exit(code),
  (error) => {
    if (error instanceof UsageError || (error as { code?: string }).code === "ERR_PARSE_ARGS_UNKNOWN_OPTION") {
      console.error(`laya-snake: ${error.message}\n\n${HELP}`);
      process.exit(2);
    }
    console.error(error);
    process.exit(1);
  },
);
