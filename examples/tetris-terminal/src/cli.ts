#!/usr/bin/env node
/**
 * laya-tetris (JS): the live terminal Tetris demo driven by real Laya
 * predictions on MLX, WebGPU or CPU. Same shape as snake-terminal's/
 * flappy-terminal's cli.ts (`--episodes N --headless` benchmark mode
 * included), but decisions are per gravity STEP, not per piece -- see
 * core/policy.ts for why. This game is JS-original -- see core/rng.ts --
 * so the flag family is mirrored from Snake for consistency across the
 * demos, not because there's a laya-mlx tetris.py to match.
 */
import { execFileSync } from "node:child_process";
import { createWriteStream, mkdirSync, type WriteStream } from "node:fs";
import { dirname } from "node:path";
import { parseArgs } from "node:util";
import { SPAWN_ROW, TetrisGame, type GameSnapshot } from "./core/game.ts";
import { spawnColFor } from "./core/pieces.ts";
import { DEFAULT_MODEL, LayaPolicy, type PromptStyle, type StepDecision, type StepPosition } from "./core/policy.ts";
import { TetrisSession } from "./core/session.ts";
import { Keyboard } from "./keyboard.ts";
import { loadTetrisAgent, type BackendName } from "./load-agent.ts";
import { compose, layoutSize, type FallingPiece } from "./ui.ts";

const HELP = `laya-tetris — Tetris driven by real Laya predictions (MLX / WebGPU / CPU)

Usage: laya-tetris [options]

  --model <id|dir>       Cached Hub id or local checkpoint (default ${DEFAULT_MODEL})
  --backend <name>       auto | mlx | webgpu | cpu (default auto)
  --dtype <f16|f32>      Weight/compute dtype (default f16)
  --prompt <style>       compact | detailed (default compact)
  --optimize             Ask the agent for 16-token buckets + prefix cache (if supported)
  --seed <n>             Seed (default 7); round r uses seed + r - 1
  --pace <n>             Paced gravity-steps per second when spectating (default 3)
  --max-speed            One step per completed inference, no pacing
  --duration <s>         Stop after this many seconds (excluding warmup)
  --steps <n>            Stop after this many PIECES locked (per episode with --episodes)
  --unassisted           Execute raw Laya top-1 at lock time; disable the safety shield
  --record <file.jsonl>  Write per-step decisions and boards (laya-tetris-v2 JSONL)
  --headless             No terminal display
  --episodes <n>         Benchmark: n uncapped episodes (seeds seed..seed+n-1), prints pieces/s
  --no-alt-screen        Keep the final frame in scrollback
  --online               Allow downloading a missing checkpoint

Keys: SPACE pause · ↓ hold to speed up each step · R reset · Q quit

Every downward move is a real decision now, not a cosmetic replay of one
per-piece choice: at each gravity step the model picks a rotation, a
horizontal direction and a distance; once the piece can no longer descend
it gets exactly one more such decision (a bounded "lock delay") before it
locks. This means many more, much smaller predict() calls per piece than
before -- expect it to feel slower per piece, especially headless.`;

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
      seed: { type: "string", default: "7" },
      pace: { type: "string", default: "3" },
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
  const seed = Number(a.seed);
  let pace = positive("pace", a.pace)!;
  const duration = positive("duration", a.duration);
  const steps = a.steps === undefined ? undefined : Number(a.steps);
  if (steps !== undefined && !(Number.isInteger(steps) && steps >= 1)) throw new UsageError("--steps must be positive");
  const episodes = a.episodes === undefined ? undefined : Number(a.episodes);
  if (episodes !== undefined && !(Number.isInteger(episodes) && episodes >= 1)) throw new UsageError("--episodes must be positive");
  const interactive = !a.headless && episodes === undefined;
  if (interactive && !process.stdout.isTTY) {
    throw new UsageError("Interactive display needs a TTY. Use --headless for a non-interactive run.");
  }

  const model = a.model ?? DEFAULT_MODEL;
  process.stderr.write(`Loading ${model} (${backend}, ${a.dtype})${a.online ? "" : "; offline, no network requests"}...\n`);
  const loadStart = performance.now();
  const { agent, engine } = await loadTetrisAgent(model, {
    backend,
    dtype: a.dtype,
    offline: !a.online,
    optimize: a.optimize,
  });
  process.stderr.write(`Loaded in ${((performance.now() - loadStart) / 1000).toFixed(1)} s on ${engine}\n`);
  const hardware = hardwareName();
  const policy = new LayaPolicy(agent, { guarded: !a.unassisted, prompt: a.prompt as PromptStyle });

  // Warmup: compiles/JITs backend pipelines before the clock starts, like the other games'.
  // A handful of steps is enough -- this only needs to touch every code path once, not play a full piece.
  {
    const warm = new TetrisGame(seed + 10000);
    let position: StepPosition = { row: SPAWN_ROW, rotation: "0", col: spawnColFor(warm.active) };
    let lockChancePending = false;
    for (let i = 0; i < 6 && warm.alive; i++) {
      const step = await policy.decideStep(warm, position, lockChancePending);
      if (lockChancePending) {
        warm.applyPlacement(step.executed!);
        position = { row: SPAWN_ROW, rotation: "0", col: spawnColFor(warm.active) };
        lockChancePending = false;
      } else {
        position = step.canDescend ? { ...step.position, row: step.position.row + 1 } : step.position;
        lockChancePending = !step.canDescend;
      }
    }
  }

  try {
    if (episodes !== undefined) return await benchmark(policy, seed, episodes, steps ?? 200, { hardware, engine, model, backend });
    return await play();
  } finally {
    await agent.dispose?.();
  }

  async function play(): Promise<number> {
    const session = new TetrisSession(policy, seed, { hardware, engine });
    const stats = session.stats;
    let record: WriteStream | undefined;
    if (a.record) {
      mkdirSync(dirname(a.record), { recursive: true });
      record = createWriteStream(a.record, { flags: "wx" });
      record.write(
        JSON.stringify({
          type: "metadata",
          format: "laya-tetris-v2",
          created_utc: new Date().toISOString(),
          model: { name: model, hardware, engine, runtime: runtimeName(), prompt: a.prompt, network: a.online ? "online" : "offline" },
          settings: a,
          note: "One real prediction per gravity step, not per piece. Board is shown before the announced position/placement.",
        }) + "\n",
      );
    }
    const out = process.stdout;
    const keys = interactive ? new Keyboard() : undefined;
    const altScreen = interactive && !a["no-alt-screen"];
    const noColor = !!process.env.NO_COLOR;
    if (interactive) out.write((altScreen ? "\x1b[?1049h" : "") + "\x1b[?25l\x1b[2J");
    const draw = (text: string) => out.write("\x1b[H" + text);
    let displayed: { board: GameSnapshot; step?: StepDecision } = { board: session.game.snapshot() };
    const inference: number[] = [];
    let calls = 0;
    let quit = false;
    const onSigint = () => (quit = true);
    process.on("SIGINT", onSigint);

    const NORMAL_STEP_MS = () => 1000 / pace;
    const FAST_STEP_MS = 12;

    function overlayFor(board: GameSnapshot, step: StepDecision): FallingPiece {
      return { kind: board.active, rotation: step.position.rotation, col: step.position.col, row: step.position.row };
    }

    session.restartClock();
    try {
      while (!quit) {
        if ((duration && session.elapsedMs >= duration * 1000) || (steps && stats.pieces >= steps)) break;
        const pressed = keys?.read().toLowerCase() ?? "";
        if (pressed.includes("q") || pressed.includes("\x03")) break;
        if (pressed.includes(" ")) stats.paused = !stats.paused;
        if (pressed.includes("+")) pace = Math.min(60, pace + 1);
        if (pressed.includes("-")) pace = Math.max(1, pace - 1);
        if (pressed.includes("r")) {
          session.reset();
          displayed = { board: session.game.snapshot() };
        }
        if (stats.paused) {
          stats.elapsed = session.elapsedMs / 1000;
          if (interactive) draw(compose(displayed.board, displayed.step ?? {}, stats, displayed.step ? overlayFor(displayed.board, displayed.step) : undefined).ansi(!noColor));
          await sleep(30);
          continue;
        }
        if (interactive) {
          const [mw, mh] = layoutSize();
          if ((out.columns ?? 0) < mw || (out.rows ?? 0) < mh) {
            out.write(`\x1b[2J\x1b[HResize terminal to at least ${mw} columns × ${mh} rows.\nThe game is waiting. Q quits.`);
            await sleep(100);
            continue;
          }
        }
        const t0 = performance.now();
        const result = await session.stepDecide();
        const advance = session.stepAdvance(result);
        calls++;
        inference.push(result.step.inference_ms);
        displayed = { board: result.board, step: result.step };
        record?.write(JSON.stringify({ type: "step", at: session.elapsedMs / 1000, game: result.board, position: result.position, is_lock_chance: result.isLockChance, step: result.step }) + "\n");
        if (interactive) {
          const fast = pressed.includes("\x1b[b") || pressed.includes("s");
          draw(compose(result.board, result.step, stats, overlayFor(result.board, result.step)).ansi(!noColor));
          if (!a["max-speed"]) {
            const remaining = (fast ? FAST_STEP_MS : NORMAL_STEP_MS()) - (performance.now() - t0);
            if (remaining > 0) await sleep(remaining);
          }
        }
        if (advance.locked) {
          record?.write(JSON.stringify({ type: "piece", at: session.elapsedMs / 1000, game: session.game.snapshot(), step: result.step, stats: { ...stats } }) + "\n");
          if (advance.roundEnd) {
            record?.write(JSON.stringify({ type: "round_end", at: session.elapsedMs / 1000, game: advance.roundEnd }) + "\n");
            if (advance.stop) break;
            if (interactive) {
              draw(compose(advance.roundEnd, {}, stats).ansi(!noColor));
              await sleep(1200);
            }
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
      pieces: stats.pieces,
      inference_calls: calls,
      seconds,
      steps_per_second: seconds ? calls / seconds : 0,
      pieces_per_second: seconds ? stats.pieces / seconds : 0,
      score: session.game.score,
      lines: session.game.linesCleared,
      level: session.game.level,
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

/** `--episodes N --headless`: uncapped episodes, same protocol as the other games' benchmarks. Now a per-step loop against a bare `TetrisGame`, no rendering, no `TetrisSession`. */
async function benchmark(
  policy: LayaPolicy,
  seed0: number,
  episodes: number,
  maxPieces: number,
  meta: { hardware: string; engine: string; model: string; backend: string },
): Promise<number> {
  const results = [];
  let totalPieces = 0;
  let totalSteps = 0;
  let totalSeconds = 0;
  let deaths = 0;
  let interventions = 0;
  const allInference: number[] = [];
  for (let e = 0; e < episodes; e++) {
    const seed = seed0 + e;
    const game = new TetrisGame(seed);
    let position: StepPosition = { row: SPAWN_ROW, rotation: "0", col: spawnColFor(game.active) };
    let lockChancePending = false;
    const inference: number[] = [];
    let episodeInterventions = 0;
    let pieces = 0;
    let stepCount = 0;
    const started = performance.now();
    while (game.alive && pieces < maxPieces) {
      const isLockChance = lockChancePending;
      const step = await policy.decideStep(game, position, isLockChance);
      inference.push(step.inference_ms);
      stepCount++;
      if (isLockChance) {
        episodeInterventions += step.intervened ? 1 : 0;
        game.applyPlacement(step.executed!);
        pieces++;
        lockChancePending = false;
        position = { row: SPAWN_ROW, rotation: "0", col: spawnColFor(game.active) };
      } else {
        position = step.canDescend ? { ...step.position, row: step.position.row + 1 } : step.position;
        lockChancePending = !step.canDescend;
      }
    }
    const seconds = (performance.now() - started) / 1000;
    const r = {
      seed,
      pieces,
      steps: stepCount,
      seconds,
      pieces_per_second: pieces / seconds,
      steps_per_second: stepCount / seconds,
      score: game.score,
      lines: game.linesCleared,
      level: game.level,
      alive: game.alive,
      interventions: episodeInterventions,
      inference_ms: {
        mean: inference.reduce((x, y) => x + y, 0) / inference.length,
        p50: percentile(inference, 0.5),
        p95: percentile(inference, 0.95),
        p99: percentile(inference, 0.99),
        max: Math.max(...inference),
      },
    };
    results.push(r);
    totalPieces += r.pieces;
    totalSteps += r.steps;
    totalSeconds += seconds;
    deaths += game.alive ? 0 : 1;
    interventions += episodeInterventions;
    allInference.push(...inference);
    console.error(
      `${policy.guarded ? "shield" : "top-1"} seed=${seed} pieces=${r.pieces} steps=${r.steps} score=${r.score} lines=${r.lines} alive=${r.alive} ` +
        `actual=${r.pieces_per_second.toFixed(1)} pieces/s (${r.steps_per_second.toFixed(1)} steps/s) p50=${r.inference_ms.p50.toFixed(1)}ms interventions=${episodeInterventions}`,
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
    pieces: totalPieces,
    steps: totalSteps,
    seconds: totalSeconds,
    pieces_per_second: totalPieces / totalSeconds,
    steps_per_second: totalSteps / totalSeconds,
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
      console.error(`laya-tetris: ${error.message}\n\n${HELP}`);
      process.exit(2);
    }
    console.error(error);
    process.exit(1);
  },
);
