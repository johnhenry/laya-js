/**
 * Laya Tetris in the browser: the same game, safety shield, and prompt as
 * examples/tetris-terminal (shared core), running the model on WebGPU.
 * Decisions are made once per gravity STEP, not once per piece -- see
 * tetris-terminal/src/core/policy.ts for why.
 */
import {
  SPAWN_ROW,
  TetrisGame,
  LayaPolicy,
  TetrisSession,
  shapeOf,
  spawnColFor,
  type GameSnapshot,
  type PieceKind,
  type RotationLabel,
  type StepDecision,
  type StepPosition,
} from "../../tetris-terminal/src/core/index.ts";
import { detectWebGpu } from "../../web-playground/src/lib/gpu.ts";
import { loadBrowserAgent, type BrowserAgent } from "../../web-playground/src/lib/loader.ts";
import { CHECKPOINTS, formatBytes } from "../../web-playground/src/lib/models.ts";

const PIECE_COLORS: Record<PieceKind, string> = {
  I: "#5ee7ff",
  O: "#ffe066",
  T: "#c792ea",
  S: "#7ee787",
  Z: "#ff7c8c",
  J: "#66aaff",
  L: "#ffab66",
};

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

let agent: BrowserAgent | undefined;
let session: TetrisSession | undefined;
let paused = false;
let stopped = false; // unassisted game over
let pace = 3; // gravity-steps per second when spectating, not pieces per second
let maxSpeed = false;
let shown: { board: GameSnapshot; step?: StepDecision; over?: boolean } = { board: new TetrisGame(7).snapshot() };
let dirty = true;
let resetRequested = false;
/** Holding Down speeds up each step; it's never instant, and every step is now a real decision, not a cosmetic frame. */
let downHeld = false;
const NORMAL_STEP_MS = () => 1000 / pace;
const FAST_STEP_MS = 12;

// ------------------------------------------------------------------ rendering
const canvas = $<HTMLCanvasElement>("board");
const ctx = canvas.getContext("2d")!;
const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function drawBoard(g: GameSnapshot, position?: StepPosition): void {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const cols = g.board[0]?.length ?? 10;
  const rows = g.board.length;
  const cell = Math.min(w / cols, h / rows);
  const ox = (w - cell * cols) / 2;
  const oy = (h - cell * rows) / 2;
  ctx.fillStyle = css("--bg");
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = css("--dot");
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      ctx.beginPath();
      ctx.arc(ox + (c + 0.5) * cell, oy + (r + 0.5) * cell, Math.max(1, cell * 0.05), 0, Math.PI * 2);
      ctx.fill();
    }
  const pad = Math.max(1, cell * 0.06);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const kind = g.board[r]![c];
      if (!kind) continue;
      ctx.fillStyle = PIECE_COLORS[kind];
      ctx.beginPath();
      ctx.roundRect(ox + c * cell + pad, oy + r * cell + pad, cell - 2 * pad, cell - 2 * pad, cell * 0.12);
      ctx.fill();
    }
  }
  if (position && g.alive) {
    ctx.fillStyle = PIECE_COLORS[g.active];
    for (const [dr, dc] of shapeOf(g.active, position.rotation)) {
      const r = position.row + dr;
      const c = position.col + dc;
      ctx.beginPath();
      ctx.roundRect(ox + c * cell + pad, oy + r * cell + pad, cell - 2 * pad, cell - 2 * pad, cell * 0.12);
      ctx.fill();
    }
  }
}

const pad4 = (n: number) => String(n).padStart(4, "0");

function render(): void {
  dirty = false;
  const { board: g, step } = shown;
  drawBoard(g, step?.position);
  const s = session?.stats;
  $("score").textContent = pad4(g.score);
  $("lines").textContent = pad4(g.linesCleared);
  $("level").textContent = pad4(g.level);
  $("round").textContent = `ROUND ${String(s?.round ?? 1).padStart(2, "0")}`;
  const state = $("state");
  const label = !agent ? "NOT LOADED" : paused ? "PAUSED" : !g.alive || shown.over ? "GAME OVER" : "LIVE";
  state.textContent = label;
  state.className = `state${label === "GAME OVER" ? " over" : !agent || label === "PAUSED" ? " idle" : ""}`;

  const active = $("active");
  active.textContent = g.active;
  active.style.color = PIECE_COLORS[g.active];
  $("next").textContent = g.queue.slice(0, 5).join(" ");
  $("executed").textContent = step?.executed ? `${step.executed.kind}-${step.executed.rotation}-c${step.executed.col}` : "—";
  $("shield").hidden = !step?.intervened;
  const risk = step?.risk ?? 0;
  const riskBar = $("risk");
  riskBar.style.width = `${100 * risk}%`;
  riskBar.className = `fill ${risk < 0.5 ? "amber" : "red"}`;
  $("risk-v").textContent = risk.toFixed(2);
  $("clears").style.width = `${100 * (step?.clears_signal ?? 0)}%`;
  $("clears-v").textContent = (step?.clears_signal ?? 0).toFixed(2);
  $("inference").textContent = step ? `${step.inference_ms.toFixed(1)} ms` : "—";
  $("rate").textContent = s ? `${s.steps_per_second.toFixed(1)} /s` : "—";
  $("piece-rate").textContent = s ? `${s.pieces_per_second.toFixed(1)} /s` : "—";
  $("tokens").textContent = step ? String(step.input_tokens) : "—";
  $("interventions").textContent = String(s?.interventions ?? 0).padStart(4, "0");
  $("deaths").textContent = String(s?.deaths ?? 0);
  $("pieces").textContent = String(s?.pieces ?? 0);
  $("guard-label").textContent = session?.policy.guarded === false ? "Laya · shield OFF" : "Laya + lock-time shield";
  $("play").firstChild!.textContent = paused ? "Play " : "Pause ";
}

function frame(): void {
  if (dirty) render();
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------ game loop
async function loop(): Promise<void> {
  for (;;) {
    if (!session || paused || stopped) {
      await sleep(30);
      continue;
    }
    if (resetRequested) {
      resetRequested = false;
      session.reset();
      shown = { board: session.game.snapshot() };
      dirty = true;
    }
    const t0 = performance.now();
    let result;
    try {
      result = await session.stepDecide();
    } catch (e) {
      console.error(e);
      paused = true;
      $("state").textContent = `ERROR: ${(e as Error).message}`;
      continue;
    }
    const advance = session.stepAdvance(result);
    shown = { board: result.board, step: result.step };
    dirty = true;
    if (!maxSpeed) {
      const remaining = (downHeld ? FAST_STEP_MS : NORMAL_STEP_MS()) - (performance.now() - t0);
      if (remaining > 0) await sleep(remaining);
    }
    if (advance.roundEnd) {
      shown = { board: advance.roundEnd, over: !advance.roundEnd.alive };
      dirty = true;
      if (advance.stop) {
        stopped = true; // shield off: stop at the first block-out, like the terminal demo's --unassisted
        continue;
      }
      await sleep(maxSpeed ? 400 : 1200);
      shown = { board: session.game.snapshot() };
      dirty = true;
    }
  }
}

function newSession(): void {
  if (!agent) return;
  const guarded = !$<HTMLInputElement>("unassisted").checked;
  const policy = new LayaPolicy(agent, { guarded, prompt: "compact" });
  session = new TetrisSession(policy, 7, { hardware: "Browser", engine: "WebGPU" });
  stopped = false;
  shown = { board: session.game.snapshot() };
  dirty = true;
}

// ------------------------------------------------------------------ load
async function loadAndPlay(): Promise<void> {
  const repo = $<HTMLSelectElement>("model").value;
  const btn = $<HTMLButtonElement>("load");
  const bar = $<HTMLProgressElement>("progress");
  const text = $("progress-text");
  btn.disabled = true;
  bar.hidden = false;
  bar.removeAttribute("value");
  text.textContent = "Resolving files on huggingface.co…";
  const approx = CHECKPOINTS.find((c) => c.repo === repo)?.approxBytes ?? 0;
  try {
    const res = await loadBrowserAgent(repo, {
      onProgress: (p) => {
        const total = Math.max(p.total, approx);
        bar.max = total;
        bar.value = Math.min(p.loaded, total);
        text.textContent = `${formatBytes(p.loaded)} / ${formatBytes(total)} · ${Math.floor((100 * p.loaded) / total)}%`;
      },
    });
    agent = res.agent;
    $("engine").textContent = `WebGPU · ${agent.dtype.toUpperCase()}`;
    text.textContent = `Loaded in ${res.seconds.toFixed(1)} s. Warming up…`;
    // Warmup, like the terminal demo (a handful of steps on seed + 10000): compiles the GPU pipelines.
    const warmPolicy = new LayaPolicy(agent);
    const warm = new TetrisGame(10007);
    let position: StepPosition = { row: SPAWN_ROW, rotation: "0", col: spawnColFor(warm.active) };
    let lockChancePending = false;
    for (let i = 0; i < 6 && warm.alive; i++) {
      const step = await warmPolicy.decideStep(warm, position, lockChancePending);
      if (lockChancePending) {
        warm.applyPlacement(step.executed!);
        position = { row: SPAWN_ROW, rotation: "0" as RotationLabel, col: spawnColFor(warm.active) };
        lockChancePending = false;
      } else {
        position = step.canDescend ? { ...step.position, row: step.position.row + 1 } : step.position;
        lockChancePending = !step.canDescend;
      }
    }
    newSession();
    $("overlay").hidden = true;
    $<HTMLButtonElement>("play").disabled = false;
    $<HTMLButtonElement>("reset").disabled = false;
    (window as unknown as { __tetris: unknown }).__tetris = { get session() { return session; }, get agent() { return agent; } };
  } catch (e) {
    console.error(e);
    text.textContent = `Could not load ${repo}: ${(e as Error).message}`;
    btn.disabled = false;
  }
}

// ------------------------------------------------------------------ controls
function setPace(v: number): void {
  pace = Math.max(1, Math.min(10, v));
  $<HTMLInputElement>("pace").value = String(pace);
  $("pace-v").textContent = `${pace}/s`;
}
function togglePause(): void {
  if (!agent) return;
  if (stopped) {
    newSession();
    return;
  }
  paused = !paused;
  dirty = true;
}
function reset(): void {
  if (!agent) return;
  if (stopped) newSession();
  else resetRequested = true;
}

async function main(): Promise<void> {
  const sel = $<HTMLSelectElement>("model");
  for (const c of CHECKPOINTS) sel.append(new Option(`${c.label} · ${formatBytes(c.approxBytes)}`, c.repo));
  $("load").addEventListener("click", loadAndPlay);
  $("play").addEventListener("click", togglePause);
  $("reset").addEventListener("click", reset);
  $("max").addEventListener("change", (e) => (maxSpeed = (e.target as HTMLInputElement).checked));
  $("pace").addEventListener("input", (e) => setPace(Number((e.target as HTMLInputElement).value)));
  $("unassisted").addEventListener("change", () => newSession());
  document.addEventListener("keydown", (e) => {
    const t = e.target as HTMLElement;
    if (t instanceof HTMLSelectElement || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " ") {
      e.preventDefault();
      togglePause();
    } else if (e.key === "r" || e.key === "R") reset();
    else if (e.key === "ArrowDown") {
      e.preventDefault();
      downHeld = true;
    }
  });
  document.addEventListener("keyup", (e) => {
    if (e.key === "ArrowDown") downHeld = false;
  });
  window.addEventListener("blur", () => (downHeld = false));
  new ResizeObserver(() => (dirty = true)).observe(canvas);
  matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => (dirty = true));
  requestAnimationFrame(frame);
  void loop();

  const gpu = await detectWebGpu();
  if (!gpu.ok) {
    const b = $("unsupported");
    b.hidden = false;
    b.innerHTML = `<strong>WebGPU is not available.</strong> `;
    b.append(gpu.reason ?? "");
    $<HTMLButtonElement>("load").disabled = true;
  } else {
    $("hardware").textContent = `${gpu.adapter ?? "GPU"} · ${gpu.f16 ? "shader-f16" : "f32 fallback"} · Local`;
  }
}

void main();
