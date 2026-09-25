/**
 * Laya Tetris in the browser: the same game, safety-margin shield, and
 * prompt as examples/tetris-terminal (shared core), running the model on
 * WebGPU. Decisions are made once per piece, not once per tick -- see
 * tetris-terminal/src/core/game.ts for why.
 */
import { SPAWN_ROW, TetrisGame, LayaPolicy, TetrisSession, shapeOf, type Decision, type GameSnapshot, type PieceKind, type Placement } from "../../tetris-terminal/src/core/index.ts";
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
let pace = 3;
let maxSpeed = false;
let shown: { board: GameSnapshot; decision?: Decision; over?: boolean } = { board: new TetrisGame(7).snapshot() };
let dirty = true;
let resetRequested = false;
/** The piece mid-descent, rendered as an overlay -- not yet part of `shown.board`. */
let fallingPiece: { kind: PieceKind; row: number; col: number; rotation: Placement["rotation"] } | null = null;
/** Holding Down speeds the current piece's fall up (a soft drop); it's never instant. */
let downHeld = false;
const NORMAL_ROW_MS = 45;
const FAST_ROW_MS = 12;

// ------------------------------------------------------------------ rendering
const canvas = $<HTMLCanvasElement>("board");
const ctx = canvas.getContext("2d")!;
const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function drawBoard(g: GameSnapshot): void {
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
  if (fallingPiece) {
    ctx.fillStyle = PIECE_COLORS[fallingPiece.kind];
    for (const [dr, dc] of shapeOf(fallingPiece.kind, fallingPiece.rotation)) {
      const r = fallingPiece.row + dr;
      const c = fallingPiece.col + dc;
      ctx.beginPath();
      ctx.roundRect(ox + c * cell + pad, oy + r * cell + pad, cell - 2 * pad, cell - 2 * pad, cell * 0.12);
      ctx.fill();
    }
  }
}

const pad4 = (n: number) => String(n).padStart(4, "0");

function render(): void {
  dirty = false;
  const { board: g, decision: d } = shown;
  drawBoard(g);
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
  $("executed").textContent = d ? `${d.executed.kind}-${d.executed.rotation}-c${d.executed.col}` : "—";
  $("shield").hidden = !d?.intervened;
  const risk = d?.risk ?? 0;
  const riskBar = $("risk");
  riskBar.style.width = `${100 * risk}%`;
  riskBar.className = `fill ${risk < 0.5 ? "amber" : "red"}`;
  $("risk-v").textContent = risk.toFixed(2);
  $("clears").style.width = `${100 * (d?.clears_signal ?? 0)}%`;
  $("clears-v").textContent = (d?.clears_signal ?? 0).toFixed(2);
  $("inference").textContent = d ? `${d.inference_ms.toFixed(1)} ms` : "—";
  $("rate").textContent = s ? `${s.pieces_per_second.toFixed(1)} /s` : "—";
  $("tokens").textContent = d ? String(d.input_tokens) : "—";
  $("interventions").textContent = String(s?.interventions ?? 0).padStart(4, "0");
  $("deaths").textContent = String(s?.deaths ?? 0);
  $("pieces").textContent = String(s?.pieces ?? 0);
  $("guard-label").textContent = session?.policy.guarded === false ? "Laya · shield OFF" : "Laya + safety-margin shield";
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
      result = await session.decide();
    } catch (e) {
      console.error(e);
      paused = true;
      $("state").textContent = `ERROR: ${(e as Error).message}`;
      continue;
    }
    shown = { board: result.board, decision: result.decision };
    dirty = true;
    if (!maxSpeed) await animateDrop(result.decision.executed);
    if (!maxSpeed) {
      const remaining = 1000 / pace - (performance.now() - t0);
      if (remaining > 0) await sleep(remaining);
    }
    const r = session.advance(result.decision);
    if (r.roundEnd) {
      shown = { board: r.roundEnd, over: !r.roundEnd.alive };
      dirty = true;
      if (r.stop) {
        stopped = true; // shield off: stop at the first block-out, like the terminal demo's --unassisted
        continue;
      }
      await sleep(maxSpeed ? 400 : 1200);
      shown = { board: session.game.snapshot() };
      dirty = true;
    }
  }
}

/**
 * Animate the piece's descent instead of snapping straight to its resting
 * row: the model decides the FINAL placement in one call (no per-tick
 * predictions -- see tetris-terminal/src/core/game.ts), but that's a
 * decision-efficiency choice, not a reason the viewer has to see it
 * teleport. Holding Down speeds the fall up (a soft drop), but never skips
 * straight to instant.
 */
async function animateDrop(target: Placement): Promise<void> {
  for (let row = SPAWN_ROW + 1; row <= target.restRow; row++) {
    fallingPiece = { kind: target.kind, rotation: target.rotation, col: target.col, row };
    dirty = true;
    await sleep(downHeld ? FAST_ROW_MS : NORMAL_ROW_MS);
  }
  fallingPiece = null;
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
    // Warmup, like the terminal demo (6 decisions on seed + 10000): compiles the GPU pipelines.
    const warmPolicy = new LayaPolicy(agent);
    const warm = new TetrisGame(10007);
    for (let i = 0; i < 6 && warm.alive; i++) warm.applyPlacement((await warmPolicy.decide(warm)).executed);
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
