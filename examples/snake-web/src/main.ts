/**
 * Laya Snake in the browser: the same game, planner, prompt and cycle safety
 * shield as examples/snake-terminal (shared core), running the model on WebGPU.
 */
import { LayaPolicy, SnakeGame, SnakeSession, type Decision, type GameSnapshot } from "../../snake-terminal/src/core/index.ts";
import { detectWebGpu } from "../../web-playground/src/lib/gpu.ts";
import { loadBrowserAgent, type BrowserAgent } from "../../web-playground/src/lib/loader.ts";
import { CHECKPOINTS, formatBytes } from "../../web-playground/src/lib/models.ts";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const DIRS = ["UP", "DOWN", "LEFT", "RIGHT"] as const;
const BOARD = { width: 24, height: 16, seed: 7, initialLength: 6 };

let agent: BrowserAgent | undefined;
let session: SnakeSession | undefined;
let paused = false;
let stopped = false; // unassisted game over
let fps = 12;
let maxSpeed = false;
let shown: { board: GameSnapshot; decision?: Decision; over?: boolean } = { board: new SnakeGame(BOARD.width, BOARD.height, BOARD.seed, BOARD.initialLength).snapshot() };
let dirty = true;
let resetRequested = false;

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
  const cell = Math.min(w / g.width, h / g.height);
  const ox = (w - cell * g.width) / 2;
  const oy = (h - cell * g.height) / 2;
  ctx.fillStyle = css("--bg");
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = css("--dot");
  for (let y = 0; y < g.height; y++)
    for (let x = 0; x < g.width; x++) {
      ctx.beginPath();
      ctx.arc(ox + (x + 0.5) * cell, oy + (y + 0.5) * cell, Math.max(1, cell * 0.07), 0, Math.PI * 2);
      ctx.fill();
    }
  const pad = Math.max(1, cell * 0.08);
  const light = css("color-scheme") === "light";
  for (let i = g.body.length - 1; i >= 0; i--) {
    const [x, y] = g.body[i]!;
    const f = 1 - i / Math.max(1, g.body.length);
    // Same gradient as the terminal UI (ui.py): head #dcfff0, tail fades to dark green.
    ctx.fillStyle = i === 0 ? (light ? "#0a8f62" : "#dcfff0") : `rgb(${18 + 64 * f}, ${73 + 150 * f}, ${57 + 102 * f})`;
    ctx.beginPath();
    ctx.roundRect(ox + x * cell + pad, oy + y * cell + pad, cell - 2 * pad, cell - 2 * pad, cell * 0.18);
    ctx.fill();
  }
  if (g.food) {
    const [x, y] = g.food;
    ctx.fillStyle = css("--amber");
    ctx.beginPath();
    ctx.arc(ox + (x + 0.5) * cell, oy + (y + 0.5) * cell, cell * 0.3, 0, Math.PI * 2);
    ctx.fill();
  }
}

const pad3 = (n: number) => String(n).padStart(3, "0");

function render(): void {
  dirty = false;
  const { board: g, decision: d } = shown;
  drawBoard(g);
  const s = session?.stats;
  $("score").textContent = pad3(g.score);
  $("length").textContent = pad3(g.length);
  $("best").textContent = pad3(Math.max(s?.best ?? 0, g.score));
  const fill = g.length / (g.width * g.height);
  $("fill").style.width = `${(100 * fill).toFixed(2)}%`;
  $("fill-pct").textContent = `${(100 * fill).toFixed(1)}%`;
  $("round").textContent = `ROUND ${String(s?.round ?? 1).padStart(2, "0")}`;
  const state = $("state");
  const label = !agent ? "NOT LOADED" : paused ? "PAUSED" : g.won ? "BOARD CLEAR" : !g.alive || shown.over ? "GAME OVER" : "LIVE";
  state.textContent = label;
  state.className = `state${label === "GAME OVER" ? " over" : !agent || label === "PAUSED" ? " idle" : ""}`;

  $("probs").replaceChildren(
    ...DIRS.map((dir) => {
      const p = d?.probabilities[dir] ?? 0;
      const li = document.createElement("li");
      if (d?.proposed === dir) li.className = "sel";
      li.innerHTML = `<span>${d?.proposed === dir ? "›" : " "} ${dir}</span><span class="track"><span class="fill" style="display:block;width:${(100 * p).toFixed(1)}%"></span></span><span class="v">${p.toFixed(2)}</span>`;
      li.setAttribute("aria-label", `${dir} ${p.toFixed(2)}${d?.proposed === dir ? ", proposed" : ""}`);
      return li;
    }),
  );
  $("executed").textContent = d?.executed ?? "—";
  $("shield").hidden = !d?.intervened;
  const risk = d?.dead_end_risk ?? 0;
  const riskBar = $("risk");
  riskBar.style.width = `${100 * risk}%`;
  riskBar.className = `fill ${risk < 0.5 ? "amber" : "red"}`;
  $("risk-v").textContent = risk.toFixed(2);
  $("risk-v").className = `mono ${risk < 0.5 ? "amber" : ""}`;
  $("food").style.width = `${100 * (d?.food_reachable ?? 0)}%`;
  $("food-v").textContent = (d?.food_reachable ?? 0).toFixed(2);
  $("inference").textContent = d ? `${d.inference_ms.toFixed(1)} ms` : "—";
  $("rate").textContent = s ? `${s.steps_per_second.toFixed(1)} /s` : "—";
  $("tokens").textContent = d ? String(d.input_tokens) : "—";
  $("interventions").textContent = String(s?.interventions ?? 0).padStart(4, "0");
  $("deaths").textContent = String(s?.deaths ?? 0);
  $("steps").textContent = String(s?.steps ?? 0);
  $("guard-label").textContent = session?.policy.guarded === false ? "Laya · shield OFF" : "Laya + cycle safety";
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
    if (!maxSpeed) {
      const remaining = 1000 / fps - (performance.now() - t0);
      if (remaining > 0) await sleep(remaining);
    }
    const r = session.advance(result.decision);
    if (r.roundEnd) {
      shown = { board: r.roundEnd, over: !r.roundEnd.alive };
      dirty = true;
      if (r.stop) {
        stopped = true; // shield off: stop at the first death, like Python --unassisted
        continue;
      }
      await sleep(maxSpeed ? 400 : 1000);
      shown = { board: session.game.snapshot() };
      dirty = true;
    }
  }
}

function newSession(): void {
  if (!agent) return;
  const guarded = !$<HTMLInputElement>("unassisted").checked;
  const policy = new LayaPolicy(agent, { guarded, prompt: "compact" });
  session = new SnakeSession(policy, BOARD, { hardware: "Browser", engine: "WebGPU" });
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
    const warm = new SnakeGame(BOARD.width, BOARD.height, BOARD.seed + 10000, BOARD.initialLength);
    for (let i = 0; i < 6 && warm.alive; i++) warm.step((await warmPolicy.decide(warm)).executed);
    newSession();
    $("overlay").hidden = true;
    $<HTMLButtonElement>("play").disabled = false;
    $<HTMLButtonElement>("reset").disabled = false;
    (window as unknown as { __snake: unknown }).__snake = { get session() { return session; }, get agent() { return agent; } };
  } catch (e) {
    console.error(e);
    text.textContent = `Could not load ${repo}: ${(e as Error).message}`;
    btn.disabled = false;
  }
}

// ------------------------------------------------------------------ controls
function setFps(v: number): void {
  fps = Math.max(1, Math.min(60, v));
  $<HTMLInputElement>("fps").value = String(fps);
  $("fps-v").textContent = `${fps}/s`;
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
  $("fps").addEventListener("input", (e) => setFps(Number((e.target as HTMLInputElement).value)));
  $("unassisted").addEventListener("change", () => newSession());
  document.addEventListener("keydown", (e) => {
    const t = e.target as HTMLElement;
    if (t instanceof HTMLInputElement && t.type === "range" && e.key.startsWith("Arrow")) return; // slider handles arrows
    if (t instanceof HTMLSelectElement || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " ") {
      e.preventDefault();
      togglePause();
    } else if (e.key === "ArrowUp" || e.key === "+") {
      e.preventDefault();
      setFps(fps + 2);
    } else if (e.key === "ArrowDown" || e.key === "-") {
      e.preventDefault();
      setFps(fps - 2);
    } else if (e.key === "r" || e.key === "R") reset();
  });
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
