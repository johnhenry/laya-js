/**
 * Laya Dino Run in the browser: the same game, reactive lookahead planner,
 * prompt and shield as examples/dino-terminal (shared core), running the
 * model on WebGPU. No prediction is requested on airborne ticks -- see
 * dino-terminal/src/core/game.ts for why.
 */
import { ACTIONS, LayaPolicy, DinoGame, DinoSession, type Decision, type GameSnapshot } from "../../dino-terminal/src/core/index.ts";
import { detectWebGpu } from "../../web-playground/src/lib/gpu.ts";
import { loadBrowserAgent, type BrowserAgent } from "../../web-playground/src/lib/loader.ts";
import { CHECKPOINTS, formatBytes } from "../../web-playground/src/lib/models.ts";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const BOARD = { width: 60, seed: 7 };

let agent: BrowserAgent | undefined;
let session: DinoSession | undefined;
let paused = false;
let stopped = false; // unassisted game over
let fps = 20;
let maxSpeed = false;
let shown: { board: GameSnapshot; decision?: Decision | null; over?: boolean } = { board: new DinoGame(BOARD).snapshot() };
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
  const cell = w / g.width;
  const groundY = h * 0.72;
  const airY = h * 0.42;
  const skyY = h * 0.14;
  ctx.fillStyle = css("--bg");
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = css("--ground");
  ctx.fillRect(0, groundY, w, Math.max(2, h * 0.02));
  ctx.fillStyle = css("--dot");
  for (let x = 0; x < g.width; x += 3) {
    ctx.beginPath();
    ctx.arc((x + 0.5) * cell, groundY + h * 0.06, Math.max(1, cell * 0.06), 0, Math.PI * 2);
    ctx.fill();
  }

  const airborne = g.airborne_ticks_left > 0;
  for (const o of g.obstacles) {
    const px = o.x * cell;
    if (px < -cell || px > w + cell) continue;
    if (o.kind === "cactus") {
      ctx.fillStyle = css("--cactus");
      ctx.beginPath();
      ctx.roundRect(px - cell * 0.3, groundY - h * 0.16, cell * 0.6, h * 0.16, cell * 0.1);
      ctx.fill();
    } else {
      ctx.fillStyle = css("--ptero");
      const y = o.kind === "pterodactyl-low" ? airY : skyY;
      ctx.beginPath();
      ctx.roundRect(px - cell * 0.5, y - h * 0.04, cell, h * 0.08, cell * 0.15);
      ctx.fill();
    }
  }

  const dinoX = g.dino_x * cell;
  const dinoY = airborne ? airY : groundY - (g.ducking ? h * 0.06 : h * 0.16);
  const dinoH = g.ducking && !airborne ? h * 0.08 : h * 0.16;
  ctx.fillStyle = css("--amber");
  ctx.beginPath();
  ctx.roundRect(dinoX - cell * 0.4, dinoY, cell * 0.8, dinoH, cell * 0.12);
  ctx.fill();
}

const pad4 = (n: number) => String(n).padStart(4, "0");

function render(): void {
  dirty = false;
  const { board: g, decision: d } = shown;
  drawBoard(g);
  const s = session?.stats;
  $("score").textContent = pad4(g.score);
  $("best").textContent = pad4(Math.max(s?.best ?? 0, g.score));
  $("speed").textContent = g.speed.toFixed(2);
  $("round").textContent = `ROUND ${String(s?.round ?? 1).padStart(2, "0")}`;
  const state = $("state");
  const label = !agent ? "NOT LOADED" : paused ? "PAUSED" : !g.alive || shown.over ? "GAME OVER" : "LIVE";
  state.textContent = label;
  state.className = `state${label === "GAME OVER" ? " over" : !agent || label === "PAUSED" ? " idle" : ""}`;

  const airborne = g.airborne_ticks_left > 0;
  $("probs").replaceChildren(
    ...ACTIONS.map((action) => {
      const p = d?.probabilities?.[action] ?? 0;
      const li = document.createElement("li");
      if (d?.proposed === action) li.className = "sel";
      li.innerHTML = `<span>${d?.proposed === action ? "›" : " "} ${action}</span><span class="track"><span class="fill" style="display:block;width:${(100 * p).toFixed(1)}%"></span></span><span class="v">${p.toFixed(2)}</span>`;
      li.setAttribute("aria-label", `${action} ${p.toFixed(2)}${d?.proposed === action ? ", proposed" : ""}`);
      return li;
    }),
  );
  $("exec-label").textContent = airborne ? "AIRBORNE" : "EXECUTING";
  $("executed").textContent = airborne ? "—" : (d?.executed ?? "—");
  $("shield").hidden = !d?.intervened;
  const risk = d?.risk ?? 0;
  const riskBar = $("risk");
  riskBar.style.width = `${100 * risk}%`;
  $("risk-v").textContent = risk.toFixed(2);
  $("low").style.width = `${100 * (d?.low_obstacle ?? 0)}%`;
  $("low-v").textContent = (d?.low_obstacle ?? 0).toFixed(2);
  $("inference").textContent = d ? `${d.inference_ms.toFixed(1)} ms` : "—";
  $("rate").textContent = s ? `${s.steps_per_second.toFixed(1)} /s` : "—";
  $("tokens").textContent = d ? String(d.input_tokens) : "—";
  $("interventions").textContent = String(s?.interventions ?? 0).padStart(4, "0");
  $("deaths").textContent = String(s?.deaths ?? 0);
  $("steps").textContent = String(s?.steps ?? 0);
  $("guard-label").textContent = session?.policy.guarded === false ? "Laya · shield OFF" : "Laya + reactive shield";
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
        stopped = true; // shield off: stop at the first death, like the terminal demo's --unassisted
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
  session = new DinoSession(policy, BOARD, { hardware: "Browser", engine: "WebGPU" });
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
    const warm = new DinoGame({ ...BOARD, seed: BOARD.seed + 10000 });
    for (let i = 0; i < 6 && warm.alive; i++) {
      if (warm.airborne) {
        warm.step("RUN");
        continue;
      }
      const d = await warmPolicy.decide(warm);
      warm.step(d?.executed ?? "RUN");
    }
    newSession();
    $("overlay").hidden = true;
    $<HTMLButtonElement>("play").disabled = false;
    $<HTMLButtonElement>("reset").disabled = false;
    (window as unknown as { __dino: unknown }).__dino = { get session() { return session; }, get agent() { return agent; } };
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
    if (t instanceof HTMLInputElement && t.type === "range" && e.key.startsWith("Arrow")) return;
    if (t instanceof HTMLSelectElement || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " ") {
      e.preventDefault();
      togglePause();
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
