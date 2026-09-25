/**
 * Laya Checkers in the browser: the same rules, shield, and three
 * interchangeable opponents (Laya/bot/human) as examples/checkers-terminal
 * (shared core), with a canvas board and click-to-move for a human seat.
 * Unlike Snake/Flappy Bird, a Laya seat is optional here -- bot-vs-bot and
 * any human-involving combination play fully without WebGPU.
 */
import {
  BotActor,
  CheckersRng,
  CheckersSession,
  HumanActor,
  LayaActor,
  LayaPolicy,
  compliantHopsFor,
  makeSquare,
  newGame,
  parseSquare,
  type Actor,
  type Decision,
  type GameSnapshot,
  type Hop,
  type HumanIO,
  type Player,
  type Square,
  type TickResult,
} from "../../checkers-terminal/src/core/index.ts";
import { detectWebGpu } from "../../web-playground/src/lib/gpu.ts";
import { loadBrowserAgent, type BrowserAgent } from "../../web-playground/src/lib/loader.ts";
import { CHECKPOINTS, formatBytes } from "../../web-playground/src/lib/models.ts";

type SeatKind = "laya" | "bot" | "human";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

class BrowserHumanIO implements HumanIO {
  compliant: Hop[] = [];
  #resolve: ((v: string) => void) | null = null;
  async prompt(): Promise<string> {
    return new Promise((resolve) => {
      this.#resolve = resolve;
    });
  }
  submit(index: number): void {
    this.#resolve?.(String(index));
    this.#resolve = null;
  }
}

let agent: BrowserAgent | undefined;
let session: CheckersSession | undefined;
let humanIO: BrowserHumanIO | undefined;
let lastResult: TickResult | undefined;
let selectedFrom: Square | null = null;
let pace = 2;
let maxSpeed = false;
let gpuOk = false;
let dirty = true;

// ------------------------------------------------------------------ rendering
const canvas = $<HTMLCanvasElement>("board");
const ctx = canvas.getContext("2d")!;
const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function rowFiOf(sq: Square): { row: number; fi: number } {
  const { fi, rank } = parseSquare(sq);
  return { row: 8 - rank, fi };
}

function humanTurnNow(): boolean {
  return !!session && session.actors[session.game.toMove].kind === "human";
}

function drawBoard(state: GameSnapshot): void {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const size = Math.max(1, Math.round(Math.min(rect.width, rect.height) * dpr));
  if (canvas.width !== size || canvas.height !== size) {
    canvas.width = size;
    canvas.height = size;
  }
  const cell = size / 8;
  const light = css("--light-square");
  const dark = css("--dark-square");
  for (let row = 0; row < 8; row++) {
    const rank = 8 - row;
    for (let fi = 0; fi < 8; fi++) {
      ctx.fillStyle = (fi + rank) % 2 === 1 ? dark : light;
      ctx.fillRect(fi * cell, row * cell, cell, cell);
    }
  }
  if (humanTurnNow() && humanIO) {
    const selectable = selectedFrom ? [] : [...new Set(humanIO.compliant.map((h) => h.from))];
    const destinations = selectedFrom ? humanIO.compliant.filter((h) => h.from === selectedFrom).map((h) => h.to) : [];
    ctx.fillStyle = "rgba(232, 180, 100, 0.35)";
    for (const sq of selectable) {
      const { row, fi } = rowFiOf(sq);
      ctx.fillRect(fi * cell, row * cell, cell, cell);
    }
    ctx.fillStyle = "rgba(224, 72, 62, 0.45)";
    for (const sq of destinations) {
      const { row, fi } = rowFiOf(sq);
      ctx.fillRect(fi * cell, row * cell, cell, cell);
    }
    if (selectedFrom) {
      const { row, fi } = rowFiOf(selectedFrom);
      ctx.strokeStyle = css("--red");
      ctx.lineWidth = Math.max(2, cell * 0.04);
      ctx.strokeRect(fi * cell + 2, row * cell + 2, cell - 4, cell - 4);
    }
  }
  for (const [sq, piece] of Object.entries(state.board)) {
    if (!piece) continue;
    const { row, fi } = rowFiOf(sq as Square);
    const cx = fi * cell + cell / 2;
    const cy = row * cell + cell / 2;
    ctx.fillStyle = piece.player === "red" ? css("--red") : css("--ivory");
    ctx.beginPath();
    ctx.arc(cx, cy, cell * 0.34, 0, Math.PI * 2);
    ctx.fill();
    if (piece.kind === "king") {
      ctx.strokeStyle = piece.player === "red" ? css("--amber") : css("--bg");
      ctx.lineWidth = Math.max(2, cell * 0.05);
      ctx.beginPath();
      ctx.arc(cx, cy, cell * 0.2, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

function squareFromEvent(e: MouseEvent): Square | null {
  const rect = canvas.getBoundingClientRect();
  const xFrac = (e.clientX - rect.left) / rect.width;
  const yFrac = (e.clientY - rect.top) / rect.height;
  if (xFrac < 0 || xFrac >= 1 || yFrac < 0 || yFrac >= 1) return null;
  const fi = Math.floor(xFrac * 8);
  const rank = 8 - Math.floor(yFrac * 8);
  return makeSquare(fi, rank);
}

function updatePanel(): void {
  const state = $("state");
  state.textContent = !agent && !session ? "NOT LOADED" : "LIVE";
  if (!session) return;
  const s = session.stats;
  $("round").textContent = `ROUND ${String(s.round).padStart(2, "0")}`;
  $("red-wins").textContent = String(s.red_wins);
  $("black-wins").textContent = String(s.black_wins);
  $("turns").textContent = String(s.turns);
  $("interventions").textContent = String(s.interventions).padStart(4, "0");
  $("rate").textContent = `${s.turns_per_second.toFixed(1)} /s`;
  $("to-move").textContent = session.game.toMove.toUpperCase();
  $("to-move").className = session.game.toMove === "red" ? "red" : "";
  $("prompt").hidden = !humanTurnNow();
  const r = lastResult;
  if (r) {
    $("last-actor").textContent = `${r.board.toMove} (${r.actor})`;
    $("last-move").textContent = r.hop.kind === "capture" ? `${r.hop.from}x${r.hop.to}` : `${r.hop.from}-${r.hop.to}`;
    $("shield").hidden = !r.decision?.intervened;
    const d: Decision | undefined = r.decision;
    if (d) {
      const risk = d.material_at_risk;
      $("risk").style.width = `${100 * risk}%`;
      $("risk").className = `fill ${risk < 0.5 ? "amber" : "red"}`;
      $("risk-v").textContent = risk.toFixed(2);
      $("inference").textContent = `${d.inference_ms.toFixed(1)} ms`;
      $("tokens").textContent = String(d.input_tokens);
    }
  }
}

function render(): void {
  dirty = false;
  drawBoard(session?.game ?? newGame());
  updatePanel();
}

function frame(): void {
  if (dirty) render();
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------ game loop
async function loop(): Promise<void> {
  for (;;) {
    if (!session) {
      await sleep(30);
      continue;
    }
    if (humanIO) humanIO.compliant = compliantHopsFor(session.game);
    selectedFrom = null;
    dirty = true;
    const t0 = performance.now();
    let r: TickResult & { stop: boolean };
    try {
      r = await session.tick();
    } catch (e) {
      console.error(e);
      $("state").textContent = `ERROR: ${(e as Error).message}`;
      session = undefined;
      continue;
    }
    lastResult = r;
    dirty = true;
    if (r.roundEnd) {
      await sleep(maxSpeed ? 400 : 1200);
    } else if (!maxSpeed && r.actor !== "human") {
      const remaining = 1000 / pace - (performance.now() - t0);
      if (remaining > 0) await sleep(remaining);
    }
  }
}

// ------------------------------------------------------------------ setup
function makeActor(kind: SeatKind, seed: number): Actor {
  if (kind === "laya") return new LayaActor(new LayaPolicy(agent!, { prompt: "compact" }));
  if (kind === "bot") return new BotActor(new CheckersRng(seed));
  return new HumanActor(humanIO!);
}

function startNewGame(redKind: SeatKind, blackKind: SeatKind): void {
  humanIO = new BrowserHumanIO();
  selectedFrom = null;
  const actors: Record<Player, Actor> = { red: makeActor(redKind, 1), black: makeActor(blackKind, 1_000_000) };
  session = new CheckersSession(actors, { hardware: "Browser", engine: agent ? `WebGPU · ${agent.dtype.toUpperCase()}` : "—" });
  lastResult = undefined;
  dirty = true;
}

async function loadAndPlay(): Promise<void> {
  const redKind = $<HTMLSelectElement>("red-seat").value as SeatKind;
  const blackKind = $<HTMLSelectElement>("black-seat").value as SeatKind;
  const needsAgent = redKind === "laya" || blackKind === "laya";
  const btn = $<HTMLButtonElement>("load");
  const text = $("progress-text");
  btn.disabled = true;
  if (needsAgent && !agent) {
    if (!gpuOk) {
      text.textContent = "WebGPU is required for a Laya seat -- pick bot or human for both sides instead, or use a supported browser.";
      btn.disabled = false;
      return;
    }
    const repo = $<HTMLSelectElement>("model").value;
    const bar = $<HTMLProgressElement>("progress");
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
      text.textContent = `Loaded in ${res.seconds.toFixed(1)} s.`;
    } catch (e) {
      console.error(e);
      text.textContent = `Could not load ${repo}: ${(e as Error).message}`;
      btn.disabled = false;
      return;
    }
  }
  startNewGame(redKind, blackKind);
  $("overlay").hidden = true;
  $<HTMLButtonElement>("new-game").disabled = false;
}

async function main(): Promise<void> {
  const sel = $<HTMLSelectElement>("model");
  for (const c of CHECKPOINTS) sel.append(new Option(`${c.label} · ${formatBytes(c.approxBytes)}`, c.repo));
  $("load").addEventListener("click", loadAndPlay);
  $("new-game").addEventListener("click", () => {
    const redKind = $<HTMLSelectElement>("red-seat").value as SeatKind;
    const blackKind = $<HTMLSelectElement>("black-seat").value as SeatKind;
    if ((redKind === "laya" || blackKind === "laya") && !agent) {
      $("overlay").hidden = false;
      return;
    }
    startNewGame(redKind, blackKind);
  });
  $("max").addEventListener("change", (e) => (maxSpeed = (e.target as HTMLInputElement).checked));
  $("pace").addEventListener("input", (e) => {
    pace = Number((e.target as HTMLInputElement).value);
    $("pace-v").textContent = `${pace}/s`;
  });
  canvas.addEventListener("click", (e) => {
    if (!session || !humanIO || !humanTurnNow()) return;
    const sq = squareFromEvent(e);
    if (!sq) return;
    if (selectedFrom) {
      const idx = humanIO.compliant.findIndex((h) => h.from === selectedFrom && h.to === sq);
      if (idx >= 0) {
        humanIO.submit(idx + 1);
        selectedFrom = null;
        dirty = true;
        return;
      }
      selectedFrom = humanIO.compliant.some((h) => h.from === sq) ? sq : null;
      dirty = true;
      return;
    }
    if (humanIO.compliant.some((h) => h.from === sq)) {
      selectedFrom = sq;
      dirty = true;
    }
  });
  new ResizeObserver(() => (dirty = true)).observe(canvas);
  matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => (dirty = true));
  requestAnimationFrame(frame);
  void loop();

  const gpu = await detectWebGpu();
  gpuOk = gpu.ok;
  if (!gpu.ok) {
    const b = $("unsupported");
    b.hidden = false;
    b.innerHTML = `<strong>WebGPU is not available.</strong> `;
    b.append(gpu.reason ?? "", " Bot-vs-bot and any human-involving game still work without it -- only a Laya seat needs it.");
  } else {
    $("hardware").textContent = `${gpu.adapter ?? "GPU"} · ${gpu.f16 ? "shader-f16" : "f32 fallback"} · Local`;
  }
}

void main();
