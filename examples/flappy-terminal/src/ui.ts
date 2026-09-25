/**
 * Fixed-cell terminal composition, rendered as 24-bit ANSI escapes. Same
 * `Canvas`/layout approach as `snake-terminal/src/ui.ts` (same bare-bones,
 * dark-dashboard style, no dependencies), with a Flappy Bird palette: the
 * same near-black canvas, but a sky-blue/green/amber accent set instead of
 * Snake's mint-green/amber/red/cyan one -- same visual language, different
 * game.
 */
import type { GameSnapshot } from "./core/game.ts";
import type { Decision } from "./core/policy.ts";
import type { SessionStats } from "./core/session.ts";

export const BG = "#081018";
export const FG = "#eaf6ff";
export const MUTED = "#5f7f96";
export const DIM = "#16283a";
export const SKY = "#7dd3fc";
export const GREEN = "#34d399";
export const AMBER = "#fbbf6d";
export const RED = "#ff7c8c";

const DIGITS: Record<string, [string, string, string]> = {
  "0": ["█▀█", "█ █", "▀▀▀"],
  "1": ["▄█ ", " █ ", "▀▀▀"],
  "2": ["▀▀█", "█▀▀", "▀▀▀"],
  "3": ["▀▀█", "▀▀█", "▀▀▀"],
  "4": ["█ █", "▀▀█", "  ▀"],
  "5": ["█▀▀", "▀▀█", "▀▀▀"],
  "6": ["█▀▀", "█▀█", "▀▀▀"],
  "7": ["▀▀█", "  █", "  ▀"],
  "8": ["█▀█", "█▀█", "▀▀▀"],
  "9": ["█▀█", "▀▀█", "▀▀▀"],
};

export class Canvas {
  readonly width: number;
  readonly height: number;
  chars: string[][];
  styles: string[][];

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.chars = Array.from({ length: height }, () => Array<string>(width).fill(" "));
    this.styles = Array.from({ length: height }, () => Array<string>(width).fill(FG));
  }

  put(row: number, column: number, text: string | number, color = FG): void {
    if (row < 0 || row >= this.height) return;
    [...String(text)].forEach((ch, offset) => {
      const x = column + offset;
      if (x >= 0 && x < this.width) {
        this.chars[row]![x] = ch;
        this.styles[row]![x] = color;
      }
    });
  }

  bar(row: number, column: number, value: number, length = 20, color = GREEN): void {
    const count = pyRound(Math.max(0, Math.min(1, value)) * length);
    this.put(row, column, "━".repeat(length), DIM);
    this.put(row, column, "━".repeat(count), color);
  }

  number(row: number, column: number, value: number, color = GREEN): void {
    const digits = String(value).padStart(3, "0");
    [...digits].forEach((digit, index) => DIGITS[digit]!.forEach((glyphs, line) => this.put(row + line, column + index * 4, glyphs, color)));
  }

  /** ANSI truecolor (or plain text with `color = false`, e.g. NO_COLOR). */
  ansi(color = true): string {
    const lines: string[] = [];
    for (let r = 0; r < this.height; r++) {
      let line = "";
      let current = "";
      for (let x = 0; x < this.width; x++) {
        const style = this.styles[r]![x]!;
        if (color && style !== current) {
          line += fg(style);
          current = style;
        }
        line += this.chars[r]![x];
      }
      lines.push(color ? bg(BG) + line + "\x1b[0m" : line);
    }
    return lines.join("\n");
  }
}

function rgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
const fg = (hex: string) => `\x1b[38;2;${rgb(hex).join(";")}m`;
const bg = (hex: string) => `\x1b[48;2;${rgb(hex).join(";")}m`;

/** CPython round(): half to even. */
export function pyRound(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

const fixed = (v: number, d: number, w = 0) => v.toFixed(d).padStart(w);

export function layoutSize(width: number, height: number): [number, number] {
  return [Math.max(104, width * 2 + 50), Math.max(33, height + 19)];
}

export type UiStats = Partial<SessionStats> & { replay?: boolean };

/** Probabilities describe the displayed board, before its announced next action. */
export function compose(game: GameSnapshot, decision: Partial<Decision>, stats: UiStats): Canvas {
  const [width, height] = layoutSize(game.width, game.height);
  const c = new Canvas(width, height);
  const left = 3;
  const right = Math.max(58, game.width * 2 + 10);
  const top = 6;
  const side = width - right - 4;
  const bottom = top + game.height + 1;
  let state = stats.paused ? "PAUSED" : !game.alive ? "GAME OVER" : "LIVE";
  if (stats.replay && state === "LIVE") state = "RECORDED RUN · 1×";
  c.put(1, left, "LAYA  /  LOCAL INTELLIGENCE", MUTED);
  c.put(1, width - state.length - 3, state, game.alive ? SKY : RED);
  c.put(2, left, "─".repeat(width - 6), DIM);
  c.put(4, left, "F L A P P Y   B I R D", FG);
  c.put(4, left + 31, `ROUND ${String(stats.round ?? 1).padStart(2, "0")}`, MUTED);
  c.put(top, left, "┌" + "─".repeat(game.width * 2) + "┐", DIM);
  c.put(bottom, left, "└" + "─".repeat(game.width * 2) + "┘", DIM);
  for (let y = 0; y < game.height; y++) {
    c.put(top + 1 + y, left, "│", DIM);
    c.put(top + 1 + y, left + game.width * 2 + 1, "│", DIM);
    c.put(top + 1 + y, left + 1, "· ".repeat(game.width), "#0f2334");
  }
  c.put(top + game.height, left + 1, "▀".repeat(game.width * 2), AMBER); // ground line, the row just above the box border
  for (const p of game.pipes) {
    const col = Math.round(p.x);
    if (col < 0 || col >= game.width) continue;
    for (let y = 0; y < game.height - 1; y++) {
      if (y >= p.gap_y && y < p.gap_y + game.gap_height) continue;
      c.put(top + 1 + y, left + 1 + 2 * col, "██", GREEN);
    }
  }
  const birdRow = Math.round(game.bird_y);
  c.put(top + 1 + birdRow, left + 1 + 2 * game.bird_x, "◤◢", AMBER);
  for (const [offset, label, value, color] of [
    [0, "SCORE", game.score, SKY],
    [18, "BEST", stats.best ?? game.score, MUTED],
  ] as const) {
    c.put(bottom + 2, left + offset, label, MUTED);
    c.number(bottom + 3, left + offset, value, color);
  }

  c.put(4, right, `Laya · ${stats.engine ?? "FP16"}`, SKY);
  c.put(5, right, `${stats.hardware ?? "Local"} · Local`, MUTED);
  c.put(7, right, "NEXT ACTION", FG);
  c.put(7, right + 15, "MODEL PROBABILITIES", MUTED);
  const probabilities = (decision.probabilities ?? {}) as Record<string, number>;
  (["FLAP", "NOFLAP"] as const).forEach((action, index) => {
    const row = 9 + index;
    const probability = probabilities[action] ?? 0;
    const selected = action === decision.proposed;
    const color = selected ? SKY : MUTED;
    c.put(row, right, `${selected ? "›" : " "} ${action.padEnd(6)}`, color);
    c.put(row, right + 10, "░".repeat(18), DIM);
    c.put(row, right + 10, "█".repeat(pyRound(probability * 18)), color);
    c.put(row, right + 30, probability.toFixed(2), color);
  });
  c.put(12, right, "EXECUTING", MUTED);
  c.put(12, right + 12, decision.executed ?? "—", SKY);
  if (decision.intervened) c.put(12, right + 22, "SHIELD", AMBER);
  c.put(14, right, "COLLISION RISK", MUTED);
  const risk = decision.collision_risk ?? 0;
  c.bar(15, right, risk, Math.min(24, side - 9), risk < 0.5 ? AMBER : RED);
  c.put(15, right + 29, risk.toFixed(2), risk < 0.5 ? AMBER : RED);
  c.put(17, right, "GAP ALIGNED", MUTED);
  c.bar(18, right, decision.gap_aligned ?? 0, Math.min(24, side - 9), GREEN);
  c.put(18, right + 29, (decision.gap_aligned ?? 0).toFixed(2), GREEN);
  c.put(20, right, "INFERENCE", MUTED);
  c.put(20, right + 18, `${fixed(decision.inference_ms ?? 0, 1, 5)} ms`, FG);
  c.put(21, right, "DECISIONS", MUTED);
  c.put(21, right + 18, `${fixed(stats.steps_per_second ?? 0, 1, 5)} /s`, FG);
  c.put(22, right, "OUTPUT TOKENS", MUTED);
  c.put(22, right + 18, String(decision.output_tokens ?? 0), FG);
  c.put(23, right, "NETWORK", MUTED);
  c.put(23, right + 18, "OFFLINE", GREEN);
  c.put(24, right, "ENGINE", MUTED);
  c.put(24, right + 18, stats.engine ?? "FP16", MUTED);
  c.put(26, right, stats.guarded ?? true ? "Laya + lookahead shield" : "Laya · shield OFF", MUTED);
  c.put(27, right, `Shield interventions  ${String(stats.interventions ?? 0).padStart(4, "0")}`, AMBER);
  c.put(height - 3, left, "─".repeat(width - 6), DIM);
  c.put(height - 2, left, "SPACE pause   ↑/↓ speed   R reset   Q quit", MUTED);
  const elapsed = Math.trunc(stats.elapsed ?? 0);
  const clock = `${String(Math.trunc(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;
  c.put(height - 2, right, `ESTIMATES BY LAYA            ${clock}`, MUTED);
  return c;
}
