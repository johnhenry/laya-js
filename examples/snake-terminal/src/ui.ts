/**
 * Fixed-cell terminal composition (port of laya-mlx `snake/ui.py`), rendered
 * as 24-bit ANSI escapes instead of Rich text. No dependencies.
 */
import type { GameSnapshot } from "./core/game.ts";
import type { Decision } from "./core/policy.ts";
import type { SessionStats } from "./core/session.ts";

export const BG = "#090f13";
export const FG = "#e3f3ef";
export const MUTED = "#68868c";
export const DIM = "#20353c";
export const GREEN = "#62f5b5";
export const AMBER = "#ffce73";
export const RED = "#ff7c8c";
export const CYAN = "#8ad8e9";

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
    [...digits].forEach((digit, index) =>
      DIGITS[digit]!.forEach((glyphs, line) => this.put(row + line, column + index * 4, glyphs, color)),
    );
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

const hex2 = (n: number) => Math.trunc(n).toString(16).padStart(2, "0");
const fixed = (v: number, d: number, w = 0) => v.toFixed(d).padStart(w);

export function layoutSize(width: number, height: number): [number, number] {
  return [Math.max(104, width * 2 + 50), Math.max(35, height + 19)];
}

export type UiStats = Partial<SessionStats> & { replay?: boolean };

/** Probabilities describe the displayed board, before its announced next step. */
export function compose(game: GameSnapshot, decision: Partial<Decision>, stats: UiStats): Canvas {
  const [width, height] = layoutSize(game.width, game.height);
  const c = new Canvas(width, height);
  const left = 3;
  const right = Math.max(58, game.width * 2 + 10);
  const top = 6;
  const side = width - right - 4;
  const bottom = top + game.height + 1;
  let state = stats.paused ? "PAUSED" : game.won ? "BOARD CLEAR" : !game.alive ? "GAME OVER" : "LIVE";
  if (stats.replay && state === "LIVE") state = "RECORDED RUN · 1×";
  c.put(1, left, "LAYA  /  LOCAL INTELLIGENCE", MUTED);
  c.put(1, width - state.length - 3, state, game.alive ? GREEN : RED);
  c.put(2, left, "─".repeat(width - 6), DIM);
  c.put(4, left, "S N A K E", FG);
  c.put(4, left + 31, `ROUND ${String(stats.round ?? 1).padStart(2, "0")}`, MUTED);
  c.put(top, left, "┌" + "─".repeat(game.width * 2) + "┐", DIM);
  c.put(bottom, left, "└" + "─".repeat(game.width * 2) + "┘", DIM);
  for (let y = 0; y < game.height; y++) {
    c.put(top + 1 + y, left, "│", DIM);
    c.put(top + 1 + y, left + game.width * 2 + 1, "│", DIM);
    c.put(top + 1 + y, left + 1, "· ".repeat(game.width), "#13272e");
  }
  const body = game.body;
  for (let index = body.length - 1; index >= 0; index--) {
    const [x, y] = body[index]!;
    const fraction = 1 - index / Math.max(1, body.length);
    const color =
      index === 0 ? "#dcfff0" : `#${hex2(18 + 64 * fraction)}${hex2(73 + 150 * fraction)}${hex2(57 + 102 * fraction)}`;
    c.put(top + y + 1, left + 1 + 2 * x, "██", color);
  }
  if (game.food) {
    const [x, y] = game.food;
    c.put(top + y + 1, left + 1 + 2 * x, "● ", AMBER);
  }
  for (const [offset, label, value, color] of [
    [0, "SCORE", game.score, GREEN],
    [18, "LENGTH", game.length, FG],
    [36, "BEST", stats.best ?? game.score, MUTED],
  ] as const) {
    c.put(bottom + 2, left + offset, label, MUTED);
    c.number(bottom + 3, left + offset, value, color);
  }
  const fill = game.length / (game.width * game.height);
  c.bar(bottom + 7, left, fill, 41);
  c.put(bottom + 7, left + 43, fixed(100 * fill, 1, 4) + "%", MUTED);

  c.put(4, right, `Laya · ${stats.engine ?? "FP16"}`, GREEN);
  c.put(5, right, `${stats.hardware ?? "Local"} · Local`, MUTED);
  c.put(7, right, "NEXT MOVE", FG);
  c.put(7, right + 15, "MODEL PROBABILITIES", MUTED);
  const probabilities = (decision.probabilities ?? {}) as Record<string, number>;
  (["UP", "DOWN", "LEFT", "RIGHT"] as const).forEach((direction, index) => {
    const row = 9 + index;
    const probability = probabilities[direction] ?? 0;
    const selected = direction === decision.proposed;
    const color = selected ? GREEN : MUTED;
    c.put(row, right, `${selected ? "›" : " "} ${direction.padEnd(5)}`, color);
    c.put(row, right + 9, "░".repeat(18), DIM);
    c.put(row, right + 9, "█".repeat(pyRound(probability * 18)), color);
    c.put(row, right + 29, probability.toFixed(2), color);
  });
  c.put(14, right, "EXECUTING", MUTED);
  c.put(14, right + 12, decision.executed ?? "—", GREEN);
  if (decision.intervened) c.put(14, right + 20, "SHIELD", AMBER);
  c.put(16, right, "DEAD-END RISK", MUTED);
  const risk = decision.dead_end_risk ?? 0;
  c.bar(17, right, risk, Math.min(24, side - 9), risk < 0.5 ? AMBER : RED);
  c.put(17, right + 29, risk.toFixed(2), risk < 0.5 ? AMBER : RED);
  c.put(19, right, "FOOD REACHABLE", MUTED);
  c.bar(20, right, decision.food_reachable ?? 0, Math.min(24, side - 9), CYAN);
  c.put(20, right + 29, (decision.food_reachable ?? 0).toFixed(2), CYAN);
  c.put(22, right, "INFERENCE", MUTED);
  c.put(22, right + 18, `${fixed(decision.inference_ms ?? 0, 1, 5)} ms`, FG);
  c.put(23, right, "DECISIONS", MUTED);
  c.put(23, right + 18, `${fixed(stats.steps_per_second ?? 0, 1, 5)} /s`, FG);
  c.put(24, right, "OUTPUT TOKENS", MUTED);
  c.put(24, right + 18, String(decision.output_tokens ?? 0), FG);
  c.put(25, right, "NETWORK", MUTED);
  c.put(25, right + 18, "OFFLINE", GREEN);
  c.put(26, right, "ENGINE", MUTED);
  c.put(26, right + 18, stats.engine ?? "FP16", MUTED);
  c.put(28, right, stats.guarded ?? true ? "Laya + cycle safety" : "Laya · shield OFF", MUTED);
  c.put(29, right, `Shield interventions  ${String(stats.interventions ?? 0).padStart(4, "0")}`, AMBER);
  c.put(height - 3, left, "─".repeat(width - 6), DIM);
  c.put(height - 2, left, "SPACE pause   ↑/↓ speed   R reset   Q quit", MUTED);
  const elapsed = Math.trunc(stats.elapsed ?? 0);
  const clock = `${String(Math.trunc(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;
  c.put(height - 2, right, `ESTIMATES BY LAYA            ${clock}`, MUTED);
  return c;
}
