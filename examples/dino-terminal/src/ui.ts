/**
 * Fixed-cell terminal composition, rendered as 24-bit ANSI escapes. Same
 * bare-bones, dependency-free approach as the other three games' `ui.ts`,
 * with a desert/sand palette (warm browns, cactus green, dino amber)
 * instead of Snake's mint-green, Flappy's sky-blue, or Checkers'
 * red/black. Three rows tall: sky (high pterodactyls), mid-air (low
 * pterodactyls and the airborne dino), ground (cacti, the grounded dino).
 */
import type { GameSnapshot } from "./core/game.ts";
import type { Decision } from "./core/policy.ts";
import type { SessionStats } from "./core/session.ts";

export const BG = "#12140f";
export const FG = "#f3ecd8";
export const MUTED = "#8a8168";
export const DIM = "#2a2718";
export const GROUND = "#a5793c";
export const CACTUS = "#5fae5f";
export const PTERO = "#e08a72";
export const DINO = "#f0d264";
export const AMBER = "#ffce73";
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

  bar(row: number, column: number, value: number, length = 20, color = AMBER): void {
    const count = pyRound(Math.max(0, Math.min(1, value)) * length);
    this.put(row, column, "━".repeat(length), DIM);
    this.put(row, column, "━".repeat(count), color);
  }

  number(row: number, column: number, value: number, color = AMBER): void {
    const digits = String(value).padStart(4, "0");
    [...digits].forEach((digit, index) => DIGITS[digit]!.forEach((glyphs, line) => this.put(row + line, column + index * 4, glyphs, color)));
  }

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

export function pyRound(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

const fixed = (v: number, d: number, w = 0) => v.toFixed(d).padStart(w);

export function layoutSize(width: number): [number, number] {
  return [Math.max(104, width * 2 + 40), 33];
}

export type UiStats = Partial<SessionStats> & { replay?: boolean };

export function compose(game: GameSnapshot, decision: Partial<Decision> & { proposed?: string; executed?: string }, stats: UiStats): Canvas {
  const [width, height] = layoutSize(game.width);
  const c = new Canvas(width, height);
  const left = 3;
  const right = Math.max(58, game.width * 2 + 10);
  const top = 8;
  const skyRow = top;
  const airRow = top + 1;
  const groundRow = top + 2;
  const side = width - right - 4;

  const state = stats.paused ? "PAUSED" : !game.alive ? "GAME OVER" : "LIVE";
  c.put(1, left, "LAYA  /  LOCAL INTELLIGENCE", MUTED);
  c.put(1, width - state.length - 3, state, game.alive ? DINO : RED);
  c.put(2, left, "─".repeat(width - 6), DIM);
  c.put(4, left, "D I N O   R U N", FG);
  c.put(4, left + 16, `ROUND ${String(stats.round ?? 1).padStart(2, "0")}`, MUTED);

  c.put(top - 1, left, "┌" + "─".repeat(game.width * 2) + "┐", DIM);
  c.put(groundRow + 1, left, "└" + "─".repeat(game.width * 2) + "┘", DIM);
  for (const row of [skyRow, airRow, groundRow]) {
    c.put(row, left, "│", DIM);
    c.put(row, left + game.width * 2 + 1, "│", DIM);
  }
  c.put(groundRow, left + 1, "▀".repeat(game.width * 2), GROUND); // ground line

  const airborne = game.airborne_ticks_left > 0;
  const dinoRow = airborne ? airRow : groundRow;
  const dinoGlyph = airborne ? "◤◢" : game.ducking ? "▂▂" : "◥█";
  c.put(dinoRow, left + 1 + 2 * game.dino_x, dinoGlyph, DINO);

  for (const o of game.obstacles) {
    const col = Math.round(o.x);
    if (col < 0 || col >= game.width) continue;
    if (o.kind === "cactus") c.put(groundRow, left + 1 + 2 * col, "▲▲", CACTUS);
    else if (o.kind === "pterodactyl-low") c.put(airRow, left + 1 + 2 * col, "▬▬", PTERO);
    else c.put(skyRow, left + 1 + 2 * col, "▬▬", PTERO);
  }

  c.put(groundRow + 3, left, "SCORE", MUTED);
  c.number(groundRow + 4, left, game.score, AMBER);
  c.put(groundRow + 3, left + 22, "BEST", MUTED);
  c.number(groundRow + 4, left + 22, stats.best ?? game.score, MUTED);
  c.put(groundRow + 3, left + 44, "SPEED", MUTED);
  c.put(groundRow + 4, left + 44, game.speed.toFixed(2), FG);

  c.put(4, right, `Laya · ${stats.engine ?? "FP16"}`, DINO);
  c.put(5, right, `${stats.hardware ?? "Local"} · Local`, MUTED);
  c.put(7, right, "NEXT ACTION", FG);
  c.put(7, right + 15, "MODEL PROBABILITIES", MUTED);
  const probabilities = (decision.probabilities ?? {}) as Record<string, number>;
  (["JUMP", "DUCK", "RUN"] as const).forEach((action, index) => {
    const row = 9 + index;
    const probability = probabilities[action] ?? 0;
    const selected = action === decision.proposed;
    const color = selected ? DINO : MUTED;
    c.put(row, right, `${selected ? "›" : " "} ${action.padEnd(5)}`, color);
    c.put(row, right + 9, "░".repeat(18), DIM);
    c.put(row, right + 9, "█".repeat(pyRound(probability * 18)), color);
    c.put(row, right + 29, probability.toFixed(2), color);
  });
  c.put(13, right, airborne ? "AIRBORNE" : "EXECUTING", MUTED);
  c.put(13, right + 12, airborne ? "—" : (decision.executed ?? "—"), DINO);
  if (decision.intervened) c.put(13, right + 20, "SHIELD", AMBER);
  c.put(15, right, "COLLISION RISK", MUTED);
  const risk = decision.risk ?? 0;
  c.bar(16, right, risk, Math.min(24, side - 9), risk < 0.5 ? AMBER : RED);
  c.put(16, right + 29, risk.toFixed(2), risk < 0.5 ? AMBER : RED);
  c.put(18, right, "LOW OBSTACLE", MUTED);
  c.bar(19, right, decision.low_obstacle ?? 0, Math.min(24, side - 9), PTERO);
  c.put(19, right + 29, (decision.low_obstacle ?? 0).toFixed(2), PTERO);
  c.put(21, right, "INFERENCE", MUTED);
  c.put(21, right + 18, `${fixed(decision.inference_ms ?? 0, 1, 5)} ms`, FG);
  c.put(22, right, "DECISIONS", MUTED);
  c.put(22, right + 18, `${fixed(stats.steps_per_second ?? 0, 1, 5)} /s`, FG);
  c.put(23, right, "OUTPUT TOKENS", MUTED);
  c.put(23, right + 18, String(decision.output_tokens ?? 0), FG);
  c.put(25, right, stats.guarded ?? true ? "Laya + reactive shield" : "Laya · shield OFF", MUTED);
  c.put(26, right, `Shield interventions  ${String(stats.interventions ?? 0).padStart(4, "0")}`, AMBER);
  c.put(height - 3, left, "─".repeat(width - 6), DIM);
  c.put(height - 2, left, "SPACE pause   R reset   Q quit", MUTED);
  const elapsed = Math.trunc(stats.elapsed ?? 0);
  const clock = `${String(Math.trunc(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;
  c.put(height - 2, right, `ESTIMATES BY LAYA            ${clock}`, MUTED);
  return c;
}
