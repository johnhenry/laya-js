/**
 * Fixed-cell terminal composition, rendered as 24-bit ANSI escapes. Same
 * bare-bones, dependency-free approach as the other three games' `ui.ts`,
 * with the classic per-piece guideline color scheme (I cyan, O yellow, T
 * purple, S green, Z red, J blue, L orange) instead of one accent palette
 * -- Tetris is the one game in the family where the pieces themselves
 * carry the color identity, not a single "player" or "bird".
 */
import type { GameSnapshot } from "./core/game.ts";
import type { PieceKind } from "./core/pieces.ts";
import type { Decision } from "./core/policy.ts";
import type { SessionStats } from "./core/session.ts";

export const BG = "#0a0e16";
export const FG = "#eef2fb";
export const MUTED = "#5f6b85";
export const DIM = "#1a2032";
export const AMBER = "#ffce73";
export const RED = "#ff7c8c";

export const PIECE_COLORS: Record<PieceKind, string> = {
  I: "#5ee7ff",
  O: "#ffe066",
  T: "#c792ea",
  S: "#7ee787",
  Z: "#ff7c8c",
  J: "#66aaff",
  L: "#ffab66",
};

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

export function layoutSize(): [number, number] {
  return [100, 33];
}

export type UiStats = Partial<SessionStats> & { replay?: boolean };

export function compose(game: GameSnapshot, decision: Partial<Decision>, stats: UiStats): Canvas {
  const [width, height] = layoutSize();
  const c = new Canvas(width, height);
  const left = 3;
  const boardWidth = game.board[0]?.length ?? 10;
  const boardHeight = game.board.length;
  const right = left + boardWidth * 2 + 8;
  const top = 6;
  const bottom = top + boardHeight + 1;

  const state = stats.paused ? "PAUSED" : !game.alive ? "GAME OVER" : "LIVE";
  c.put(1, left, "LAYA  /  LOCAL INTELLIGENCE", MUTED);
  c.put(1, width - state.length - 3, state, game.alive ? PIECE_COLORS.I : RED);
  c.put(2, left, "─".repeat(width - 6), DIM);
  c.put(4, left, "T E T R I S", FG);
  c.put(4, left + 14, `ROUND ${String(stats.round ?? 1).padStart(2, "0")}`, MUTED);
  c.put(top, left, "┌" + "─".repeat(boardWidth * 2) + "┐", DIM);
  c.put(bottom, left, "└" + "─".repeat(boardWidth * 2) + "┘", DIM);
  for (let r = 0; r < boardHeight; r++) {
    c.put(top + 1 + r, left, "│", DIM);
    c.put(top + 1 + r, left + boardWidth * 2 + 1, "│", DIM);
    for (let col = 0; col < boardWidth; col++) {
      const cell = game.board[r]![col];
      if (cell) c.put(top + 1 + r, left + 1 + 2 * col, "██", PIECE_COLORS[cell]);
      else c.put(top + 1 + r, left + 1 + 2 * col, "· ", "#1f2740");
    }
  }

  c.put(bottom + 2, left, "SCORE", MUTED);
  c.number(bottom + 3, left, game.score, AMBER);
  c.put(bottom + 2, left + 22, "LINES", MUTED);
  c.number(bottom + 3, left + 22, game.linesCleared, PIECE_COLORS.I);
  c.put(bottom + 2, left + 40, "LEVEL", MUTED);
  c.number(bottom + 3, left + 40, game.level, FG);

  c.put(4, right, `Laya · ${stats.engine ?? "FP16"}`, PIECE_COLORS.I);
  c.put(5, right, `${stats.hardware ?? "Local"} · Local`, MUTED);
  c.put(7, right, "ACTIVE", MUTED);
  c.put(7, right + 10, game.active, PIECE_COLORS[game.active]);
  c.put(8, right, "NEXT", MUTED);
  c.put(8, right + 10, game.queue.slice(0, 5).join(" "), FG);

  c.put(10, right, "PROPOSED", MUTED);
  c.put(10, right + 11, decision.proposed ? `${decision.proposed.kind}-${decision.proposed.rotation}-c${decision.proposed.col}` : "—", MUTED);
  c.put(11, right, "EXECUTING", MUTED);
  c.put(11, right + 11, decision.executed ? `${decision.executed.kind}-${decision.executed.rotation}-c${decision.executed.col}` : "—", PIECE_COLORS.I);
  if (decision.intervened) c.put(11, right + 30, "SHIELD", AMBER);

  c.put(13, right, "TOPPING-OUT RISK", MUTED);
  const risk = decision.risk ?? 0;
  c.bar(14, right, risk, 26, risk < 0.5 ? AMBER : RED);
  c.put(14, right + 29, risk.toFixed(2), risk < 0.5 ? AMBER : RED);
  c.put(16, right, "CLEARS A LINE", MUTED);
  c.bar(17, right, decision.clears_signal ?? 0, 26, PIECE_COLORS.S);
  c.put(17, right + 29, (decision.clears_signal ?? 0).toFixed(2), PIECE_COLORS.S);

  c.put(19, right, "INFERENCE", MUTED);
  c.put(19, right + 18, `${fixed(decision.inference_ms ?? 0, 1, 5)} ms`, FG);
  c.put(20, right, "DECISIONS", MUTED);
  c.put(20, right + 18, `${fixed(stats.pieces_per_second ?? 0, 1, 5)} /s`, FG);
  c.put(21, right, "OUTPUT TOKENS", MUTED);
  c.put(21, right + 18, String(decision.output_tokens ?? 0), FG);
  c.put(22, right, "NETWORK", MUTED);
  c.put(22, right + 18, "OFFLINE", PIECE_COLORS.S);
  c.put(23, right, "ENGINE", MUTED);
  c.put(23, right + 18, stats.engine ?? "FP16", MUTED);
  c.put(25, right, stats.guarded ?? true ? "Laya + safety-margin shield" : "Laya · shield OFF", MUTED);
  c.put(26, right, `Shield interventions  ${String(stats.interventions ?? 0).padStart(4, "0")}`, AMBER);
  c.put(height - 3, left, "─".repeat(width - 6), DIM);
  c.put(height - 2, left, "SPACE pause   R reset   Q quit", MUTED);
  const elapsed = Math.trunc(stats.elapsed ?? 0);
  const clock = `${String(Math.trunc(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;
  c.put(height - 2, right, `ESTIMATES BY LAYA            ${clock}`, MUTED);
  return c;
}
