/**
 * Fixed-cell terminal composition, rendered as 24-bit ANSI escapes. Same
 * bare-bones, dependency-free approach as snake-terminal/flappy-terminal's
 * `ui.ts`, with a warm red/black-on-walnut palette instead of Snake's
 * mint-green or Flappy's sky-blue one. Checkers needs one thing those two
 * don't: a real light/dark checkerboard pattern, which needs a
 * *per-cell* background color -- the shared `Canvas` shape in the other two
 * examples only supports one background for the whole line, so this
 * `Canvas` is extended with a background grid alongside the foreground one.
 */
import type { Board, Player, Square } from "./core/board.ts";
import { FILES } from "./core/board.ts";
import type { Decision } from "./core/policy.ts";
import type { Hop } from "./core/moves.ts";
import type { Actor } from "./core/actors.ts";
import type { SessionStats } from "./core/session.ts";

export const BG = "#140a08";
export const FG = "#f3e6d8";
export const MUTED = "#8a7060";
export const DIM = "#2a1c16";
export const LIGHT_SQUARE = "#3a281f";
export const DARK_SQUARE = "#1c120d";
export const RED = "#e0483e";
export const IVORY = "#efe6da";
export const AMBER = "#e8b464";

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
  bgs: string[][];

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.chars = Array.from({ length: height }, () => Array<string>(width).fill(" "));
    this.styles = Array.from({ length: height }, () => Array<string>(width).fill(FG));
    this.bgs = Array.from({ length: height }, () => Array<string>(width).fill(BG));
  }

  put(row: number, column: number, text: string | number, color = FG, cellBg?: string): void {
    if (row < 0 || row >= this.height) return;
    [...String(text)].forEach((ch, offset) => {
      const x = column + offset;
      if (x >= 0 && x < this.width) {
        this.chars[row]![x] = ch;
        this.styles[row]![x] = color;
        if (cellBg) this.bgs[row]![x] = cellBg;
      }
    });
  }

  /** Fill a cell's background without changing its character/foreground. */
  fillBg(row: number, column: number, width: number, color: string): void {
    if (row < 0 || row >= this.height) return;
    for (let x = column; x < column + width && x < this.width; x++) if (x >= 0) this.bgs[row]![x] = color;
  }

  bar(row: number, column: number, value: number, length = 20, color = RED): void {
    const count = pyRound(Math.max(0, Math.min(1, value)) * length);
    this.put(row, column, "━".repeat(length), DIM);
    this.put(row, column, "━".repeat(count), color);
  }

  number(row: number, column: number, value: number, color = RED): void {
    const digits = String(value).padStart(2, "0");
    [...digits].forEach((digit, index) => DIGITS[digit]!.forEach((glyphs, line) => this.put(row + line, column + index * 4, glyphs, color)));
  }

  /** ANSI truecolor (or plain text with `color = false`, e.g. NO_COLOR). */
  ansi(color = true): string {
    const lines: string[] = [];
    for (let r = 0; r < this.height; r++) {
      let line = "";
      let currentFg = "";
      let currentBg = "";
      for (let x = 0; x < this.width; x++) {
        const style = this.styles[r]![x]!;
        const cellBg = this.bgs[r]![x]!;
        if (color && style !== currentFg) {
          line += fg(style);
          currentFg = style;
        }
        if (color && cellBg !== currentBg) {
          line += bg(cellBg);
          currentBg = cellBg;
        }
        line += this.chars[r]![x];
      }
      lines.push(color ? line + "\x1b[0m" : line);
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
  return [104, 33];
}

export interface LastTurn {
  hop: Hop;
  player: Player;
  actor: Actor["kind"];
  decision?: Decision;
}

/** The 8x8 board plus the side panel, showing the current position with the last turn's outcome (if any). */
export function compose(board: Board, toMove: Player, round: number, last: LastTurn | undefined, stats: Partial<SessionStats> & { engine?: string }): Canvas {
  const [width, height] = layoutSize();
  const c = new Canvas(width, height);
  const left = 3;
  const right = 60;
  const top = 6;
  const cell = 6; // characters per board column
  const bottom = top + 8 * 1; // one text row per board rank

  c.put(1, left, "LAYA  /  LOCAL INTELLIGENCE", MUTED);
  const state = stats.paused ? "PAUSED" : "LIVE";
  c.put(1, width - state.length - 3, state, RED);
  c.put(2, left, "─".repeat(width - 6), DIM);
  c.put(4, left, "C H E C K E R S", FG);
  c.put(4, left + 20, `ROUND ${String(round).padStart(2, "0")}`, MUTED);

  for (let rank = 8; rank >= 1; rank--) {
    const row = top + (8 - rank);
    c.put(row, left, String(rank), MUTED);
    for (let fi = 0; fi < 8; fi++) {
      const col = left + 2 + fi * cell;
      const dark = (fi + rank) % 2 === 1;
      const squareColor = dark ? DARK_SQUARE : LIGHT_SQUARE;
      c.fillBg(row, col, cell, squareColor);
      const sq = `${FILES[fi]}${rank}` as Square;
      const piece = board[sq];
      if (piece) {
        const glyph = piece.kind === "king" ? "◆◆" : "██";
        const pieceColor = piece.player === "red" ? RED : IVORY;
        c.put(row, col + 2, glyph, pieceColor, squareColor);
      }
    }
  }
  c.put(top + 8, left + 2, FILES.map((f) => f.toUpperCase()).join(" ".repeat(cell - 1)), MUTED);

  c.put(bottom + 3, left, "TO MOVE", MUTED);
  c.put(bottom + 4, left, toMove.toUpperCase(), toMove === "red" ? RED : IVORY);

  c.put(4, right, `Laya · ${stats.engine ?? "FP16"}`, RED);
  c.put(5, right, `${stats.hardware ?? "Local"} · Local`, MUTED);
  c.put(7, right, "LAST MOVE", FG);
  if (last) {
    const notation = last.hop.kind === "capture" ? `${last.hop.from}x${last.hop.to}` : `${last.hop.from}-${last.hop.to}`;
    c.put(8, right, `${last.player} (${last.actor})`, last.player === "red" ? RED : IVORY);
    c.put(9, right, notation, FG);
    if (last.decision?.intervened) c.put(9, right + notation.length + 2, "SHIELD", AMBER);
  } else {
    c.put(8, right, "—", MUTED);
  }
  if (last?.decision) {
    c.put(11, right, "MATERIAL AT RISK", MUTED);
    const risk = last.decision.material_at_risk;
    c.bar(12, right, risk, 24, risk < 0.5 ? AMBER : RED);
    c.put(12, right + 27, risk.toFixed(2), risk < 0.5 ? AMBER : RED);
    c.put(14, right, "INFERENCE", MUTED);
    c.put(14, right + 16, `${fixed(last.decision.inference_ms, 1, 5)} ms`, FG);
    c.put(15, right, "INPUT TOKENS", MUTED);
    c.put(15, right + 16, String(last.decision.input_tokens), FG);
  }
  c.put(17, right, "DECISIONS", MUTED);
  c.put(17, right + 16, `${fixed(stats.turns_per_second ?? 0, 1, 5)} /s`, FG);
  c.put(18, right, "TURNS", MUTED);
  c.put(18, right + 16, String(stats.turns ?? 0), FG);
  c.put(20, right, "Laya + mandatory-capture shield", MUTED);
  c.put(21, right, `Shield interventions  ${String(stats.interventions ?? 0).padStart(4, "0")}`, AMBER);
  c.put(23, right, `Wins  red ${stats.red_wins ?? 0}  ·  black ${stats.black_wins ?? 0}`, MUTED);
  c.put(height - 3, left, "─".repeat(width - 6), DIM);
  c.put(height - 2, left, "Ctrl-C quits", MUTED);
  return c;
}
