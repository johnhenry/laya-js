/**
 * The result queue's data shape and pure derivations (priority sort value,
 * summaries, JSON/CSV export). No DOM -- main.ts owns rendering and the
 * actual download; this module is what's directly testable.
 */
import type { QType } from "./questions.ts";
import { str } from "./questions.ts";
import type { Json } from "./state-fields.ts";

export interface AnswerLike {
  type: QType;
  confidence: number;
  action: { act_probability: number };
  choice?: string;
  score?: number;
  legend?: Record<string, unknown>;
  noul?: number;
  probabilities?: Record<string, number>;
}

export interface RunOutcome {
  label: string;
  ms?: number;
  result?: { answers: Record<string, AnswerLike>; usage?: { input_tokens?: number } };
  error?: string;
}

export interface QueueEntry {
  id: string;
  timestamp: number;
  state: Json;
  questions: Record<string, unknown>;
  outcomes: RunOutcome[];
}

/** Normalizes a priority question's answer to 0..1 for sorting, or null if not applicable to this entry.
 * Uses `entry.outcomes[0]` (the primary backend's result) even in compare mode. */
export function computePriorityValue(entry: QueueEntry, priorityQid: string): number | null {
  if (!priorityQid) return null;
  const q = entry.questions[priorityQid] as { type?: QType; criteria?: unknown[] } | undefined;
  const a = entry.outcomes[0]?.result?.answers[priorityQid] as AnswerLike | undefined;
  if (!q || !a) return null;
  if (q.type === "noul") return a.noul ?? null;
  if (q.type === "score") {
    const levels = Array.isArray(q.criteria) ? q.criteria.length : Object.keys(a.probabilities ?? {}).length;
    return levels > 1 ? (a.score ?? 0) / (levels - 1) : null;
  }
  return null;
}

export function stateSummary(state: Json): string {
  const text = typeof state === "string" ? state : Object.values(state as Record<string, Json>)[0];
  return str(text).slice(0, 80) || "(empty)";
}

export function csvCell(v: unknown): string {
  const s = v === undefined || v === null ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** Columns are the union actually seen across queued runs -- not a fixed schema (mirrors gui-demo's CSV export). */
export function queueToCsv(queue: QueueEntry[], priorityQid: string): string {
  const stateKeys = new Set<string>();
  const questionKeys = new Set<string>();
  for (const e of queue) {
    if (e.state && typeof e.state === "object" && !Array.isArray(e.state)) Object.keys(e.state).forEach((k) => stateKeys.add(k));
    Object.keys(e.questions).forEach((k) => questionKeys.add(k));
  }
  const stateCols = [...stateKeys];
  const qCols = [...questionKeys];
  const header = ["id", "timestamp", "engine", ...stateCols, "priorityValue", "latencyMs", ...qCols.flatMap((n) => [n, `${n}.confidence`])];
  const rows = queue.flatMap((entry) =>
    entry.outcomes.map((o) => {
      const stateObj = entry.state && typeof entry.state === "object" && !Array.isArray(entry.state) ? (entry.state as Record<string, Json>) : {};
      const answers = (o.result?.answers ?? {}) as Record<string, AnswerLike>;
      const row = [
        entry.id,
        new Date(entry.timestamp).toISOString(),
        o.label,
        ...stateCols.map((k) => str(stateObj[k])),
        String(computePriorityValue(entry, priorityQid) ?? ""),
        String(o.ms ?? ""),
        ...qCols.flatMap((n) => {
          const a = answers[n];
          const val = a?.choice ?? (a?.score !== undefined ? String(a.score) : a?.noul !== undefined ? String(a.noul) : "");
          return [val, a?.confidence !== undefined ? String(a.confidence) : ""];
        }),
      ];
      return row.map(csvCell).join(",");
    }),
  );
  return [header.join(","), ...rows].join("\n");
}

export function queueToJson(queue: QueueEntry[]): string {
  return JSON.stringify(queue, null, 2);
}
