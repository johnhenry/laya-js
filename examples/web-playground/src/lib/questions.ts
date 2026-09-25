/**
 * The Questions builder's draft <-> Laya-questions-object conversions.
 * Pure (no DOM), so it's directly testable and shared between the builder
 * UI and its "Edit as JSON" escape hatch in main.ts.
 */
export type QType = "choice" | "score" | "noul";

export interface QDraft {
  id: string;
  type: QType;
  instructions: string;
  /** choice: label + description; score: label only (the level text). */
  rows: { label: string; desc: string }[];
  /** noul: optional descriptions for true / false. */
  yes: string;
  no: string;
}

export const str = (v: unknown): string => (v === null || v === undefined ? "" : typeof v === "string" ? v : JSON.stringify(v));

export function toDrafts(questions: Record<string, unknown>): QDraft[] {
  return Object.entries(questions).map(([id, raw]) => {
    const q = raw as { type: QType; instructions?: unknown; criteria?: unknown };
    const d: QDraft = { id, type: q.type, instructions: str(q.instructions), rows: [], yes: "", no: "" };
    if (q.type === "choice") {
      d.rows = Array.isArray(q.criteria)
        ? q.criteria.map((l) => ({ label: str(l), desc: "" }))
        : Object.entries((q.criteria ?? {}) as Record<string, unknown>).map(([label, desc]) => ({ label, desc: str(desc) }));
    } else if (q.type === "score") {
      d.rows = ((q.criteria ?? []) as unknown[]).map((l) => ({ label: str(l), desc: "" }));
    } else {
      const c = (q.criteria ?? {}) as { true?: unknown; false?: unknown };
      d.yes = str(c.true);
      d.no = str(c.false);
    }
    return d;
  });
}

export function fromDrafts(list: QDraft[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const seen = new Set<string>();
  for (const d of list) {
    const id = d.id.trim();
    if (!id) throw new Error("Every question needs an id.");
    if (seen.has(id)) throw new Error(`Duplicate question id "${id}".`);
    seen.add(id);
    if (!d.instructions.trim()) throw new Error(`Question "${id}" needs instructions.`);
    if (d.type === "choice") {
      const rows = d.rows.filter((r) => r.label.trim());
      if (rows.length < 2) throw new Error(`Choice "${id}" needs at least two options.`);
      out[id] = {
        type: "choice",
        instructions: d.instructions,
        criteria: rows.every((r) => !r.desc.trim())
          ? rows.map((r) => r.label.trim())
          : Object.fromEntries(rows.map((r) => [r.label.trim(), r.desc.trim() || null])),
      };
    } else if (d.type === "score") {
      const rows = d.rows.filter((r) => r.label.trim());
      if (rows.length < 2) throw new Error(`Score "${id}" needs at least two levels.`);
      out[id] = { type: "score", instructions: d.instructions, criteria: rows.map((r) => r.label.trim()) };
    } else {
      const q: Record<string, unknown> = { type: "noul", instructions: d.instructions };
      if (d.yes.trim() || d.no.trim()) q.criteria = { ...(d.yes.trim() ? { true: d.yes.trim() } : {}), ...(d.no.trim() ? { false: d.no.trim() } : {}) };
      out[id] = q;
    }
  }
  if (!Object.keys(out).length) throw new Error("Add at least one question.");
  return out;
}

export function newDraft(existing: QDraft[], type: QType): QDraft {
  let n = existing.length;
  const base = type === "noul" ? "yesno" : type;
  while (existing.some((d) => d.id === `${base}${n}`)) n++;
  return {
    id: `${base}${n}`,
    type,
    instructions: "",
    rows: type === "choice" ? [{ label: "", desc: "" }, { label: "", desc: "" }] : type === "score" ? [{ label: "low", desc: "" }, { label: "medium", desc: "" }, { label: "high", desc: "" }] : [],
    yes: "",
    no: "",
  };
}
