/**
 * The State builder's typed-field drafts <-> Laya `state` value conversions.
 * Pure (no DOM). A single default text field collapses to a plain string on
 * read, matching the zero-config free-text behavior every preset and the
 * README example expect; the inverse infers fields from any preset's shape.
 */
export type Json = null | boolean | number | string | Json[] | { [k: string]: Json };

export type SFType = "text" | "number" | "boolean";

export interface SFDraft {
  key: string;
  label: string;
  type: SFType;
  /** Raw draft value: the field's own text/number as typed, or "true"/"false" for boolean. */
  value: string;
}

export function newStateField(existing: SFDraft[], type: SFType): SFDraft {
  let n = existing.length;
  while (existing.some((f) => f.key === `field${n}`)) n++;
  return { key: `field${n}`, label: `Field ${n + 1}`, type, value: type === "boolean" ? "false" : "" };
}

/** Fields -> the actual state value sent to predict() (validation mirrors gui-demo's resolveStateValues). */
export function stateFieldsToJson(fields: SFDraft[]): Json {
  const out: Record<string, Json> = {};
  const seen = new Set<string>();
  for (const f of fields) {
    const key = f.key.trim();
    if (!key) throw new Error("Every state field needs a key.");
    if (seen.has(key)) throw new Error(`Duplicate state field key "${key}".`);
    seen.add(key);
    if (f.type === "text") {
      if (!f.value.trim()) throw new Error(`Field "${key}" needs a value.`);
      out[key] = f.value;
    } else if (f.type === "number") {
      const n = Number(f.value);
      if (!Number.isFinite(n)) throw new Error(`Field "${key}" needs a valid number.`);
      out[key] = n;
    } else {
      out[key] = f.value === "true";
    }
  }
  if (!Object.keys(out).length) throw new Error("Add at least one state field.");
  if (fields.length === 1 && fields[0]!.key === "text" && fields[0]!.type === "text") return out.text!;
  return out;
}

/** The inverse: an incoming state (from a preset, or "Use builder" from JSON mode) -> fields. */
export function jsonToStateFields(state: Json): SFDraft[] {
  if (typeof state === "string") return [{ key: "text", label: "Text", type: "text", value: state }];
  if (state && typeof state === "object" && !Array.isArray(state)) {
    const entries = Object.entries(state as Record<string, Json>);
    if (entries.length) {
      return entries.map(([key, v]) => {
        const type: SFType = typeof v === "number" ? "number" : typeof v === "boolean" ? "boolean" : "text";
        const value = type === "boolean" || type === "number" ? String(v) : typeof v === "string" ? v : JSON.stringify(v);
        return { key, label: key, type, value };
      });
    }
  }
  // Arrays, null, top-level numbers/booleans, or an empty object: not representable
  // as typed fields -- stash as JSON text; "Edit as JSON" is the real escape hatch.
  return [{ key: "text", label: "Text", type: "text", value: typeof state === "string" ? state : JSON.stringify(state) }];
}

/** Batch only makes sense with exactly one text field: "one value per line" is
 * ambiguous once there's more than one field to fill in per line (mirrors
 * gui-demo's exact constraint). */
export function batchEligible(fields: SFDraft[]): boolean {
  return fields.length === 1 && fields[0]!.type === "text";
}
