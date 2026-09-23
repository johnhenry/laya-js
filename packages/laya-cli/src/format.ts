/** Python `json.dumps(obj, ensure_ascii=False, indent=2)` output for predict results. */
import { dumps, encodeString, pyFloat, isPyFloat } from "@johnhenry/pyjson";

/** `json.dumps(value, ensure_ascii=False, indent=indent)` (scalars and key order via @johnhenry/pyjson). */
export function dumpsIndent(value: unknown, indent = 2, level = 0): string {
  const pad = " ".repeat(indent * (level + 1));
  const end = " ".repeat(indent * level);
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    return "[\n" + value.map((v) => pad + dumpsIndent(v, indent, level + 1)).join(",\n") + "\n" + end + "]";
  }
  if (value !== null && typeof value === "object" && !isPyFloat(value) && Object.getPrototypeOf(value) === Object.prototype) {
    const entries = Object.entries(value);
    if (!entries.length) return "{}";
    return "{\n" + entries.map(([k, v]) => pad + encodeString(k, false) + ": " + dumpsIndent(v, indent, level + 1)).join(",\n") + "\n" + end + "}";
  }
  return dumps(value, { ensureAscii: false });
}

const FLOAT_FIELDS = ["confidence", "score", "noul"] as const;

/**
 * Marks the fields Python holds as floats (`round(float, 4)`) so they print like
 * Python (`1.0`, not `1`): confidence, score, noul, act_probability, probabilities.
 */
export function pythonFloats<T extends { answers?: Record<string, any>; routing?: any }>(result: T): T {
  const answers: Record<string, unknown> = {};
  for (const [qid, a] of Object.entries(result.answers ?? {})) {
    const out: Record<string, unknown> = { ...a };
    for (const f of FLOAT_FIELDS) if (typeof a[f] === "number") out[f] = pyFloat(a[f]);
    if (a.action) out.action = { ...a.action, act_probability: pyFloat(a.action.act_probability) };
    if (a.probabilities) out.probabilities = Object.fromEntries(Object.entries(a.probabilities).map(([k, v]) => [k, pyFloat(v as number)]));
    answers[qid] = out;
  }
  const out: T = { ...result, answers };
  const det = result.routing?.detection;
  if (det) {
    // langdetect `analyse` floats: script_profile fractions, diacritic_rate, non_latin_fraction
    out.routing = {
      ...result.routing,
      detection: {
        ...det,
        script_profile: Object.fromEntries(Object.entries(det.script_profile ?? {}).map(([k, v]) => [k, pyFloat(v as number)])),
        diacritic_rate: pyFloat(det.diacritic_rate),
        non_latin_fraction: pyFloat(det.non_latin_fraction),
      },
    };
  }
  return out;
}
