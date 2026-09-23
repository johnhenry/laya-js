/**
 * Dependency-light language/script detection used to route between Laya checkpoints.
 * A line-for-line port of laya-mlx `laya_mlx/lang.py` (derived from Laya, Apache-2.0).
 *
 * Script detection is exact. The Latin-script language guess is a stopword/diacritic
 * heuristic and is explicitly best-effort.
 */
import { pyRound } from "@johnhenry/pyjson";

export type ScriptName =
  | "latin" | "greek" | "cyrillic" | "armenian" | "hebrew" | "arabic" | "devanagari"
  | "bengali" | "gurmukhi" | "gujarati" | "oriya" | "tamil" | "telugu" | "kannada"
  | "malayalam" | "sinhala" | "thai" | "lao" | "tibetan" | "myanmar" | "georgian"
  | "ethiopic" | "khmer" | "hangul" | "kana" | "han";

/** Unicode blocks the English (ModernBERT-large) checkpoint cannot read. Order matters. */
export const SCRIPT_RANGES: ReadonlyArray<readonly [ScriptName, ReadonlyArray<readonly [number, number]>]> = [
  ["greek", [[0x0370, 0x03ff], [0x1f00, 0x1fff]]],
  ["cyrillic", [[0x0400, 0x052f], [0x2de0, 0x2dff], [0xa640, 0xa69f]]],
  ["armenian", [[0x0530, 0x058f]]],
  ["hebrew", [[0x0590, 0x05ff]]],
  ["arabic", [[0x0600, 0x06ff], [0x0750, 0x077f], [0x08a0, 0x08ff], [0xfb50, 0xfdff], [0xfe70, 0xfeff]]],
  ["devanagari", [[0x0900, 0x097f], [0xa8e0, 0xa8ff]]],
  ["bengali", [[0x0980, 0x09ff]]],
  ["gurmukhi", [[0x0a00, 0x0a7f]]],
  ["gujarati", [[0x0a80, 0x0aff]]],
  ["oriya", [[0x0b00, 0x0b7f]]],
  ["tamil", [[0x0b80, 0x0bff]]],
  ["telugu", [[0x0c00, 0x0c7f]]],
  ["kannada", [[0x0c80, 0x0cff]]],
  ["malayalam", [[0x0d00, 0x0d7f]]],
  ["sinhala", [[0x0d80, 0x0dff]]],
  ["thai", [[0x0e00, 0x0e7f]]],
  ["lao", [[0x0e80, 0x0eff]]],
  ["tibetan", [[0x0f00, 0x0fff]]],
  ["myanmar", [[0x1000, 0x109f]]],
  ["georgian", [[0x10a0, 0x10ff]]],
  ["ethiopic", [[0x1200, 0x137f]]],
  ["khmer", [[0x1780, 0x17ff]]],
  ["hangul", [[0x1100, 0x11ff], [0x3130, 0x318f], [0xac00, 0xd7af]]],
  ["kana", [[0x3040, 0x309f], [0x30a0, 0x30ff], [0x31f0, 0x31ff]]],
  ["han", [[0x3400, 0x4dbf], [0x4e00, 0x9fff], [0xf900, 0xfaff]]],
];

const words = (s: string) => new Set(s.split(" "));

/** Function words per Latin-script language (insertion order is the tie-break order). */
export const STOPWORDS: Readonly<Record<string, ReadonlySet<string>>> = {
  en: words(
    "the and is are was were to of in for with that this it you have has not but on at be as from will can would there their what which please we i",
  ),
  fr: words("le la les des une est pour dans que qui avec sur pas plus nous vous être cette mais sont ont aux ce"),
  de: words("der die das und ist ein eine den dem nicht mit für auf von zu sich auch werden wurde haben sind oder aber"),
  es: words("el los las que por con para una es se del como pero son está este esta todo más muy hay sus"),
  pt: words("os as que em um uma para com não é se do da dos das mas são está este esta muito pelo pela"),
  it: words("il lo gli che di per con non è si del della sono questo questa anche come più sono nella alla"),
  nl: words("het een van is op te dat niet met voor zijn aan door maar ook worden deze naar wordt"),
  // Romanian words its Romance neighbours do not share (see lang.py)
  ro: words(
    "și să este sunt care pentru din dar după până fără ale lui în fost acum vreau trebuie foarte acest această acesta aceasta mi ți vă nu",
  ),
};

/** Letters that ordinary English does not use (matched after lowercasing). */
export const NON_EN_DIACRITICS: ReadonlySet<string> = new Set(
  "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ" +
    "ăâîșțşţ" +
    "ąćęłńśźż" +
    "čďěňřšťůž" +
    "őű" +
    "ğı" +
    "āēģīķļņūž" +
    "đ",
);

/** A diacritic rate at or above this is taken as evidence the text is not English. */
export const NON_EN_DIACRITIC_RATE = 0.02;

/**
 * Python `re.compile(r"[^\W\d_]+", re.UNICODE)`: word characters (str.isalnum) minus
 * decimal digits and underscore, i.e. letters plus non-decimal numerics.
 */
const WORD = /[\p{L}\p{Nl}\p{No}]+/gu;
/** Python `str.isalpha()`: general category L*. */
const ALPHA = /^\p{L}$/u;

export type State = string | null | undefined | boolean | number | readonly unknown[] | { [k: string]: unknown };

function iterText(state: unknown, depth = 0): string[] {
  if (depth > 6 || state === null || state === undefined) return [];
  if (typeof state === "string") return [state];
  if (Array.isArray(state)) return state.flatMap((v) => iterText(v, depth + 1));
  if (typeof state === "object" && !(state instanceof Map)) {
    return Object.values(state).flatMap((v) => iterText(v, depth + 1));
  }
  if (state instanceof Map) return [...state.values()].flatMap((v) => iterText(v, depth + 1));
  return [];
}

/** Code-point prefix (Python `s[:n]`). */
function cpSlice(s: string, n: number): string {
  let i = 0;
  let count = 0;
  while (i < s.length && count < n) {
    const c = s.charCodeAt(i);
    i += c >= 0xd800 && c <= 0xdbff && i + 1 < s.length ? 2 : 1;
    count++;
  }
  return s.slice(0, i);
}

/** Flatten a state into detection text (string leaves only; keys are ignored). */
export function stateText(state: unknown, maxChars = 4000): string {
  return cpSlice(iterText(state).join(" "), maxChars);
}

function scriptOf(cp: number): ScriptName | null {
  if (cp < 0x0250 || (cp >= 0x1e00 && cp <= 0x1eff)) return "latin";
  for (const [name, ranges] of SCRIPT_RANGES) {
    if (ranges.some(([lo, hi]) => lo <= cp && cp <= hi)) return name;
  }
  return null;
}

/** Dominant script of `text`: 'latin', 'han', 'devanagari', ... or 'unknown' if there are no letters. */
export function detectScript(text: string): ScriptName | "unknown" {
  const counts = new Map<string, number>();
  let latin = 0;
  for (const ch of text) {
    if (!ALPHA.test(ch)) continue;
    const s = scriptOf(ch.codePointAt(0)!);
    if (s === "latin") latin++;
    else if (s) counts.set(s, (counts.get(s) ?? 0) + 1);
  }
  counts.set("latin", latin);
  let best: string | null = null;
  let bestN = -1;
  let total = 0;
  for (const [k, v] of counts) {
    total += v;
    if (v > bestN) [best, bestN] = [k, v];
  }
  return total === 0 ? "unknown" : (best as ScriptName);
}

/** Fraction of alphabetic characters belonging to each detected script. */
export function scriptProfile(text: string): Record<string, number> {
  const counts = new Map<string, number>([["latin", 0]]);
  for (const ch of text) {
    if (!ALPHA.test(ch)) continue;
    const s = scriptOf(ch.codePointAt(0)!);
    if (s) counts.set(s, (counts.get(s) ?? 0) + 1);
  }
  let total = 0;
  for (const v of counts.values()) total += v;
  const out: Record<string, number> = {};
  if (!total) return out;
  for (const [k, v] of counts) if (v) out[k] = v / total;
  return out;
}

export interface LatinProfile {
  language: string | null;
  english_hits: number;
  diacritic_rate: number;
  looks_non_english: boolean;
}

/** Evidence behind the Latin-script language guess (see lang.py `latin_profile`). */
export function latinProfile(text: string): LatinProfile {
  const ws = (text.match(WORD) ?? []).map((w) => w.toLowerCase());
  const lowered = [...text.toLowerCase()];
  let diac = 0;
  for (const ch of lowered) if (NON_EN_DIACRITICS.has(ch)) diac++;
  const diacRate = diac / Math.max(1, lowered.length);
  const nonEnglish = diacRate >= NON_EN_DIACRITIC_RATE;
  if (ws.length < 4) {
    return { language: null, english_hits: 0, diacritic_rate: diacRate, looks_non_english: nonEnglish };
  }
  const scores: Array<[string, number]> = Object.entries(STOPWORDS).map(([lg, sw]) => [
    lg,
    ws.reduce((n, w) => n + (sw.has(w) ? 1 : 0), 0),
  ]);
  const en = scores.find(([lg]) => lg === "en")?.[1] ?? 0;
  let bestLg: string | null = null;
  let best = 0;
  let first = true;
  for (const [lg, s] of scores) {
    if (lg === "en") continue;
    if (first || s > best) [bestLg, best, first] = [lg, s, false];
  }
  if (best === 0) bestLg = null; // a 0-0 tie is no evidence for a particular language
  let lang: string | null = null;
  if (bestLg && best >= Math.max(2, en + 2)) lang = bestLg;
  else if (bestLg && nonEnglish && best >= Math.max(2, en)) lang = bestLg;
  else if (en && !nonEnglish) lang = "en";
  return { language: lang, english_hits: en, diacritic_rate: diacRate, looks_non_english: nonEnglish };
}

/** Best-effort language code for Latin-script text, or null when undecided. */
export function guessLatinLanguage(text: string): string | null {
  return latinProfile(text).language;
}

export interface Analysis {
  script: ScriptName | "unknown";
  script_profile: Record<string, number>;
  language: string | null;
  is_english: boolean;
  language_undecided: boolean;
  diacritic_rate: number;
  non_latin_fraction: number;
}

/** Full detection result for a state (same keys and values as lang.py `analyse`). */
export function analyse(state: unknown): Analysis {
  const text = stateText(state);
  const prof = scriptProfile(text);
  const script = detectScript(text);
  const nonLatin = Object.keys(prof).length ? pyRound(1.0 - (prof.latin ?? 0.0), 4) : 0.0;
  if (script === "unknown") {
    return {
      script: "unknown", script_profile: prof, language: null, is_english: true,
      language_undecided: true, diacritic_rate: 0.0, non_latin_fraction: 0.0,
    };
  }
  if (script !== "latin") {
    return {
      script, script_profile: prof, language: null, is_english: false,
      language_undecided: true, diacritic_rate: 0.0, non_latin_fraction: nonLatin,
    };
  }
  const lat = latinProfile(text);
  const undecided = lat.language === null;
  const english = lat.language === "en" || (undecided && !lat.looks_non_english);
  return {
    script: "latin", script_profile: prof, language: lat.language, is_english: english,
    language_undecided: undecided, diacritic_rate: pyRound(lat.diacritic_rate, 4), non_latin_fraction: nonLatin,
  };
}

/** True when the English checkpoint can be expected to read this state. */
export function isEnglish(state: unknown): boolean {
  return analyse(state).is_english;
}

// snake_case aliases matching the Python module
export {
  stateText as state_text,
  detectScript as detect_script,
  scriptProfile as script_profile,
  latinProfile as latin_profile,
  guessLatinLanguage as guess_latin_language,
  isEnglish as is_english,
};
