// Derived from Laya (Apache-2.0); see NOTICE. Ported from laya-mlx email.py.
/** Cleaning and structuring email inputs (quoted history, signatures, disclaimers). */

const QUOTE_HEADERS = [
  /^\s*On .{0,300}wrote:\s*$/iu,
  /^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}/iu,
  /^\s*_{8,}\s*$/u,
  /^\s*From:\s.+$/iu,
];
// Python `\w` (str patterns) is Unicode: letters, digits, underscore.
const SIGNATURE_MARKERS = [
  /^\s*--\s*$/u,
  /^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\p{L}\p{N}_ ,!.]*$/iu,
  /^\s*sent from my (iphone|android|mobile|ipad)/iu,
];
const DISCLAIMER =
  /(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|if you (have )?received this (e-?mail|message) in error)/iu;
const SENTENCE = /(?<=[.!?])\s+/u;

/** Python `len()` / slicing count code points, not UTF-16 units. */
const cpLength = (s: string) => [...s].length;

/**
 * Drop boilerplate disclaimer text from one paragraph: the whole paragraph only when every
 * sentence in it is boilerplate, otherwise just the boilerplate sentences.
 */
function stripDisclaimer(paragraph: string): string {
  if (!DISCLAIMER.test(paragraph)) return paragraph;
  const parts = paragraph.split(SENTENCE).map((p) => p.trim()).filter(Boolean);
  return parts.filter((p) => !DISCLAIMER.test(p)).join(" ");
}

/** Remove quoted email history, signatures and disclaimers to keep input focused (`clean_email_body`). */
export function cleanEmailBody(body: string | null | undefined, maxChars = 3000): string {
  const text = (body || "").replaceAll("\r\n", "\n").replaceAll("\r", "\n").replaceAll("\\n", "\n");
  let lines: string[] = [];
  for (const line of text.split("\n")) {
    if (QUOTE_HEADERS.some((p) => p.test(line)) && lines.length) break;
    if (line.trimStart().startsWith(">")) continue;
    lines.push(line.trimEnd());
  }
  let cut = lines.length;
  for (let i = Math.max(1, Math.min(Math.trunc(lines.length * 0.6), lines.length - 8)); i < lines.length; i++) {
    if (cpLength(lines[i]!.trim()) <= 40 && SIGNATURE_MARKERS.some((p) => p.test(lines[i]!))) {
      cut = i;
      break;
    }
  }
  lines = lines.slice(0, cut);
  const paragraphs = lines.join("\n").split(/\n\s*\n/u).map(stripDisclaimer);
  const joined = paragraphs.map((p) => p.trim()).filter(Boolean).join("\n\n").replace(/[ \t]+/g, " ");
  return [...joined].slice(0, maxChars).join("");
}

export interface EmailStateOptions {
  sender?: string | null;
  /** Run `cleanEmailBody` on the body (default true). */
  clean?: boolean;
  /** Extra state fields; null/undefined values are dropped. */
  extra?: Record<string, unknown>;
}

/** A clean state object for email classification (`email_state`). */
export function emailState(subject: string | null | undefined, body: string | null | undefined, opts: EmailStateOptions = {}): Record<string, unknown> {
  const state: Record<string, unknown> = {
    subject: (subject || "").trim(),
    body: opts.clean ?? true ? cleanEmailBody(body) : body || "",
  };
  if (opts.sender) state.from = opts.sender;
  for (const [k, v] of Object.entries(opts.extra ?? {})) if (v !== null && v !== undefined) state[k] = v;
  return state;
}
