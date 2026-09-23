/**
 * Tokenizer adapter over `@huggingface/tokenizers` (browser-safe; no fs).
 * Mirrors laya-mlx `laya_mlx/tokenizer.py`: add_special_tokens=False, no padding or
 * truncation, special tokens resolved from tokenizer_config.json.
 *
 * Tokenizers.js diverges from the Rust `tokenizers` crate in a few pre-tokenizers;
 * {@link patchTokenizer} fixes the ones that affect Laya checkpoints (see README).
 */
import { Tokenizer } from "@huggingface/tokenizers";
import type { LayaTokenizer } from "./types.ts";

type Cfg = Record<string, any>;

const SPECIALS = ["cls_token", "sep_token", "pad_token", "mask_token"] as const;

/** Rust Metaspace `split: true` (the default): split MergedWithNext on the replacement char. */
function splitMergedWithNext(s: string, rep: string): string[] {
  const out: string[] = [];
  let start = 0;
  let from = s.startsWith(rep) ? rep.length : 0;
  for (;;) {
    const j = s.indexOf(rep, from);
    if (j < 0) break;
    if (j > start) out.push(s.slice(start, j));
    start = j;
    from = j + rep.length;
  }
  if (start < s.length) out.push(s.slice(start));
  return out;
}

// Rust regex `\w` / `\s` are Unicode-aware; tokenizers.js uses ASCII `\w` for Whitespace.
const RUST_WORD = String.raw`\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}`;
const WHITESPACE_RE = new RegExp(`[${RUST_WORD}]+|[^${RUST_WORD}\\p{White_Space}]+`, "gu");

/**
 * Align tokenizers.js pre-tokenizers with the Rust crate, driven by the tokenizer.json
 * config (not class names, which bundlers may mangle). Returns the patch names applied.
 */
export function patchTokenizer(tok: Tokenizer, tokenizerJson: Cfg): string[] {
  const applied: string[] = [];
  const walk = (cfg: Cfg | null | undefined, inst: any) => {
    if (!cfg || !inst) return;
    if (cfg.type === "Sequence" && Array.isArray(cfg.pretokenizers) && Array.isArray(inst.tokenizers)) {
      cfg.pretokenizers.forEach((c: Cfg, i: number) => walk(c, inst.tokenizers[i]));
    } else if (cfg.type === "Metaspace" && cfg.split !== false) {
      const orig = inst.pre_tokenize_text.bind(inst);
      const rep: string = cfg.replacement ?? "▁";
      inst.pre_tokenize_text = (text: string, options?: unknown) =>
        (orig(text, options) as string[]).flatMap((s) => splitMergedWithNext(s, rep));
      applied.push("Metaspace.split");
    } else if (cfg.type === "Whitespace") {
      inst.pre_tokenize_text = (text: string) => text.match(WHITESPACE_RE) ?? [];
      applied.push("Whitespace.unicode");
    }
  };
  walk(tokenizerJson.pre_tokenizer, (tok as any).pre_tokenizer);
  // tokenizers.js has no WordLevel model: it falls back to a "Legacy" model that takes
  // unk_token from tokenizer_config.json, so unknown words map to `undefined`.
  const model = tokenizerJson.model;
  const inst = (tok as any).model;
  if (model?.type === "WordLevel" && inst && typeof model.unk_token === "string") {
    const unk = inst.tokens_to_ids?.get(model.unk_token);
    if (unk !== undefined && inst.unk_token_id !== unk) {
      inst.unk_token = model.unk_token;
      inst.unk_token_id = unk;
      applied.push("WordLevel.unk");
    }
  }
  return applied;
}

/** A {@link LayaTokenizer} backed by tokenizers.js. */
export class HFLayaTokenizer implements LayaTokenizer {
  readonly backend: Tokenizer;
  readonly clsToken: string;
  readonly clsTokenId: number;
  readonly sepToken: string;
  readonly sepTokenId: number;
  readonly padToken: string;
  readonly padTokenId: number;
  readonly maskToken: string;
  readonly maskTokenId: number;
  /** Compatibility patches applied to the backend (see {@link patchTokenizer}). */
  readonly patches: readonly string[];

  constructor(tokenizerJson: Cfg, tokenizerConfig: Cfg) {
    this.backend = new Tokenizer(tokenizerJson, tokenizerConfig);
    this.patches = patchTokenizer(this.backend, tokenizerJson);
    const special: Record<string, [string, number]> = {};
    for (const name of SPECIALS) {
      let value = tokenizerConfig[name];
      if (value && typeof value === "object") value = value.content;
      const id = typeof value === "string" ? this.tokenToId(value) : undefined;
      if (id === undefined) throw new Error(`Tokenizer is missing a valid ${name}`);
      special[name] = [value, id];
    }
    [this.clsToken, this.clsTokenId] = special.cls_token!;
    [this.sepToken, this.sepTokenId] = special.sep_token!;
    [this.padToken, this.padTokenId] = special.pad_token!;
    [this.maskToken, this.maskTokenId] = special.mask_token!;
  }

  /** Rust `token_to_id`: model vocab, then added tokens. */
  tokenToId(token: string): number | undefined {
    const id = this.backend.token_to_id(token);
    if (id !== undefined) return id;
    for (const [i, t] of this.backend.get_added_tokens_decoder()) if (t.content === token) return i;
    return undefined;
  }

  /** Token ids without special tokens (`add_special_tokens=False`). */
  encode(text: string): number[] {
    return this.backend.encode(text, { add_special_tokens: false }).ids;
  }

  /** Token ids with the post-processor's special tokens (`add_special_tokens=True`). */
  encodeWithSpecialTokens(text: string): number[] {
    return this.backend.encode(text, { add_special_tokens: true }).ids;
  }

  decode(ids: number[]): string {
    return this.backend.decode(ids);
  }
}

const asObject = (v: object | string): Cfg => (typeof v === "string" ? JSON.parse(v) : (v as Cfg));

/** Build a tokenizer from parsed (or raw JSON text) `tokenizer.json` and `tokenizer_config.json`. */
export function loadTokenizer(tokenizerJson: object | string, tokenizerConfig: object | string): HFLayaTokenizer {
  return new HFLayaTokenizer(asObject(tokenizerJson), asObject(tokenizerConfig));
}
