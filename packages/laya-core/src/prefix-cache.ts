/**
 * Bounded tokenized-prefix reuse (port of laya-mlx `prepared.py`). Encoder states and
 * predictions are never cached.
 */
import type { Prepared } from "./agent.ts";
import { toInternal, pyRepr } from "./agent.ts";
import { buildPrefix, renderOptions, serializeState } from "./common.ts";
import { QTYPES, type AgentConfig, type InternalQuestion, type LayaTokenizer, type PreparedItem } from "./types.ts";

export interface PreparedQuestion {
  readonly ids: readonly number[];
  readonly markers: readonly number[];
}

let nextTokenizerId = 1;
const tokenizerIds = new WeakMap<object, number>();
const tokenizerId = (tok: object) => {
  let id = tokenizerIds.get(tok);
  if (id === undefined) tokenizerIds.set(tok, (id = nextTokenizerId++));
  return id;
};

const isDict = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/** LRU cache of question prefixes keyed by tokenizer, head budget, type, instructions and options. */
export class PrefixCache {
  capacity: number;
  readonly entries = new Map<string, PreparedQuestion>();

  constructor(capacity = 128) {
    this.capacity = capacity;
  }

  /** Same result as the uncached `prepare`, tokenizing the state once. */
  prepare(tok: LayaTokenizer, state: unknown, questions: unknown, cfg: AgentConfig = {}): Prepared {
    if (!isDict(questions)) throw new Error("questions must be a dictionary keyed by question id");
    const items: PreparedItem[] = [];
    const internal: InternalQuestion[] = [];
    const entries = Object.entries(questions);
    if (entries.length === 0) return { items, internal };
    const maxLen = cfg.max_len ?? 512;
    const headLen = cfg.head_max_len ?? 192;
    const stateIds = tok.encode(serializeState(state).split(tok.maskToken).join(" "));
    for (const [qid, definition] of entries) {
      const q = toInternal(definition);
      const options = renderOptions(q);
      const key = JSON.stringify([
        tokenizerId(tok), tok.clsTokenId, tok.sepTokenId, tok.maskTokenId, tok.maskToken,
        headLen, q.t, q.ins, options,
      ]);
      let prefix = this.entries.get(key);
      if (prefix === undefined) {
        const built = buildPrefix(tok, q, headLen);
        prefix = { ids: Object.freeze(built.ids), markers: Object.freeze(built.markers) };
        this.entries.set(key, prefix);
        if (this.entries.size > this.capacity) this.entries.delete(this.entries.keys().next().value!);
      }
      // move to end (most recently used)
      this.entries.delete(key);
      this.entries.set(key, prefix);
      const room = Math.max(0, maxLen - prefix.ids.length - 1);
      const ids = [...prefix.ids, ...stateIds.slice(0, room), tok.sepTokenId].slice(0, maxLen);
      const markers = prefix.markers.filter((m) => m < maxLen);
      if (markers.length !== options.length) {
        throw new Error(`Question ${pyRepr(qid)} has too many options for the token budget`);
      }
      items.push({ ids, markers, qtype: QTYPES[q.t] });
      internal.push({ ...q, id: qid });
    }
    return { items, internal };
  }
}
