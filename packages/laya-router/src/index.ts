// Derived from Laya (Apache-2.0); see NOTICE. Ported from laya-mlx router.py.
/**
 * Route a request to the Laya checkpoint best suited to it.
 *
 *   english          convaiinnovations/laya                 421M  ModernBERT-large, 512 tokens
 *   multilingual     convaiinnovations/laya-multilingual    322M  mmBERT-base, 1024 tokens, 100+ langs
 *   typed-decisions  convaiinnovations/laya-typed-decisions 421M  ModernBERT-large, 1024 tokens,
 *                                                                 fine-tuned on four typed-decisions workflows
 *
 * The English checkpoint collapses off English (20-option MASSIVE intent: 0.100 on Hindi, 0.103
 * on Korean vs 0.050 random) while reporting high confidence, so script detection is the primary
 * routing signal. `typed-decisions` is never selected automatically unless you opt in with
 * `autoTaskDetection: true` or pass `task: "typed_decisions"`.
 */
import { analyse, type Analysis } from "@johnhenry/langdetect-lite";
import { pyRound } from "@johnhenry/pyjson";
import { pyRepr } from "@johnhenry/laya-core";

export type ModelName = "english" | "multilingual" | "typed-decisions";
/** A repo id / local path, or [repo, subfolder]. */
export type ModelSpec = string | readonly [string, string | null];

/** The hub repo bundles all three checkpoints; only the requested subfolder is downloaded. */
export const BUNDLE_REPO = "convaiinnovations/laya";
export const DEFAULT_MODELS: Readonly<Record<ModelName, ModelSpec>> = Object.freeze({
  english: [BUNDLE_REPO, null] as const,
  multilingual: [BUNDLE_REPO, "multilingual"] as const,
  "typed-decisions": [BUNDLE_REPO, "typed-decisions"] as const,
});
/** The same checkpoints in their own repos. */
export const STANDALONE_MODELS: Readonly<Record<ModelName, ModelSpec>> = Object.freeze({
  english: "convaiinnovations/laya",
  multilingual: "convaiinnovations/laya-multilingual",
  "typed-decisions": "convaiinnovations/laya-typed-decisions",
});
/**
 * The laya-mlx fp16 MLX exports of the same checkpoints (what laya-js is
 * validated against; smaller downloads). Not in Python's router.
 */
export const MLX_MODELS: Readonly<Record<ModelName, ModelSpec>> = Object.freeze({
  english: "aac6fef/laya-mlx",
  multilingual: "aac6fef/laya-multilingual-mlx",
  "typed-decisions": "aac6fef/laya-typed-decisions-mlx",
});

/** Aliases people are likely to type. */
export const ALIASES: Readonly<Record<string, ModelName>> = Object.freeze({
  en: "english",
  laya: "english",
  default: "english",
  multi: "multilingual",
  ml: "multilingual",
  "laya-multilingual": "multilingual",
  typed: "typed-decisions",
  typed_decisions: "typed-decisions",
  "laya-typed-decisions": "typed-decisions",
  decisions: "typed-decisions",
});

/** Question-id signatures of the four typed-decisions workflows (only used with autoTaskDetection). */
export const TYPED_DECISION_WORKFLOWS: Readonly<Record<string, ReadonlySet<string>>> = Object.freeze({
  agent_trace_observability: new Set(["action", "needs_review", "outcome", "risk", "urgency"]),
  customer_service: new Set(["action", "category", "churn_risk", "needs_human", "urgency"]),
  invoice_processing: new Set(["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"]),
  security_incidents: new Set(["credential_compromise", "disposition", "severity", "true_positive", "urgency"]),
});

const MODEL_NAMES = Object.keys(DEFAULT_MODELS).sort() as ModelName[];
const pyList = (xs: string[]) => `[${xs.map(pyRepr).join(", ")}]`;

/** "repo" or "repo/subfolder". */
export function repoStr(spec: ModelSpec): string {
  const [repo, sub] = splitSpec(spec);
  return sub ? `${repo}/${sub}` : repo;
}

/** A model spec as [repoOrPath, subfolder | null]. */
export function splitSpec(spec: ModelSpec): [string, string | null] {
  return typeof spec === "string" ? [spec, null] : [spec[0], spec[1] ?? null];
}

/** Canonical model name for a name or alias (case/whitespace-insensitive); throws on unknown names. */
export function normaliseName(name: unknown): ModelName {
  let key = String(name).trim().toLowerCase();
  key = ALIASES[key] ?? key;
  if (!(MODEL_NAMES as string[]).includes(key)) {
    throw new Error(`unknown model ${pyRepr(name)}; choose one of ${pyList(MODEL_NAMES)} (or an alias: ${pyList(Object.keys(ALIASES).sort())})`);
  }
  return key as ModelName;
}
export { normaliseName as normalizeName };

/**
 * Name of the typed-decisions workflow whose question ids these are, else null. Requires an
 * exact id-set match, so an unrelated schema that happens to contain `urgency` is never captured.
 */
export function matchTypedDecisionsWorkflow(questions: Record<string, unknown> | null | undefined): string | null {
  const ids = new Set(Object.keys(questions ?? {}));
  for (const [wf, sig] of Object.entries(TYPED_DECISION_WORKFLOWS)) {
    if (ids.size === sig.size && [...ids].every((i) => sig.has(i))) return wf;
  }
  return null;
}

/** The routing outcome: which model, why, and what was detected (JSON-serialisable). */
export interface RouteDecision {
  model: ModelName;
  repo: string;
  reason: string;
  detection: Analysis | null;
  workflow: string | null;
}

export interface RouteOptions {
  model?: string | null;
  task?: string | null;
  lang?: string | null;
}

/** What the router needs from an agent (a LayaAgent satisfies it). */
export interface RoutedAgent {
  predict(state: any, questions: any): Promise<any> | any;
  dispose?(): void;
}

export type Loader<A extends RoutedAgent> = (repoOrPath: string, opts: Record<string, unknown>) => Promise<A>;

export interface RouterOptions<A extends RoutedAgent = RoutedAgent> {
  /** Override checkpoints by (aliased) name. */
  models?: Partial<Record<string, ModelSpec>>;
  /** How many agents stay resident (LRU); default 1. */
  maxLoaded?: number;
  /** Model for text with no letters; default "english". */
  default?: string;
  /** Route exact typed-decisions workflow schemas to that checkpoint; default false. */
  autoTaskDetection?: boolean;
  /** Use STANDALONE_MODELS instead of the bundle repo. */
  standaloneRepos?: boolean;
  /** Use MLX_MODELS (the laya-mlx fp16 exports). Wins over standaloneRepos. */
  mlxRepos?: boolean;
  /** Passed to the loader for every model (backend, dtype, token, offline, ...). */
  loadOptions?: Record<string, unknown>;
  /** Builds an agent; default: `load` from @johnhenry/laya (imported on first use). */
  loader?: Loader<A>;
}

export interface RoutedResult {
  routing: RouteDecision;
  [key: string]: unknown;
}

const defaultLoader: Loader<RoutedAgent> = async (repo, opts) => {
  const { load } = await import("@johnhenry/laya");
  return load(repo, opts);
};

/**
 * Lazily loads Laya checkpoints and sends each request to the right one.
 *
 * Async semantics (the JS counterpart of Python's lock):
 * - Concurrent `load()`s of one model share a single in-flight promise: one build, one agent.
 *   A failed build rejects every waiter and is not cached (the next call retries).
 * - `maxLoaded` caps residency with least-recently-used eviction, but an agent with a
 *   `predict()` in flight through this router is never evicted or disposed: it stays until the
 *   call settles (so residency can briefly exceed `maxLoaded`), then eviction resumes.
 * - Agents the router built are disposed when evicted or unloaded (deferred while in use);
 *   attached agents belong to the caller and are only dropped. An agent obtained from
 *   `load()` and used directly is not protected: hold it only while it is resident, or use
 *   `predict()`.
 * - `unload()` does not cancel builds already in flight.
 */
export class Router<A extends RoutedAgent = RoutedAgent> {
  readonly models: Record<ModelName, ModelSpec>;
  maxLoaded: number;
  readonly default: ModelName;
  readonly autoTaskDetection: boolean;
  readonly loadOptions: Record<string, unknown>;
  private readonly loader: Loader<A>;
  private readonly agents = new Map<ModelName, A>();
  private order: ModelName[] = []; // least-recently-used first
  private readonly inflight = new Map<ModelName, Promise<A>>();
  private readonly inUse = new Map<A, number>();
  private readonly owned = new Set<A>();
  private readonly doomed = new Set<A>(); // owned, dropped while in use: dispose on release

  constructor(opts: RouterOptions<A> = {}) {
    this.models = { ...(opts.mlxRepos ? MLX_MODELS : opts.standaloneRepos ? STANDALONE_MODELS : DEFAULT_MODELS) };
    for (const [k, v] of Object.entries(opts.models ?? {})) if (v !== undefined) this.models[normaliseName(k)] = v;
    this.maxLoaded = Math.max(1, Math.trunc(opts.maxLoaded ?? 1));
    this.default = normaliseName(opts.default ?? "english");
    this.autoTaskDetection = !!opts.autoTaskDetection;
    this.loadOptions = { ...(opts.loadOptions ?? {}) };
    this.loader = opts.loader ?? (defaultLoader as Loader<A>);
  }

  // ------------------------------------------------------------------ loading
  /** The agent for `name`, building it on first use. Concurrent callers share one build. */
  load(name: string): Promise<A> {
    const key = normaliseName(name);
    const ready = this.agents.get(key);
    if (ready) {
      this.touch(key);
      return Promise.resolve(ready);
    }
    let p = this.inflight.get(key);
    if (!p) {
      const built = this.build(key);
      p = built;
      this.inflight.set(key, built);
      const clear = () => {
        if (this.inflight.get(key) === built) this.inflight.delete(key);
      };
      built.then(clear, clear);
    }
    return p;
  }

  private async build(key: ModelName): Promise<A> {
    const [repo, sub] = splitSpec(this.models[key]);
    const agent = await this.loader(repo, { ...this.loadOptions, ...(sub ? { subfolder: sub } : {}) });
    const existing = this.agents.get(key);
    if (existing) {
      // attach() registered one while we were building: keep that, drop ours
      agent.dispose?.();
      this.touch(key);
      return existing;
    }
    this.agents.set(key, agent);
    this.owned.add(agent);
    this.order.push(key);
    this.evict(key);
    return agent;
  }

  private touch(key: ModelName): void {
    this.order = this.order.filter((k) => k !== key);
    this.order.push(key);
  }

  private drop(agent: A): void {
    if (!this.owned.has(agent)) return;
    if ((this.inUse.get(agent) ?? 0) > 0) this.doomed.add(agent);
    else {
      this.owned.delete(agent);
      agent.dispose?.();
    }
  }

  private evict(protect?: ModelName): void {
    while (this.order.length > this.maxLoaded) {
      const victim = this.order.find((k) => k !== protect && !((this.inUse.get(this.agents.get(k)!) ?? 0) > 0));
      if (!victim) break;
      this.order = this.order.filter((k) => k !== victim);
      const agent = this.agents.get(victim);
      this.agents.delete(victim);
      if (agent) this.drop(agent);
    }
    for (const k of [...this.agents.keys()]) {
      if (!this.order.includes(k)) {
        const agent = this.agents.get(k)!;
        this.agents.delete(k);
        this.drop(agent);
      }
    }
  }

  /**
   * Register an already-built agent under `name` instead of loading a second copy. The router
   * never disposes attached agents. Raises `maxLoaded` to fit everything registered.
   */
  attach(name: string, agent: A): A {
    const key = normaliseName(name);
    const old = this.agents.get(key);
    this.agents.set(key, agent);
    if (old && old !== agent) this.drop(old);
    this.touch(key);
    this.maxLoaded = Math.max(this.maxLoaded, this.agents.size);
    return agent;
  }

  /** Build checkpoints up front (all by default) and raise `maxLoaded` to keep them resident. */
  async preload(names?: string[]): Promise<this> {
    const keys = (names ?? Object.keys(this.models)).map(normaliseName);
    this.maxLoaded = Math.max(this.maxLoaded, keys.length, this.agents.size);
    for (const k of keys) if (!this.agents.has(k)) await this.load(k);
    return this;
  }

  /** Free one model, or all of them (disposal of router-built agents waits for in-flight predicts). */
  unload(name?: string | null): void {
    const keys = name === undefined || name === null ? [...this.agents.keys()] : [normaliseName(name)];
    for (const k of keys) {
      const agent = this.agents.get(k);
      this.agents.delete(k);
      this.order = this.order.filter((o) => o !== k);
      if (agent) this.drop(agent);
    }
  }

  /** Resident model names, least-recently-used first. */
  get loaded(): ModelName[] {
    return [...this.order];
  }

  // ------------------------------------------------------------------ routing
  /**
   * Decide which checkpoint to use, without loading or running anything.
   * Precedence: explicit `model` > explicit `task` > detected workflow (opt-in) >
   * explicit `lang` > detected script/language > default.
   */
  route(state: unknown, questions?: Record<string, unknown> | null, opts: RouteOptions = {}): RouteDecision {
    const { model, task, lang } = opts;
    if (model !== undefined && model !== null) {
      const key = normaliseName(model);
      return { model: key, repo: repoStr(this.models[key]), reason: `explicit model=${pyRepr(model)}`, detection: null, workflow: null };
    }
    if (task !== undefined && task !== null) {
      const key = normaliseName(String(task).toLowerCase().replaceAll("-", "_") === "typed_decisions" ? "typed-decisions" : task);
      return { model: key, repo: repoStr(this.models[key]), reason: `explicit task=${pyRepr(task)}`, detection: null, workflow: null };
    }
    const workflow = matchTypedDecisionsWorkflow(questions ?? {});
    if (workflow && this.autoTaskDetection) {
      return {
        model: "typed-decisions",
        repo: repoStr(this.models["typed-decisions"]),
        reason: `question ids match the ${pyRepr(workflow)} typed-decisions workflow`,
        detection: null,
        workflow,
      };
    }
    if (lang !== undefined && lang !== null) {
      const key: ModelName = ["en", "eng", "english"].includes(String(lang).toLowerCase().split("-")[0]!) ? "english" : "multilingual";
      return { model: key, repo: repoStr(this.models[key]), reason: `explicit lang=${pyRepr(lang)}`, detection: null, workflow };
    }
    const det = analyse(state);
    let key: ModelName;
    let reason: string;
    const pct = (x: number) => String(pyRound(100 * x, 0)); // "%.0f" (round half to even)
    if (det.script === "unknown") {
      key = this.default;
      reason = `no letters detected in state; using default (${key})`;
    } else if (det.script !== "latin") {
      key = "multilingual";
      reason = `non-Latin script (${det.script}, ${pct(det.non_latin_fraction)}% of letters); the English checkpoint cannot read it`;
    } else if (!det.is_english) {
      key = "multilingual";
      reason = det.language
        ? `Latin script but language looks like ${pyRepr(det.language)}, not English`
        : `Latin script, language not identified but ${pct(det.diacritic_rate)}% non-English letters; not safe for the English checkpoint`;
    } else {
      key = "english";
      reason = "English Latin text";
    }
    return { model: key, repo: repoStr(this.models[key]), reason, detection: det, workflow };
  }

  // ------------------------------------------------------------------ running
  /** Route, then answer every question on the chosen checkpoint; the result gains `routing`. */
  async predict<R extends object = Record<string, unknown>>(state: unknown, questions: Record<string, unknown>, opts: RouteOptions = {}): Promise<R & RoutedResult> {
    const decision = this.route(state, questions, opts);
    const agent = await this.acquire(decision.model);
    try {
      const result = (await agent.predict(state, questions)) as R;
      return { ...result, routing: { ...decision } };
    } finally {
      this.release(agent);
    }
  }

  /** Alias of `predict`. */
  systemOne<R extends object = Record<string, unknown>>(state: unknown, questions: Record<string, unknown>, opts: RouteOptions = {}): Promise<R & RoutedResult> {
    return this.predict<R>(state, questions, opts);
  }

  /** Resident agent for `key`, marked in use (synchronously, so eviction cannot race it). */
  private async acquire(key: ModelName): Promise<A> {
    for (;;) {
      const a = this.agents.get(key);
      if (a) {
        this.touch(key);
        this.inUse.set(a, (this.inUse.get(a) ?? 0) + 1);
        return a;
      }
      await this.load(key); // may be evicted again before we resume: loop
    }
  }

  private release(agent: A): void {
    const n = (this.inUse.get(agent) ?? 1) - 1;
    if (n > 0) {
      this.inUse.set(agent, n);
      return;
    }
    this.inUse.delete(agent);
    if (this.doomed.delete(agent)) {
      this.owned.delete(agent);
      agent.dispose?.();
    }
    this.evict();
  }

  toString(): string {
    return `Router(loaded=${pyList(this.loaded)}, max_loaded=${this.maxLoaded}, default=${pyRepr(this.default)})`;
  }
}
