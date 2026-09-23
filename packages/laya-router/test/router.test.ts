/**
 * Port of the routing tests in laya-mlx tests/test_router.py and
 * tests/test_runtime.py (no weights: `route` is pure, loaders are fakes),
 * with the thread-safety tests (#95) as async-concurrency tests, plus the
 * JS-specific in-use eviction semantics. Language detection itself is tested
 * in @johnhenry/langdetect-lite.
 */
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
import { makeTest } from "./harness.ts";
const test = makeTest(bun);
import assert from "node:assert/strict";
import {
  DEFAULT_MODELS,
  MLX_MODELS,
  Router,
  matchTypedDecisionsWorkflow,
  normaliseName,
  type RoutedAgent,
} from "../src/index.ts";

const Q_GENERIC = { dept: { type: "choice", instructions: "Which team?", criteria: { billing: null, tech: null } } };
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

class FakeAgent implements RoutedAgent {
  static built = 0;
  disposed = false;
  calls = 0;
  readonly repo: string;
  readonly opts: Record<string, unknown>;
  readonly delay: number;
  constructor(repo: string, opts: Record<string, unknown>, delay = 0) {
    this.repo = repo;
    this.opts = opts;
    this.delay = delay;
    FakeAgent.built++;
  }
  async predict(state: unknown, questions: Record<string, unknown>) {
    if (this.disposed) throw new Error("used after dispose");
    this.calls++;
    await sleep(this.delay);
    if (this.disposed) throw new Error("disposed during predict");
    return { model: "laya-rl-agent", answers: Object.fromEntries(Object.keys(questions).map((k) => [k, { repo: this.repo }])), state };
  }
  dispose() {
    this.disposed = true;
  }
}
const fakeLoader = (loadMs = 0, predictMs = 0) => async (repo: string, opts: Record<string, unknown>) => {
  await sleep(loadMs);
  return new FakeAgent(repo, opts, predictMs);
};

// --------------------------------------------------------------------- routing decisions
test("unidentified Latin-script languages route to multilingual", () => {
  for (const text of [
    "Gătește-mi o rețetă de sarmale de post pentru mâine.",
    "Exportă APK-ul pentru Android și pune-l pe Drive ca să-l instalez.",
    "Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę",
    "Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım",
  ]) assert.equal(new Router().route(text).model, "multilingual", text);
});

test("the unidentified-Latin reason says what it routed on", () => {
  const r = new Router().route("Müşteriden iki kez ücret alındı ve para iadesi istiyor");
  assert.match(r.reason, /not identified/);
  assert.match(r.reason, /^Latin script, language not identified but \d+% non-English letters; not safe for the English checkpoint$/);
});

test("the identified non-English reason names the language", () => {
  const r = new Router().route("Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung die nicht korrekt ist");
  assert.match(r.reason, /'de'/);
  assert.equal(r.reason, "Latin script but language looks like 'de', not English");
});

test("English routing, explicit model, aliases", () => {
  const router = new Router();
  assert.equal(router.route("Please refund the duplicate charge on invoice 4411 today.").model, "english");
  assert.equal(router.route("refund me").model, "english");
  assert.equal(router.route("Հայերեն", null, { model: "english" }).model, "english");
  assert.equal(normaliseName("ML"), "multilingual");
  assert.equal(normaliseName("  Typed_Decisions "), "typed-decisions");
  assert.throws(
    () => normaliseName("gpt"),
    (e: Error) =>
      e.message ===
      "unknown model 'gpt'; choose one of ['english', 'multilingual', 'typed-decisions'] (or an alias: ['decisions', 'default', 'en', 'laya', 'laya-multilingual', 'laya-typed-decisions', 'ml', 'multi', 'typed', 'typed_decisions'])",
  );
});

test("languages, precedence and reasons (test_runtime)", () => {
  const router = new Router();
  const en = router.route("I was charged twice and want a refund");
  assert.equal(en.model, "english");
  assert.equal(en.reason, "English Latin text");
  assert.equal(en.repo, "convaiinnovations/laya");
  assert.equal(en.detection?.is_english, true);
  const zh = router.route("发票被重复扣款，请退款。");
  assert.equal(zh.model, "multilingual");
  assert.equal(zh.repo, "convaiinnovations/laya/multilingual");
  assert.equal(zh.reason, "non-Latin script (han, 100% of letters); the English checkpoint cannot read it");
  assert.equal(router.route("मुझे धनवापसी चाहिए।").model, "multilingual");
  assert.equal(router.route("你好", null, { model: "typed" }).model, "typed-decisions");
  assert.equal(router.route("你好", null, { model: "typed" }).reason, "explicit model='typed'");
  const t = router.route("hello", null, { task: "typed-decisions" });
  assert.deepEqual(t, { model: "typed-decisions", repo: "convaiinnovations/laya/typed-decisions", reason: "explicit task='typed-decisions'", detection: null, workflow: null });
  assert.equal(router.route("hello", null, { task: "multi" }).model, "multilingual");
  assert.equal(router.route("12345 ???").model, "english");
  assert.equal(router.route("12345 ???").reason, "no letters detected in state; using default (english)");
  assert.equal(new Router({ default: "ml" }).route("12345").model, "multilingual");
  // explicit lang
  assert.equal(router.route("Bonjour", null, { lang: "en-US" }).model, "english");
  assert.equal(router.route("Hello there", null, { lang: "fr" }).reason, "explicit lang='fr'");
  assert.equal(router.route("Hello there", null, { lang: "fr" }).model, "multilingual");
});

const CS = { action: 1, category: 1, churn_risk: 1, needs_human: 1, urgency: 1 };
test("typed-decisions workflows: exact id-set match, opt-in only, below model/task, above lang", () => {
  assert.equal(matchTypedDecisionsWorkflow(CS), "customer_service");
  assert.equal(matchTypedDecisionsWorkflow({ ...CS, extra: 1 }), null);
  assert.equal(matchTypedDecisionsWorkflow({ urgency: 1 }), null);
  assert.equal(matchTypedDecisionsWorkflow(null), null);
  const off = new Router().route("I was charged twice", CS);
  assert.equal(off.model, "english");
  assert.equal(off.workflow, "customer_service"); // reported, not acted on
  const on = new Router({ autoTaskDetection: true });
  const r = on.route("I was charged twice", CS);
  assert.equal(r.model, "typed-decisions");
  assert.equal(r.reason, "question ids match the 'customer_service' typed-decisions workflow");
  assert.equal(r.workflow, "customer_service");
  assert.equal(on.route("x", CS, { lang: "de" }).model, "typed-decisions");
  assert.equal(on.route("x", CS, { model: "en" }).model, "english");
  assert.equal(on.route("x", CS, { task: "english" }).model, "english");
  assert.equal(on.route("I was charged twice", Q_GENERIC).model, "english");
});

test("model tables: bundle, standalone, mlx, overrides", () => {
  assert.deepEqual(new Router().models, DEFAULT_MODELS);
  assert.equal(new Router({ standaloneRepos: true }).route("x", null, { model: "ml" }).repo, "convaiinnovations/laya-multilingual");
  assert.deepEqual(new Router({ mlxRepos: true }).models, MLX_MODELS);
  const r = new Router({ models: { en: "/models/english", typed: ["org/repo", "sub"] } });
  assert.equal(r.route("hello world, how are you", null, { model: "english" }).repo, "/models/english");
  assert.equal(r.route("x", null, { model: "typed" }).repo, "org/repo/sub");
  assert.throws(() => new Router({ models: { nope: "x" } }), /unknown model 'nope'/);
});

// --------------------------------------------------------------------- residency (test_runtime)
test("attach / load / unload residency", async () => {
  const router = new Router<FakeAgent>({ loader: fakeLoader() });
  const a = new FakeAgent("a", {}), b = new FakeAgent("b", {});
  router.attach("english", a);
  router.attach("multilingual", b);
  assert.equal(await router.load("english"), a);
  assert.equal(router.maxLoaded, 2);
  router.unload("english");
  assert.deepEqual(router.loaded, ["multilingual"]);
  router.unload();
  assert.deepEqual(router.loaded, []);
  assert.equal(a.disposed || b.disposed, false, "attached agents belong to the caller");
  assert.equal(String(router), "Router(loaded=[], max_loaded=2, default='english')");
});

test("load passes repo, subfolder and loadOptions to the loader; predict attaches routing", async () => {
  const router = new Router<FakeAgent>({ loader: fakeLoader(), loadOptions: { backend: "cpu", dtype: "f32" } });
  const ml = await router.load("multilingual");
  assert.equal(ml.repo, "convaiinnovations/laya");
  assert.deepEqual(ml.opts, { backend: "cpu", dtype: "f32", subfolder: "multilingual" });
  const en = await router.load("en");
  assert.deepEqual(en.opts, { backend: "cpu", dtype: "f32" });
  assert.equal(ml.disposed, true, "maxLoaded=1 evicts (and disposes) the LRU agent the router built");
  const r = await router.predict("Mein Konto wurde zweimal belastet und ich möchte eine Rückerstattung", Q_GENERIC);
  assert.equal(r.routing.model, "multilingual");
  assert.equal(r.routing.repo, "convaiinnovations/laya/multilingual");
  assert.equal(r.routing.detection?.language, "de");
  assert.deepEqual(router.loaded, ["multilingual"]);
  const r2 = await router.systemOne("I was charged twice", Q_GENERIC, { model: "typed" });
  assert.equal(r2.routing.reason, "explicit model='typed'");
});

test("preload builds everything and raises maxLoaded", async () => {
  const router = new Router<FakeAgent>({ loader: fakeLoader() });
  await router.preload();
  assert.deepEqual(router.loaded.sort(), ["english", "multilingual", "typed-decisions"]);
  assert.equal(router.maxLoaded, 3);
  const r2 = new Router<FakeAgent>({ loader: fakeLoader() });
  await r2.preload(["en", "ml"]);
  assert.deepEqual(r2.loaded, ["english", "multilingual"]);
  assert.equal(r2.maxLoaded, 2);
});

// --------------------------------------------------------------------- concurrency (#95)
test("concurrent loads share one agent (one build)", async () => {
  let constructions = 0;
  const router = new Router<FakeAgent>({
    loader: async (repo, opts) => {
      await sleep(50); // widen the check-then-build window
      constructions++;
      return new FakeAgent(repo, opts);
    },
  });
  const got = await Promise.all(Array.from({ length: 8 }, () => router.load("english")));
  assert.equal(new Set(got).size, 1);
  assert.equal(constructions, 1);
  assert.deepEqual(router.loaded, ["english"]);
});

test("the concurrent hot path keeps the LRU consistent", async () => {
  const router = new Router<FakeAgent>({ loader: fakeLoader(5), maxLoaded: 3 });
  await router.load("english");
  await Promise.all(Array.from({ length: 20 }, () => router.load("english")));
  assert.deepEqual(router.loaded, ["english"]);
  // concurrent predicts on alternating models with room for all
  await Promise.all(Array.from({ length: 30 }, (_, i) => router.predict("x", Q_GENERIC, { model: ["en", "ml", "typed"][i % 3]! })));
  assert.deepEqual([...router.loaded].sort(), ["english", "multilingual", "typed-decisions"]);
});

test("a failed build rejects every waiter and is retried on the next call", async () => {
  let attempts = 0;
  const router = new Router<FakeAgent>({
    loader: async (repo, opts) => {
      await sleep(10);
      if (++attempts === 1) throw new Error("download failed");
      return new FakeAgent(repo, opts);
    },
  });
  const results = await Promise.allSettled([router.load("english"), router.load("english"), router.predict("hello", Q_GENERIC)]);
  assert.deepEqual(results.map((r) => r.status), ["rejected", "rejected", "rejected"]);
  assert.deepEqual(router.loaded, []);
  assert.ok(await router.load("english"));
  assert.equal(attempts, 2);
});

test("LRU eviction never evicts an agent with a predict in flight", async () => {
  const router = new Router<FakeAgent>({ loader: fakeLoader(5, 60), maxLoaded: 1 });
  const slowEnglish = router.predict("I was charged twice and want a refund", Q_GENERIC); // holds english for 60 ms
  await sleep(20); // english is loaded and in use
  const english = await router.load("english");
  const multi = router.predict("发票被重复扣款，请退款。", Q_GENERIC); // loads multilingual meanwhile
  await sleep(15);
  assert.deepEqual([...router.loaded].sort(), ["english", "multilingual"], "residency may exceed maxLoaded while in use");
  assert.equal(english.disposed, false);
  const [r1, r2] = await Promise.all([slowEnglish, multi]);
  assert.equal(r1.routing.model, "english");
  assert.equal(r2.routing.model, "multilingual");
  assert.equal(english.disposed, true, "evicted once its predict settled");
  assert.equal(router.loaded.length, 1);
});

test("unload during a predict defers dispose until it settles", async () => {
  const router = new Router<FakeAgent>({ loader: fakeLoader(0, 40) });
  const p = router.predict("hello there friend", Q_GENERIC);
  await sleep(10);
  const agent = (await Promise.resolve((router as any).agents.get("english"))) as FakeAgent;
  router.unload();
  assert.deepEqual(router.loaded, []);
  assert.equal(agent.disposed, false);
  await p;
  assert.equal(agent.disposed, true);
});

test("many interleaved predicts across models with maxLoaded=1 all succeed on live agents", async () => {
  const router = new Router<FakeAgent>({ loader: fakeLoader(3, 7), maxLoaded: 1 });
  const results = await Promise.all(
    Array.from({ length: 24 }, (_, i) => router.predict("x", Q_GENERIC, { model: ["en", "ml", "typed"][i % 3]! })),
  );
  assert.equal(results.length, 24);
  await sleep(0);
  assert.equal(router.loaded.length, 1);
});
