/**
 * Browser store (Cache API) with an injected in-memory CacheStorage and a
 * mocked Hub — same signatures as the Node store.
 */
import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const test = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import assert from "node:assert/strict";
import { HfCacheError, fetchFile, resolveFile, snapshot, type CacheStorageLike } from "../src/browser.ts";

const C1 = "a".repeat(40);
const C2 = "b".repeat(40);
const ENDPOINT = "https://hub.test";
const FILES: Record<string, string> = { "config.json": '{"x":1}', "tok/vocab.txt": "a\nb", "model.safetensors": "WEIGHTS" };

function memoryCaches(): CacheStorageLike & { entries: Map<string, Map<string, Response>> } {
  const entries = new Map<string, Map<string, Response>>();
  return {
    entries,
    async open(name) {
      const m = entries.get(name) ?? new Map<string, Response>();
      entries.set(name, m);
      return {
        async match(key) { return m.get(key)?.clone(); },
        async put(key, res) { m.set(key, new Response(await res.blob(), { headers: res.headers })); },
      };
    },
  };
}

function hub() {
  const state = { main: C1, calls: [] as string[] };
  const fetchImpl = (async (input: string | URL | Request, init: RequestInit = {}) => {
    const url = new URL(String(input));
    state.calls.push(url.pathname);
    if (new Headers(init.headers).get("authorization") !== "Bearer t") return new Response(null, { status: 401 });
    const api = /^\/api\/models\/org\/m\/revision\/([^/]+)$/.exec(url.pathname);
    if (api) return Response.json({ sha: api[1] === "main" ? state.main : decodeURIComponent(api[1]!), siblings: Object.keys(FILES).map((rfilename) => ({ rfilename })) });
    const r = /^\/org\/m\/resolve\/([0-9a-f]{40})\/(.+)$/.exec(url.pathname);
    const body = r ? FILES[decodeURIComponent(r[2]!)] : undefined;
    if (!body) return new Response(null, { status: 404, headers: { "X-Error-Code": "EntryNotFound" } });
    return new Response(body, { headers: { "Content-Length": String(new TextEncoder().encode(body).byteLength) } });
  }) as typeof fetch;
  return { fetch: fetchImpl, state };
}

test("resolveFile/fetchFile: commit-pinned keys, refs entry, offline reuse", async () => {
  const caches = memoryCaches();
  const h = hub();
  const opts = { caches, fetch: h.fetch, endpoint: ENDPOINT, token: "t" };
  const key = await resolveFile("org/m", "tok/vocab.txt", opts);
  assert.equal(key, `${ENDPOINT}/org/m/resolve/${C1}/tok/vocab.txt`);
  assert.equal(await (await fetchFile("org/m", "tok/vocab.txt", opts)).text(), "a\nb");
  const buf = await fetchFile("org/m", "config.json", { ...opts, as: "arraybuffer" });
  assert.equal(new TextDecoder().decode(buf), '{"x":1}');
  assert.ok(caches.entries.get("hf-cache")!.has(`${ENDPOINT}/org/m/refs/main`));
  // offline: no network at all, "main" resolved through the cached ref
  const noNet = (() => { throw new Error("network used"); }) as unknown as typeof fetch;
  assert.equal(await resolveFile("org/m", "tok/vocab.txt", { ...opts, fetch: noNet, offline: true }), key);
  await assert.rejects(resolveFile("org/m", "model.safetensors", { ...opts, fetch: noNet, offline: true }), (e: unknown) => e instanceof HfCacheError && e.code === "OfflineCacheMiss");
  // unreachable Hub also falls back to the cached ref
  const down = (async () => { throw new TypeError("Failed to fetch"); }) as unknown as typeof fetch;
  assert.equal(await resolveFile("org/m", "tok/vocab.txt", { ...opts, fetch: down }), key);
});

test("branch moves: new commit key; errors map to codes; custom cache name", async () => {
  const caches = memoryCaches();
  const h = hub();
  const opts = { caches, fetch: h.fetch, endpoint: ENDPOINT, token: "t", cacheDir: "laya" };
  await resolveFile("org/m", "config.json", opts);
  h.state.main = C2;
  assert.equal(await resolveFile("org/m", "config.json", opts), `${ENDPOINT}/org/m/resolve/${C2}/config.json`);
  assert.ok(caches.entries.has("laya"));
  await assert.rejects(resolveFile("org/m", "nope.bin", opts), (e: unknown) => e instanceof HfCacheError && e.code === "EntryNotFound");
  await assert.rejects(resolveFile("org/m", "config.json", { ...opts, token: "x", revision: "main" }), (e: unknown) => e instanceof HfCacheError && e.code === "Unauthorized");
});

test("snapshot: patterns, progress, dir is the commit-pinned base URL", async () => {
  const caches = memoryCaches();
  const h = hub();
  const seen: string[] = [];
  const snap = await snapshot("org/m", { caches, fetch: h.fetch, endpoint: ENDPOINT, token: "t", allowPatterns: ["*.json", "tok/"], onProgress: (e) => seen.push(e.file) });
  assert.equal(snap.commit, C1);
  assert.deepEqual(snap.files, ["config.json", "tok/vocab.txt"]);
  assert.equal(snap.dir, `${ENDPOINT}/org/m/resolve/${C1}/`);
  assert.ok(seen.includes("config.json") && seen.includes("tok/vocab.txt"));
  assert.ok(!h.state.calls.some((p) => p.endsWith("model.safetensors")));
});
