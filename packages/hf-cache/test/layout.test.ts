/**
 * Cache layout with a mocked Hub (temp cacheDir, no network): the exact
 * files/symlinks huggingface_hub writes, atomicity, revision moves, cached
 * 404s, redirects/auth, network fallback, snapshot filtering — and Python's
 * huggingface_hub reading what we wrote.
 */
import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const test = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, lstatSync, readdirSync, readFileSync, readlinkSync } from "node:fs";
import { join } from "node:path";
import { HfCacheError, resolveFile, snapshot, tryToLoadFromCache, type ProgressEvent } from "../src/index.ts";
import { pyTryToLoad, pythonSkip, tempDir } from "./helpers.ts";

const REPO = "org/model";
const C1 = "1".repeat(40);
const C2 = "2".repeat(40);
const ENDPOINT = "https://hub.test";
const CDN = "https://cdn.test";

interface MockFile { body: Uint8Array; lfs: boolean }
const enc = new TextEncoder();
const gitSha = (b: Uint8Array) => createHash("sha1").update(`blob ${b.byteLength}\0`).update(b).digest("hex");
const sha256 = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");
const etagOf = (f: MockFile) => (f.lfs ? sha256(f.body) : gitSha(f.body));

interface Call { method: string; url: string; auth: string | null }

/** A tiny fake Hub mimicking huggingface.co's resolve/API responses. */
function mockHub(opts: { commits?: Record<string, Record<string, MockFile>>; branch?: string; failBody?: boolean; wrongSize?: boolean } = {}) {
  const weights = new Uint8Array(3000).map((_, i) => (i * 7) % 256);
  const commits = opts.commits ?? {
    [C1]: {
      "config.json": { body: enc.encode('{"hidden": 8}'), lfs: false },
      "sub/tok.json": { body: enc.encode('{"vocab": []}'), lfs: false },
      "model.safetensors": { body: weights, lfs: true },
      "README.md": { body: enc.encode("# hi"), lfs: false },
    },
  };
  const state = { main: opts.branch ?? C1, calls: [] as Call[] };
  const fetchImpl = (async (input: string | URL | Request, init: RequestInit = {}) => {
    const url = new URL(String(input));
    const method = init.method ?? "GET";
    const auth = new Headers(init.headers).get("authorization");
    state.calls.push({ method, url: url.href, auth });
    if (url.origin === CDN) {
      const file = Object.values(commits).flatMap((c) => Object.values(c)).find((f) => sha256(f.body) === url.pathname.slice(1));
      if (!file) return new Response("gone", { status: 404 });
      let body: BodyInit = file.body as BodyInit;
      if (opts.failBody) {
        body = new ReadableStream({ start(c) { c.enqueue(file.body.subarray(0, 100)); c.error(new Error("connection reset")); } });
      }
      return new Response(method === "HEAD" ? null : body, { status: 200, headers: { "Content-Length": String(file.body.byteLength) } });
    }
    const api = /^\/api\/models\/(.+)\/revision\/([^/]+)$/.exec(url.pathname);
    if (api) {
      const rev = decodeURIComponent(api[2]!);
      const commit = rev === "main" ? state.main : rev;
      if (!commits[commit]) return new Response("{}", { status: 404, headers: { "X-Error-Code": "RevisionNotFound" } });
      return Response.json({ sha: commit, siblings: Object.keys(commits[commit]!).map((rfilename) => ({ rfilename })) });
    }
    const cache = /^\/api\/resolve-cache\/models\/org\/model\/([0-9a-f]{40})\/(.+)$/.exec(url.pathname);
    const resolve = /^\/org\/model\/resolve\/([^/]+)\/(.+)$/.exec(url.pathname);
    const m = cache ?? resolve;
    if (!m) return new Response("?", { status: 404 });
    if (auth !== "Bearer hf_test") return new Response(null, { status: 401, headers: { "X-Error-Code": "GatedRepo", "X-Error-Message": "gated" } });
    const rev = decodeURIComponent(m[1]!);
    const commit = rev === "main" ? state.main : rev;
    const path = m[2]!.split("/").map(decodeURIComponent).join("/");
    const file = commits[commit]?.[path];
    if (!file) return new Response(null, { status: 404, headers: { "X-Repo-Commit": commit, "X-Error-Code": "EntryNotFound" } });
    const etag = `"${etagOf(file)}"`;
    const size = opts.wrongSize ? file.body.byteLength + 1 : file.body.byteLength;
    if (file.lfs) {
      // like huggingface.co: 302 to the CDN carrying the linked metadata
      return new Response(null, { status: 302, headers: { Location: `${CDN}/${sha256(file.body)}`, "X-Repo-Commit": commit, "X-Linked-Etag": etag, "X-Linked-Size": String(size) } });
    }
    if (resolve) {
      // regular files: relative 307 to /api/resolve-cache (same origin)
      return new Response(null, { status: 307, headers: { Location: `/api/resolve-cache/models/org/model/${commit}/${m[2]}`, "X-Repo-Commit": commit, "X-Linked-Etag": etag, "Content-Length": "244" } });
    }
    return new Response(method === "HEAD" ? null : (file.body as BodyInit), { status: 200, headers: { ETag: etag, "Content-Length": String(file.body.byteLength), "X-Repo-Commit": commit } });
  }) as typeof fetch;
  return { fetch: fetchImpl, state, commits, weights };
}

const base = (fetchImpl: typeof fetch, cacheDir: string) => ({ fetch: fetchImpl, cacheDir, endpoint: ENDPOINT, token: "hf_test" });
const leftovers = (dir: string): string[] =>
  readdirSync(dir, { recursive: true }).map(String).filter((p) => p.endsWith(".incomplete") || p.endsWith(".tmp"));

test("regular + nested + LFS files land in blobs/, snapshots/ symlinks, refs/main", async () => {
  const { dir, cleanup } = await tempDir();
  try {
    const hub = mockHub();
    const opts = base(hub.fetch, dir);
    const storage = join(dir, "models--org--model");
    const cfg = await resolveFile(REPO, "config.json", opts);
    assert.equal(cfg, join(storage, "snapshots", C1, "config.json"));
    assert.equal(readFileSync(join(storage, "refs", "main"), "utf8"), C1);
    const cfgEtag = gitSha(enc.encode('{"hidden": 8}'));
    assert.equal(readlinkSync(cfg), `../../blobs/${cfgEtag}`);
    assert.equal(readFileSync(join(storage, "blobs", cfgEtag), "utf8"), '{"hidden": 8}');
    const tok = await resolveFile(REPO, "sub/tok.json", opts);
    assert.equal(readlinkSync(tok), `../../../blobs/${gitSha(enc.encode('{"vocab": []}'))}`);
    const events: ProgressEvent[] = [];
    const w = await resolveFile(REPO, "model.safetensors", { ...opts, onProgress: (e) => events.push(e) });
    assert.equal(readlinkSync(w), `../../blobs/${sha256(hub.weights)}`);
    assert.deepEqual(new Uint8Array(readFileSync(w)), hub.weights);
    assert.ok(events.length >= 2 && events.at(-1)!.loaded === 3000 && events.at(-1)!.total === 3000);
    for (let i = 1; i < events.length; i++) assert.ok(events[i]!.loaded >= events[i - 1]!.loaded);
    // the token goes to the Hub but never to the CDN
    const cdn = hub.state.calls.filter((c) => c.url.startsWith(CDN));
    assert.ok(cdn.length >= 1 && cdn.every((c) => c.auth === null));
    assert.ok(hub.state.calls.filter((c) => c.url.startsWith(ENDPOINT)).every((c) => c.auth === "Bearer hf_test"));
    assert.deepEqual(leftovers(dir), []);
  } finally {
    await cleanup();
  }
});

test("cached files: a branch costs only HEAD requests (no GET); a commit revision costs nothing", async () => {
  const { dir, cleanup } = await tempDir();
  try {
    const hub = mockHub();
    const opts = base(hub.fetch, dir);
    const first = await resolveFile(REPO, "config.json", opts);
    hub.state.calls.length = 0;
    assert.equal(await resolveFile(REPO, "config.json", opts), first);
    assert.ok(hub.state.calls.every((c) => c.method === "HEAD"), JSON.stringify(hub.state.calls));
    hub.state.calls.length = 0;
    assert.equal(await resolveFile(REPO, "config.json", { ...opts, revision: C1 }), first);
    assert.deepEqual(hub.state.calls, []);
  } finally {
    await cleanup();
  }
});

test("branch moves to a new commit: new snapshot + refs update, unchanged blob reused", async () => {
  const { dir, cleanup } = await tempDir();
  try {
    const cfg = { body: enc.encode('{"hidden": 8}'), lfs: false };
    const hub = mockHub({ commits: { [C1]: { "config.json": cfg }, [C2]: { "config.json": cfg, "new.txt": { body: enc.encode("n"), lfs: false } } } });
    const opts = base(hub.fetch, dir);
    await resolveFile(REPO, "config.json", opts);
    hub.state.main = C2;
    hub.state.calls.length = 0;
    const p2 = await resolveFile(REPO, "config.json", opts);
    assert.equal(p2, join(dir, "models--org--model", "snapshots", C2, "config.json"));
    assert.equal(readFileSync(join(dir, "models--org--model", "refs", "main"), "utf8"), C2);
    assert.ok(hub.state.calls.every((c) => c.method === "HEAD"), "blob reused, not re-downloaded");
    assert.equal(readdirSync(join(dir, "models--org--model", "blobs")).length, 1);
    // the old commit's snapshot stays valid
    assert.ok(existsSync(join(dir, "models--org--model", "snapshots", C1, "config.json")));
  } finally {
    await cleanup();
  }
});

test("missing file: EntryNotFound, cached as .no_exist so offline lookups answer too", async () => {
  const { dir, cleanup } = await tempDir();
  try {
    const hub = mockHub();
    const opts = base(hub.fetch, dir);
    const notFound = (e: unknown) => e instanceof HfCacheError && e.code === "EntryNotFound";
    await assert.rejects(resolveFile(REPO, "missing.bin", opts), notFound);
    assert.ok(existsSync(join(dir, "models--org--model", ".no_exist", C1, "missing.bin")));
    await assert.rejects(resolveFile(REPO, "missing.bin", { ...opts, offline: true }), notFound);
    assert.equal(await tryToLoadFromCache(REPO, "missing.bin", { cacheDir: dir }), null);
    assert.equal(await tryToLoadFromCache(REPO, "never-asked.bin", { cacheDir: dir }), undefined);
  } finally {
    await cleanup();
  }
});

test("failed or short downloads leave no blob, pointer or temp file", async () => {
  for (const variant of [{ failBody: true }, { wrongSize: true }]) {
    const { dir, cleanup } = await tempDir();
    try {
      const hub = mockHub(variant);
      await assert.rejects(resolveFile(REPO, "model.safetensors", base(hub.fetch, dir)));
      const storage = join(dir, "models--org--model");
      assert.deepEqual(existsSync(join(storage, "blobs")) ? readdirSync(join(storage, "blobs")) : [], []);
      assert.ok(!existsSync(join(storage, "snapshots", C1, "model.safetensors")));
      assert.deepEqual(leftovers(dir), []);
    } finally {
      await cleanup();
    }
  }
});

test("auth failures map to error codes; unreachable Hub falls back to the cache", async () => {
  const { dir, cleanup } = await tempDir();
  try {
    const hub = mockHub();
    await assert.rejects(resolveFile(REPO, "config.json", { ...base(hub.fetch, dir), token: "wrong" }), (e: unknown) => e instanceof HfCacheError && e.code === "GatedRepo" && e.status === 401);
    const cached = await resolveFile(REPO, "config.json", base(hub.fetch, dir));
    const down = (async () => { throw new TypeError("fetch failed"); }) as unknown as typeof fetch;
    assert.equal(await resolveFile(REPO, "config.json", base(down, dir)), cached);
    await assert.rejects(resolveFile(REPO, "README.md", base(down, dir)), (e: unknown) => e instanceof HfCacheError && e.code === "OfflineCacheMiss" && e.cause instanceof TypeError);
  } finally {
    await cleanup();
  }
});

test("HF_HUB_CACHE selects the cache directory when cacheDir is omitted", async () => {
  const { dir, cleanup } = await tempDir();
  const prev = process.env.HF_HUB_CACHE;
  process.env.HF_HUB_CACHE = dir;
  try {
    const hub = mockHub();
    const p = await resolveFile(REPO, "config.json", { fetch: hub.fetch, endpoint: ENDPOINT, token: "hf_test" });
    assert.ok(p.startsWith(join(dir, "models--org--model")));
  } finally {
    if (prev === undefined) delete process.env.HF_HUB_CACHE;
    else process.env.HF_HUB_CACHE = prev;
    await cleanup();
  }
});

test("snapshot: pins one commit, downloads only allowPatterns minus ignorePatterns", async () => {
  const { dir, cleanup } = await tempDir();
  try {
    const hub = mockHub();
    const snap = await snapshot(REPO, { ...base(hub.fetch, dir), allowPatterns: ["*.json", "*.md"], ignorePatterns: ["sub/"] });
    assert.equal(snap.commit, C1);
    assert.deepEqual(snap.files, ["README.md", "config.json"]);
    assert.equal(snap.dir, join(dir, "models--org--model", "snapshots", C1));
    assert.ok(!hub.state.calls.some((c) => c.url.includes("model.safetensors") || c.url.includes("tok.json")));
    const all = await snapshot(REPO, base(hub.fetch, dir));
    assert.deepEqual(all.files, ["README.md", "config.json", "model.safetensors", "sub/tok.json"]);
    for (const f of all.files) assert.ok(lstatSync(join(all.dir, f)).isSymbolicLink(), f);
    const offline = await snapshot(REPO, { cacheDir: dir, offline: true, allowPatterns: ["*.json"] });
    assert.deepEqual(offline.files, ["config.json", "sub/tok.json"]);
  } finally {
    await cleanup();
  }
});

test("Python huggingface_hub reads the cache we wrote (offline)", async () => {
  const { dir, cleanup } = await tempDir();
  try {
    const hub = mockHub();
    await snapshot(REPO, base(hub.fetch, dir));
    await assert.rejects(resolveFile(REPO, "missing.bin", base(hub.fetch, dir)));
    for (const file of ["config.json", "sub/tok.json", "model.safetensors"]) {
      assert.equal(pyTryToLoad(REPO, file, dir), await tryToLoadFromCache(REPO, file, { cacheDir: dir }), file);
      assert.equal(pyTryToLoad(REPO, file, dir, C1), join(dir, "models--org--model", "snapshots", C1, ...file.split("/")));
    }
    assert.equal(pyTryToLoad(REPO, "missing.bin", dir), "<NO_EXIST>");
  } finally {
    await cleanup();
  }
}, { skip: pythonSkip });
