/**
 * Browser store: the same `resolveFile` / `snapshot` / `fetchFile`
 * signatures, backed by the Cache API (`caches.open(cacheDir ?? "hf-cache")`).
 *
 * Keys are commit-pinned resolve URLs
 * (`{endpoint}/{repo}/resolve/{commit}/{file}`), which are immutable, so a
 * hit never needs revalidation. Revision -> commit mappings are stored as
 * tiny synthetic entries under `{endpoint}/{repo}/refs/{revision}`, which
 * is what makes offline resolution of "main" possible.
 *
 * `resolveFile` returns the key URL; read it with `fetchFile` (or
 * `(await caches.open(name)).match(url)`).
 */
import {
  COMMIT_RE,
  HfCacheError,
  authHeaders,
  endpointOf,
  fetchRevisionInfo,
  filterFiles,
  httpError,
  isNetworkError,
  isOffline,
  mapLimit,
  readBody,
  resolveUrl,
  validateFilename,
  validateRepo,
  type HubOptions,
  type SnapshotOptions,
  type SnapshotResult,
} from "./common.ts";

/** Minimal CacheStorage surface used here (lets tests and non-window workers inject one). */
export interface CacheLike {
  match(request: string): Promise<Response | undefined>;
  put(request: string, response: Response): Promise<void>;
}
export interface CacheStorageLike {
  open(name: string): Promise<CacheLike>;
}

export interface BrowserHubOptions extends HubOptions {
  /** CacheStorage to use (default globalThis.caches). */
  caches?: CacheStorageLike;
}

async function openCache(options: BrowserHubOptions): Promise<CacheLike> {
  const storage = options.caches ?? (globalThis as { caches?: CacheStorageLike }).caches;
  if (!storage) throw new HfCacheError("OfflineCacheMiss", "no Cache API in this runtime (pass options.caches)");
  return storage.open(options.cacheDir ?? "hf-cache");
}

function base(options: HubOptions, repo: string): string {
  const type = options.repoType ?? "model";
  return `${endpointOf(options)}/${type === "model" ? "" : `${type}s/`}${repo}`;
}

function refKey(options: HubOptions, repo: string, revision: string): string {
  return `${base(options, repo)}/refs/${encodeURIComponent(revision)}`;
}

async function cachedCommit(cache: CacheLike, options: HubOptions, repo: string, revision: string): Promise<string | undefined> {
  if (COMMIT_RE.test(revision)) return revision;
  const hit = await cache.match(refKey(options, repo, revision));
  const commit = hit ? (await hit.text()).trim() : undefined;
  return commit && COMMIT_RE.test(commit) ? commit : undefined;
}

async function rememberCommit(cache: CacheLike, options: HubOptions, repo: string, revision: string, commit: string): Promise<void> {
  if (revision !== commit) await cache.put(refKey(options, repo, revision), new Response(commit, { headers: { "Content-Type": "text/plain" } }));
}

async function pinCommit(cache: CacheLike, repo: string, options: BrowserHubOptions): Promise<{ commit: string; files?: string[] }> {
  const revision = options.revision ?? "main";
  if (COMMIT_RE.test(revision)) return { commit: revision };
  if (isOffline(options)) {
    const commit = await cachedCommit(cache, options, repo, revision);
    if (!commit) throw new HfCacheError("OfflineCacheMiss", `${repo}@${revision}: no cached revision (offline mode)`);
    return { commit };
  }
  try {
    const info = await fetchRevisionInfo(repo, options);
    await rememberCommit(cache, options, repo, revision, info.commit);
    return info;
  } catch (e) {
    if (!isNetworkError(e)) throw e;
    const commit = await cachedCommit(cache, options, repo, revision);
    if (!commit) throw new HfCacheError("OfflineCacheMiss", `${repo}@${revision}: Hub unreachable and no cached revision`, { cause: e });
    return { commit };
  }
}

async function ensureCached(cache: CacheLike, repo: string, file: string, commit: string, options: BrowserHubOptions): Promise<string> {
  const key = resolveUrl(endpointOf(options), repo, options.repoType ?? "model", commit, file);
  if (await cache.match(key)) return key;
  if (isOffline(options)) throw new HfCacheError("OfflineCacheMiss", `${repo}@${commit}:${file} is not cached (offline mode)`);
  let res: Response;
  try {
    res = await (options.fetch ?? globalThis.fetch)(key, {
      headers: authHeaders(options.token),
      ...(options.signal ? { signal: options.signal } : {}),
    });
  } catch (e) {
    if (isNetworkError(e)) throw new HfCacheError("OfflineCacheMiss", `${repo}@${commit}:${file} is not cached and the Hub is unreachable`, { cause: e });
    throw e;
  }
  if (!res.ok) {
    await res.body?.cancel();
    throw httpError(res, `${repo}@${commit}:${file}`);
  }
  const len = res.headers.get("content-length");
  const total = len === null ? undefined : Number(len);
  const chunks: Uint8Array[] = [];
  for await (const chunk of readBody(res, file, total, options.onProgress)) chunks.push(chunk);
  const body = new Blob(chunks as BlobPart[]);
  if (total !== undefined && body.size !== total) {
    throw new HfCacheError("HttpError", `${file}: downloaded ${body.size} bytes, expected ${total}`);
  }
  // Only a complete body is ever put, so a failed download leaves no entry.
  await cache.put(key, new Response(body, { headers: { "Content-Type": res.headers.get("content-type") ?? "application/octet-stream", "Content-Length": String(body.size) } }));
  return key;
}

/** Cache key URL (commit-pinned resolve URL) of `file`, downloading it into the cache if needed. */
export async function resolveFile(repo: string, file: string, options: BrowserHubOptions = {}): Promise<string> {
  validateRepo(repo);
  validateFilename(file);
  const cache = await openCache(options);
  // Online, a branch/tag is re-pinned (one small API call); offline or with
  // the Hub unreachable the cached mapping is used. Commit hashes skip it.
  const { commit } = await pinCommit(cache, repo, options);
  return ensureCached(cache, repo, file, commit, options);
}

/** Same contract as the Node version; `dir` is the commit-pinned base URL `{endpoint}/{repo}/resolve/{commit}/`. */
export async function snapshot(repo: string, options: SnapshotOptions & BrowserHubOptions = {}): Promise<SnapshotResult> {
  validateRepo(repo);
  for (const f of options.files ?? []) validateFilename(f);
  const cache = await openCache(options);
  const pinned = await pinCommit(cache, repo, options);
  let files: string[];
  if (options.files) files = [...options.files];
  else if (pinned.files) files = filterFiles(pinned.files, options.allowPatterns, options.ignorePatterns);
  else if (!isOffline(options)) files = filterFiles((await fetchRevisionInfo(repo, { ...options, revision: pinned.commit })).files, options.allowPatterns, options.ignorePatterns);
  else throw new HfCacheError("OfflineCacheMiss", `${repo}: offline snapshot in the browser needs an explicit \`files\` list (the Cache API has no directory listing)`);
  await mapLimit(files, options.concurrency ?? 4, (f) => ensureCached(cache, repo, f, pinned.commit, options));
  const dir = resolveUrl(endpointOf(options), repo, options.repoType ?? "model", pinned.commit, "x").replace(/x$/, "");
  return { dir, files: files.sort(), commit: pinned.commit };
}

/** The file's content from the cache (downloading first if needed). Default: a Blob. */
export async function fetchFile(repo: string, file: string, options: BrowserHubOptions & { as: "arraybuffer" }): Promise<ArrayBuffer>;
export async function fetchFile(repo: string, file: string, options?: BrowserHubOptions & { as?: "blob" }): Promise<Blob>;
export async function fetchFile(repo: string, file: string, options: BrowserHubOptions & { as?: "blob" | "arraybuffer" } = {}): Promise<Blob | ArrayBuffer> {
  const key = await resolveFile(repo, file, options);
  const hit = await (await openCache(options)).match(key);
  if (!hit) throw new HfCacheError("OfflineCacheMiss", `${key} vanished from the cache`);
  return options.as === "arraybuffer" ? hit.arrayBuffer() : hit.blob();
}

export {
  HfCacheError,
  filterFiles,
  globToRegExp,
  type HfCacheErrorCode,
  type HubOptions,
  type ProgressEvent,
  type RepoType,
  type SnapshotOptions,
  type SnapshotResult,
} from "./common.ts";
