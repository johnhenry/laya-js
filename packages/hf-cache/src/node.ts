/**
 * Node/Bun store: exactly the huggingface_hub cache layout, so files fetched
 * here are reused by Python (`hf download`, `hf_hub_download`,
 * `snapshot_download`) offline and vice versa.
 *
 *   {cache}/models--{org}--{name}/
 *     refs/{revision}            text file: the commit hash the revision points to
 *     blobs/{etag}               file content; etag = sha256 for LFS/Xet files, git sha1 otherwise
 *     snapshots/{commit}/{path}  relative symlink -> ../../blobs/{etag}
 *     .no_exist/{commit}/{path}  empty marker: the file does not exist at that commit
 *     trees/{commit}.json        (read only) file listing written by huggingface_hub >= 1.x
 *
 * Downloads stream to `blobs/{etag}.{random}.incomplete` and are renamed into
 * place; symlinks are created under a temporary name and renamed too, so
 * concurrent readers never observe a partial file. There is no inter-process
 * lock (huggingface_hub's `.locks/`): two processes racing on the same blob
 * both download it and the last rename wins with identical content.
 */
import { randomBytes } from "node:crypto";
import * as fs from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, relative } from "node:path";
import {
  COMMIT_RE,
  HfCacheError,
  authHeaders,
  endpointOf,
  env,
  fetchRevisionInfo,
  filterFiles,
  httpError,
  isNetworkError,
  isOffline,
  mapLimit,
  normalizeEtag,
  readBody,
  resolveUrl,
  validateFilename,
  validateRepo,
  type HubOptions,
  type RepoType,
  type SnapshotOptions,
  type SnapshotResult,
} from "./common.ts";

/** huggingface_hub's HF_HOME: $HF_HOME, else $XDG_CACHE_HOME/huggingface, else ~/.cache/huggingface. */
export function hfHome(): string {
  return env("HF_HOME") ?? join(env("XDG_CACHE_HOME") ?? join(homedir(), ".cache"), "huggingface");
}

/** The hub cache: `options.cacheDir`, $HF_HUB_CACHE, $HUGGINGFACE_HUB_CACHE (legacy), else $HF_HOME/hub. */
export function hubCacheDir(options: Pick<HubOptions, "cacheDir"> = {}): string {
  return options.cacheDir ?? env("HF_HUB_CACHE") ?? env("HUGGINGFACE_HUB_CACHE") ?? join(hfHome(), "hub");
}

/** `models--org--name` (also `datasets--…`, `spaces--…`). */
export function repoFolderName(repo: string, repoType: RepoType = "model"): string {
  return [`${repoType}s`, ...repo.split("/")].join("--");
}

/** Token: explicit, $HF_TOKEN, $HUGGING_FACE_HUB_TOKEN, then the file `hf auth login` writes. */
async function tokenOf(options: HubOptions): Promise<string | undefined> {
  if (options.token !== undefined) return options.token || undefined;
  const fromEnv = env("HF_TOKEN") ?? env("HUGGING_FACE_HUB_TOKEN");
  if (fromEnv) return fromEnv;
  try {
    return (await fs.readFile(env("HF_TOKEN_PATH") ?? join(hfHome(), "token"), "utf8")).trim() || undefined;
  } catch {
    return undefined;
  }
}

async function exists(path: string): Promise<boolean> {
  try {
    await fs.stat(path); // follows symlinks: a dangling pointer counts as missing
    return true;
  } catch {
    return false;
  }
}

function storageFolder(repo: string, options: HubOptions): string {
  return join(hubCacheDir(options), repoFolderName(repo, options.repoType ?? "model"));
}

function pointerPath(storage: string, commit: string, file: string): string {
  return join(storage, "snapshots", commit, ...file.split("/"));
}

async function readRef(storage: string, revision: string): Promise<string | undefined> {
  try {
    const commit = (await fs.readFile(join(storage, "refs", ...revision.split("/")), "utf8")).trim();
    return COMMIT_RE.test(commit) ? commit : undefined;
  } catch {
    return undefined;
  }
}

async function atomicWrite(path: string, data: string | Uint8Array): Promise<void> {
  await fs.mkdir(dirname(path), { recursive: true });
  const tmp = `${path}.${randomBytes(4).toString("hex")}.tmp`;
  await fs.writeFile(tmp, data);
  await fs.rename(tmp, path);
}

async function writeRef(storage: string, revision: string, commit: string): Promise<void> {
  if (revision === commit) return;
  if ((await readRef(storage, revision)) === commit) return;
  await atomicWrite(join(storage, "refs", ...revision.split("/")), commit);
}

/** Relative symlink pointer -> blob (copy where symlinks are unavailable, as huggingface_hub does on Windows). */
async function linkPointer(blob: string, pointer: string): Promise<void> {
  await fs.mkdir(dirname(pointer), { recursive: true });
  const tmp = `${pointer}.${randomBytes(4).toString("hex")}.tmp`;
  try {
    await fs.symlink(relative(dirname(pointer), blob), tmp);
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== "EPERM" && (e as NodeJS.ErrnoException).code !== "ENOTSUP") throw e;
    await fs.copyFile(blob, tmp);
  }
  await fs.rename(tmp, pointer);
}

/**
 * Cache lookup only (huggingface_hub `try_to_load_from_cache`): the snapshot
 * path, `null` when the cache records the file as nonexistent at that
 * commit, or `undefined` when unknown.
 */
export async function tryToLoadFromCache(repo: string, file: string, options: HubOptions = {}): Promise<string | null | undefined> {
  validateRepo(repo);
  validateFilename(file);
  const storage = storageFolder(repo, options);
  const revision = options.revision ?? "main";
  const commit = COMMIT_RE.test(revision) ? revision : await readRef(storage, revision);
  if (!commit) return undefined;
  const pointer = pointerPath(storage, commit, file);
  if (await exists(pointer)) return pointer;
  if (await exists(join(storage, ".no_exist", commit, ...file.split("/")))) return null;
  return undefined;
}

async function resolveFromCache(repo: string, file: string, options: HubOptions, cause?: unknown): Promise<string> {
  const hit = await tryToLoadFromCache(repo, file, options);
  if (hit) return hit;
  const where = `${repo}@${options.revision ?? "main"}:${file}`;
  if (hit === null) throw new HfCacheError("EntryNotFound", `${where} does not exist (cached 404)`);
  throw new HfCacheError(
    "OfflineCacheMiss",
    `${where} is not in the cache at ${hubCacheDir(options)}${cause ? " and the Hub is unreachable" : " (offline mode)"}`,
    cause === undefined ? {} : { cause },
  );
}

interface FileMetadata {
  commit: string;
  etag: string;
  size: number | undefined;
}

const REDIRECT = new Set([301, 302, 303, 307, 308]);

/**
 * HEAD the resolve URL like huggingface_hub `get_hf_file_metadata`: follow
 * same-origin redirects (renamed repos, /api/resolve-cache), stop at the
 * cross-origin CDN redirect, and read X-Repo-Commit / X-Linked-Etag /
 * X-Linked-Size (falling back to ETag / Content-Length).
 */
async function headMetadata(repo: string, file: string, storage: string, token: string | undefined, options: HubOptions): Promise<FileMetadata> {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  let url = resolveUrl(endpointOf(options), repo, options.repoType ?? "model", options.revision ?? "main", file);
  let commit: string | undefined;
  let linkedEtag: string | undefined;
  let linkedSize: number | undefined;
  for (let hop = 0; hop < 10; hop++) {
    const res = await fetchImpl(url, {
      method: "HEAD",
      redirect: "manual",
      headers: { ...authHeaders(token), "Accept-Encoding": "identity" },
      ...(options.signal ? { signal: options.signal } : {}),
    });
    await res.body?.cancel();
    commit ??= res.headers.get("x-repo-commit") ?? undefined;
    linkedEtag ??= normalizeEtag(res.headers.get("x-linked-etag"));
    const ls = res.headers.get("x-linked-size");
    if (ls !== null && linkedSize === undefined) linkedSize = Number(ls);
    const location = res.headers.get("location");
    let size: number | undefined = linkedSize;
    let etag = linkedEtag;
    if (REDIRECT.has(res.status) && location) {
      const next = new URL(location, url);
      if (next.origin === new URL(url).origin) {
        url = next.href;
        continue;
      }
    } else if (res.ok) {
      etag ??= normalizeEtag(res.headers.get("etag"));
      const cl = res.headers.get("content-length");
      if (size === undefined && cl !== null) size = Number(cl);
    } else {
      if (res.status === 404 && commit && COMMIT_RE.test(commit) && res.headers.get("x-error-code") === "EntryNotFound") {
        // Remember the miss the way huggingface_hub does, so offline lookups answer "does not exist".
        const marker = join(storage, ".no_exist", commit, ...file.split("/"));
        await fs.mkdir(dirname(marker), { recursive: true });
        await fs.writeFile(marker, "");
        await writeRef(storage, options.revision ?? "main", commit);
      }
      throw httpError(res, `${repo}@${options.revision ?? "main"}:${file}`);
    }
    if (!commit || !COMMIT_RE.test(commit)) throw new HfCacheError("HttpError", `${url}: response has no X-Repo-Commit header`);
    if (!etag) throw new HfCacheError("HttpError", `${url}: response has no ETag`);
    return { commit, etag, size };
  }
  throw new HfCacheError("HttpError", `${url}: too many redirects`);
}

/** GET following redirects manually so the token never leaves the Hub's origin. */
async function getFollowing(url: string, token: string | undefined, options: HubOptions): Promise<Response> {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  let current = url;
  let headers: Record<string, string> = { ...authHeaders(token), "Accept-Encoding": "identity" };
  for (let hop = 0; hop < 10; hop++) {
    const res = await fetchImpl(current, { redirect: "manual", headers, ...(options.signal ? { signal: options.signal } : {}) });
    const location = res.headers.get("location");
    if (!REDIRECT.has(res.status) || !location) return res;
    await res.body?.cancel();
    const next = new URL(location, current);
    if (next.origin !== new URL(current).origin) headers = { "Accept-Encoding": "identity" };
    current = next.href;
  }
  throw new HfCacheError("HttpError", `${url}: too many redirects`);
}

async function downloadBlob(url: string, blob: string, file: string, meta: FileMetadata, token: string | undefined, options: HubOptions): Promise<void> {
  const res = await getFollowing(url, token, options);
  if (!res.ok) {
    await res.body?.cancel();
    throw httpError(res, `${url}`);
  }
  const total = meta.size ?? (res.headers.get("content-length") ? Number(res.headers.get("content-length")) : undefined);
  await fs.mkdir(dirname(blob), { recursive: true });
  const tmp = `${blob}.${randomBytes(4).toString("hex")}.incomplete`;
  const handle = await fs.open(tmp, "w");
  let written = 0;
  try {
    try {
      for await (const chunk of readBody(res, file, total, options.onProgress)) {
        await handle.write(chunk);
        written += chunk.byteLength;
      }
    } finally {
      await handle.close();
    }
    if (meta.size !== undefined && written !== meta.size) {
      throw new HfCacheError("HttpError", `${file}: downloaded ${written} bytes, expected ${meta.size}`);
    }
    await fs.rename(tmp, blob);
  } catch (e) {
    await fs.rm(tmp, { force: true });
    throw e;
  }
}

/**
 * Local path of `file` in `repo` at `revision`, downloading into the cache
 * if needed. Offline (option or $HF_HUB_OFFLINE), or when the Hub is
 * unreachable, answers from the cache only. A commit-hash revision whose
 * file is cached never touches the network.
 */
export async function resolveFile(repo: string, file: string, options: HubOptions = {}): Promise<string> {
  validateRepo(repo);
  validateFilename(file);
  const revision = options.revision ?? "main";
  const storage = storageFolder(repo, options);
  if (COMMIT_RE.test(revision)) {
    const pointer = pointerPath(storage, revision, file);
    if (await exists(pointer)) return pointer;
  }
  if (isOffline(options)) return resolveFromCache(repo, file, options);

  const token = await tokenOf(options);
  let meta: FileMetadata;
  try {
    meta = await headMetadata(repo, file, storage, token, options);
  } catch (e) {
    if (isNetworkError(e)) return resolveFromCache(repo, file, options, e);
    throw e;
  }
  await writeRef(storage, revision, meta.commit);
  const pointer = pointerPath(storage, meta.commit, file);
  if (await exists(pointer)) return pointer;
  const blob = join(storage, "blobs", meta.etag);
  if (!(await exists(blob))) {
    const url = resolveUrl(endpointOf(options), repo, options.repoType ?? "model", meta.commit, file);
    await downloadBlob(url, blob, file, meta, token, options);
  } else if (meta.size !== undefined) {
    options.onProgress?.({ file, loaded: meta.size, total: meta.size });
  }
  await linkPointer(blob, pointer);
  return pointer;
}

async function walk(dir: string, prefix = ""): Promise<string[]> {
  const out: string[] = [];
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) out.push(...(await walk(join(dir, entry.name), rel)));
    else if (await exists(join(dir, entry.name))) out.push(rel);
  }
  return out;
}

async function treeCacheFiles(storage: string, commit: string): Promise<string[] | undefined> {
  try {
    const json = JSON.parse(await fs.readFile(join(storage, "trees", `${commit}.json`), "utf8")) as { format_version?: number; files?: Record<string, unknown> };
    return json.format_version === 1 && json.files ? Object.keys(json.files) : undefined;
  } catch {
    return undefined;
  }
}

async function snapshotFromCache(repo: string, options: SnapshotOptions, cause?: unknown): Promise<SnapshotResult> {
  const storage = storageFolder(repo, options);
  const revision = options.revision ?? "main";
  const commit = COMMIT_RE.test(revision) ? revision : await readRef(storage, revision);
  const dir = commit ? join(storage, "snapshots", commit) : undefined;
  if (!commit || !dir || !(await exists(dir))) {
    throw new HfCacheError("OfflineCacheMiss", `${repo}@${revision} has no cached snapshot in ${hubCacheDir(options)}`, cause === undefined ? {} : { cause });
  }
  if (options.files) {
    for (const f of options.files) await resolveFromCache(repo, f, { ...options, revision: commit }, cause);
    return { dir, files: [...options.files].sort(), commit };
  }
  const present = await walk(dir);
  const presentSet = new Set(present);
  // Prefer the full listing huggingface_hub cached, but only report files that are actually present.
  const listing = (await treeCacheFiles(storage, commit)) ?? present;
  const files = filterFiles(listing, options.allowPatterns, options.ignorePatterns).filter((f) => presentSet.has(f));
  return { dir, files: files.sort(), commit };
}

/**
 * Downloads (or finds) a set of files at one pinned commit: `files`
 * exactly, else every repo file matching `allowPatterns` minus
 * `ignorePatterns` (all files when neither is given). Offline / unreachable
 * Hub: answers from the cached snapshot.
 */
export async function snapshot(repo: string, options: SnapshotOptions = {}): Promise<SnapshotResult> {
  validateRepo(repo);
  for (const f of options.files ?? []) validateFilename(f);
  const revision = options.revision ?? "main";
  if (isOffline(options)) return snapshotFromCache(repo, options);
  if (COMMIT_RE.test(revision) && options.files) {
    const storage = storageFolder(repo, options);
    const all = await Promise.all(options.files.map((f) => exists(pointerPath(storage, revision, f))));
    if (all.every(Boolean)) return { dir: join(storage, "snapshots", revision), files: [...options.files].sort(), commit: revision };
  }
  const token = await tokenOf(options);
  let info;
  try {
    info = await fetchRevisionInfo(repo, { ...options, token });
  } catch (e) {
    if (isNetworkError(e)) return snapshotFromCache(repo, options, e);
    throw e;
  }
  const storage = storageFolder(repo, options);
  await writeRef(storage, revision, info.commit);
  const files = options.files ? [...options.files] : filterFiles(info.files, options.allowPatterns, options.ignorePatterns);
  await mapLimit(files, options.concurrency ?? 4, (f) => resolveFile(repo, f, { ...options, token, revision: info.commit }));
  return { dir: join(storage, "snapshots", info.commit), files: files.sort(), commit: info.commit };
}

/**
 * The file's content. `as: "blob"` (default) is a lazy, disk-backed Blob in
 * Node (`fs.openAsBlob`) — ideal for `openSafetensors`; `"arraybuffer"`
 * reads it fully.
 */
export async function fetchFile(repo: string, file: string, options: HubOptions & { as: "arraybuffer" }): Promise<ArrayBuffer>;
export async function fetchFile(repo: string, file: string, options?: HubOptions & { as?: "blob" }): Promise<Blob>;
export async function fetchFile(repo: string, file: string, options: HubOptions & { as?: "blob" | "arraybuffer" } = {}): Promise<Blob | ArrayBuffer> {
  const path = await resolveFile(repo, file, options);
  if (options.as === "arraybuffer") {
    const buf = await fs.readFile(path);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  }
  const nodeFs = (await import("node:fs")) as { openAsBlob?: (p: string) => Promise<Blob> };
  if (typeof nodeFs.openAsBlob === "function") return nodeFs.openAsBlob(path);
  return new Blob([await fs.readFile(path)]);
}
