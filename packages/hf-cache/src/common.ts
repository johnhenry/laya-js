/**
 * Runtime-neutral pieces shared by the Node/Bun (filesystem) and browser
 * (Cache API) stores: options, errors, Hub URL construction, response-header
 * parsing, glob filtering. No Node or DOM imports here.
 */

export type RepoType = "model" | "dataset" | "space";

export interface ProgressEvent {
  /** Repo-relative file path. */
  readonly file: string;
  readonly loaded: number;
  /** Total bytes when the server reported it. */
  readonly total: number | undefined;
}

export interface HubOptions {
  /** Branch, tag or 40-hex commit hash (default "main"). */
  revision?: string;
  /** Default "model". */
  repoType?: RepoType;
  /** Access token. Node/Bun default: $HF_TOKEN, $HUGGING_FACE_HUB_TOKEN, then the token file huggingface_hub writes. */
  token?: string;
  /**
   * Node/Bun: the hub cache directory (default: $HF_HUB_CACHE, else
   * $HF_HOME/hub, else ~/.cache/huggingface/hub — the huggingface_hub
   * default). Browser: the Cache API cache name (default "hf-cache").
   */
  cacheDir?: string;
  /** Hub endpoint (default: $HF_ENDPOINT or https://huggingface.co). */
  endpoint?: string;
  /** fetch implementation (default globalThis.fetch). */
  fetch?: typeof fetch;
  onProgress?: (event: ProgressEvent) => void;
  /** Never touch the network (default: $HF_HUB_OFFLINE is truthy). */
  offline?: boolean;
  signal?: AbortSignal;
}

export interface SnapshotOptions extends HubOptions {
  /** Exact repo-relative paths to fetch. */
  files?: readonly string[];
  /** fnmatch-style patterns (as huggingface_hub `allow_patterns`); a trailing "/" means the whole folder. */
  allowPatterns?: readonly string[];
  ignorePatterns?: readonly string[];
  /** Parallel downloads (default 4). */
  concurrency?: number;
}

export interface SnapshotResult {
  /** Node/Bun: the snapshot directory. Browser: the commit-pinned base URL (cache key prefix). */
  readonly dir: string;
  /** Repo-relative paths now available under `dir`, sorted. */
  readonly files: string[];
  /** The commit the revision resolved to. */
  readonly commit: string;
}

export type HfCacheErrorCode =
  | "EntryNotFound"
  | "RepoNotFound"
  | "RevisionNotFound"
  | "GatedRepo"
  | "Unauthorized"
  | "OfflineCacheMiss"
  | "InvalidFilename"
  | "HttpError";

export class HfCacheError extends Error {
  readonly code: HfCacheErrorCode;
  readonly status: number | undefined;
  constructor(code: HfCacheErrorCode, message: string, options: { status?: number; cause?: unknown } = {}) {
    super(`hf-cache: ${message}`, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = "HfCacheError";
    this.code = code;
    this.status = options.status;
  }
}

export const COMMIT_RE = /^[0-9a-f]{40}$/;
export const DEFAULT_ENDPOINT = "https://huggingface.co";

/** Environment variable lookup that works without `process` (browsers). */
export function env(name: string): string | undefined {
  const p = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  const v = p?.env?.[name];
  return v === undefined || v === "" ? undefined : v;
}

/** huggingface_hub's truthy env parsing ("1", "ON", "YES", "TRUE", case-insensitive). */
export function envFlag(name: string): boolean {
  return ["1", "ON", "YES", "TRUE"].includes((env(name) ?? "").toUpperCase());
}

export function isOffline(options: HubOptions): boolean {
  return options.offline ?? envFlag("HF_HUB_OFFLINE");
}

export function endpointOf(options: HubOptions): string {
  return (options.endpoint ?? env("HF_ENDPOINT") ?? DEFAULT_ENDPOINT).replace(/\/+$/, "");
}

export function validateRepo(repo: string): void {
  if (!/^[\w.-]+(\/[\w.-]+)?$/.test(repo) || repo.includes("..")) {
    throw new HfCacheError("RepoNotFound", `invalid repo id ${JSON.stringify(repo)}`);
  }
}

/** Rejects absolute paths, `..` segments and empty segments (cache-escape protection). */
export function validateFilename(file: string): void {
  const parts = file.split("/");
  if (file.startsWith("/") || file.includes("\\") || parts.some((p) => p === "" || p === "." || p === "..")) {
    throw new HfCacheError("InvalidFilename", `invalid repo file path ${JSON.stringify(file)}`);
  }
}

function typePrefix(repoType: RepoType): string {
  return repoType === "model" ? "" : `${repoType}s/`;
}

/** `{endpoint}/{datasets/|spaces/}{repo}/resolve/{revision}/{file}` (revision fully encoded, file per segment). */
export function resolveUrl(endpoint: string, repo: string, repoType: RepoType, revision: string, file: string): string {
  const path = file.split("/").map(encodeURIComponent).join("/");
  return `${endpoint}/${typePrefix(repoType)}${repo}/resolve/${encodeURIComponent(revision)}/${path}`;
}

export function revisionApiUrl(endpoint: string, repo: string, repoType: RepoType, revision: string): string {
  return `${endpoint}/api/${repoType}s/${repo}/revision/${encodeURIComponent(revision)}`;
}

/** Strips a weak-validator prefix and quotes, as huggingface_hub's `_normalize_etag`. */
export function normalizeEtag(etag: string | null): string | undefined {
  if (!etag) return undefined;
  return etag.replace(/^W\//, "").replace(/"/g, "");
}

export function authHeaders(token: string | undefined): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** Maps an error response to HfCacheError using the Hub's X-Error-Code header. */
export function httpError(res: Response, what: string): HfCacheError {
  const code = res.headers.get("x-error-code");
  const message = res.headers.get("x-error-message") ?? res.statusText;
  if (code === "EntryNotFound") return new HfCacheError("EntryNotFound", `${what}: ${message}`, { status: res.status });
  if (code === "RevisionNotFound") return new HfCacheError("RevisionNotFound", `${what}: ${message}`, { status: res.status });
  if (code === "RepoNotFound") return new HfCacheError("RepoNotFound", `${what}: ${message}`, { status: res.status });
  if (code === "GatedRepo") return new HfCacheError("GatedRepo", `${what}: ${message}`, { status: res.status });
  if (res.status === 401 || res.status === 403) return new HfCacheError("Unauthorized", `${what}: HTTP ${res.status} ${message}`, { status: res.status });
  if (res.status === 404) return new HfCacheError("RepoNotFound", `${what}: HTTP 404 ${message}`, { status: res.status });
  return new HfCacheError("HttpError", `${what}: HTTP ${res.status} ${message}`, { status: res.status });
}

/** True for failures of the transport itself (DNS, refused, reset) rather than HTTP errors. */
export function isNetworkError(e: unknown): boolean {
  return e instanceof TypeError || (e instanceof Error && /fetch failed|ECONN|ENOTFOUND|EAI_AGAIN|network/i.test(e.message));
}

/** Python `fnmatch.fnmatchcase` translated to a RegExp (`*` also crosses "/", like fnmatch). */
export function globToRegExp(pattern: string): RegExp {
  let re = "";
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i] as string;
    if (c === "*") re += ".*";
    else if (c === "?") re += ".";
    else if (c === "[") {
      const j = pattern.indexOf("]", i + 2);
      if (j === -1) {
        re += "\\[";
      } else {
        let body = pattern.slice(i + 1, j).replace(/\\/g, "\\\\");
        if (body.startsWith("!")) body = `^${body.slice(1)}`;
        re += `[${body}]`;
        i = j;
      }
    } else re += c.replace(/[.+^${}()|\\/]/g, "\\$&");
  }
  return new RegExp(`^${re}$`, "s");
}

/** huggingface_hub `filter_repo_objects`: allow (any match) then ignore (no match); "dir/" means "dir/*". */
export function filterFiles(files: readonly string[], allow?: readonly string[], ignore?: readonly string[]): string[] {
  const compile = (ps: readonly string[] | undefined) => ps?.map((p) => globToRegExp(p.endsWith("/") ? `${p}*` : p));
  const allowRe = compile(allow);
  const ignoreRe = compile(ignore);
  return files.filter((f) => (!allowRe || allowRe.some((r) => r.test(f))) && !(ignoreRe ?? []).some((r) => r.test(f)));
}

export interface RevisionInfo {
  commit: string;
  files: string[];
}

/** `GET /api/{type}s/{repo}/revision/{rev}` → pinned commit + full file list. */
export async function fetchRevisionInfo(repo: string, options: HubOptions & { token?: string | undefined }): Promise<RevisionInfo> {
  const repoType = options.repoType ?? "model";
  const revision = options.revision ?? "main";
  const url = revisionApiUrl(endpointOf(options), repo, repoType, revision);
  const res = await (options.fetch ?? globalThis.fetch)(url, {
    headers: authHeaders(options.token),
    ...(options.signal ? { signal: options.signal } : {}),
  });
  if (!res.ok) {
    await res.body?.cancel();
    throw httpError(res, `${repo}@${revision}`);
  }
  const json = (await res.json()) as { sha?: string; siblings?: Array<{ rfilename: string }> };
  if (!json.sha || !COMMIT_RE.test(json.sha)) throw new HfCacheError("HttpError", `${url}: response has no commit sha`);
  return { commit: json.sha, files: (json.siblings ?? []).map((s) => s.rfilename).sort() };
}

/** Runs `fn` over `items` with at most `limit` in flight; rejects on the first failure. */
export async function mapLimit<T, R>(items: readonly T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i] as T);
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, worker));
  return out;
}

/** Streams a response body, reporting progress; returns the chunks. */
export async function* readBody(res: Response, file: string, total: number | undefined, onProgress?: (e: ProgressEvent) => void): AsyncGenerator<Uint8Array> {
  if (!res.body) return;
  const reader = res.body.getReader();
  let loaded = 0;
  onProgress?.({ file, loaded, total });
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    loaded += value.byteLength;
    onProgress?.({ file, loaded, total });
    yield value;
  }
}
