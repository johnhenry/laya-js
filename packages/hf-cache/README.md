# @johnhenry/hf-cache

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Fhf-cache.svg)](https://www.npmjs.com/package/@johnhenry/hf-cache)

Resolve and cache Hugging Face Hub files from JS.

- **Node/Bun/Deno:** exactly the `huggingface_hub` cache layout
  (`~/.cache/huggingface/hub/models--{org}--{name}/{blobs,snapshots,refs}`).
  Files Python's `hf download` / `hf_hub_download` / `snapshot_download`
  already fetched are reused offline, and Python reuses what this package
  downloads.
- **Browsers:** the same functions backed by the Cache API (picked
  automatically through the `browser` export condition, or import
  `@johnhenry/hf-cache/browser`).

No dependencies: plain `fetch` against `https://huggingface.co/{repo}/resolve/{revision}/{file}`
and `/api/models/{repo}/revision/{revision}`.

## Install

```bash
npm install @johnhenry/hf-cache
bun add @johnhenry/hf-cache
deno add jsr:@johnhenry/hf-cache
```

Node ≥ 24, Bun ≥ 1.2 and Deno use the `huggingface_hub` disk cache; browsers get the Cache API build through the `browser` export condition. No dependencies.

```js
import { resolveFile, snapshot, fetchFile } from "@johnhenry/hf-cache";
import { openSafetensors } from "@johnhenry/math-plus-safetensors";

// Node: absolute path inside the cache (downloads if missing)
const path = await resolveFile("aac6fef/laya-mlx", "model.safetensors", {
  onProgress: ({ loaded, total }) => console.log(loaded, total),
});
const model = await openSafetensors(path);           // header only; tensors on demand

// Several files, one pinned commit
const { dir, files, commit } = await snapshot("aac6fef/laya-mlx", {
  allowPatterns: ["*.json", "tokenizer/"],
});

// Content: a lazy disk-backed Blob in Node (fs.openAsBlob), a cached Blob in browsers
const cfg = JSON.parse(await (await fetchFile("aac6fef/laya-mlx", "mlx_config.json")).text());
```

## API

| Export | Node/Bun | Browser |
|---|---|---|
| `resolveFile(repo, file, opts?)` → `Promise<string>` | absolute snapshot path | cache key URL `{endpoint}/{repo}/resolve/{commit}/{file}` |
| `snapshot(repo, opts?)` → `{ dir, files, commit }` | `dir` = `…/snapshots/{commit}` | `dir` = `{endpoint}/{repo}/resolve/{commit}/` |
| `fetchFile(repo, file, opts?)` → `Blob` (or `ArrayBuffer` with `{ as: "arraybuffer" }`) | `fs.openAsBlob` (lazy) | from the Cache API |
| `tryToLoadFromCache(repo, file, opts?)` → path \| `null` (cached 404) \| `undefined` | ✓ (`huggingface_hub.try_to_load_from_cache`) | — |
| `hubCacheDir(opts?)`, `hfHome()`, `repoFolderName(repo, type?)` | ✓ | — |
| `HfCacheError` (`code`: `EntryNotFound`, `RepoNotFound`, `RevisionNotFound`, `GatedRepo`, `Unauthorized`, `OfflineCacheMiss`, `InvalidFilename`, `HttpError`), `filterFiles`, `globToRegExp` | ✓ | ✓ |

Options (`HubOptions`): `revision` (default `"main"`; branch, tag or commit),
`repoType` (`"model"` | `"dataset"` | `"space"`), `token`, `cacheDir`
(Node: hub cache dir; browser: Cache API cache name, default `"hf-cache"`),
`endpoint`, `fetch`, `onProgress({ file, loaded, total })`, `offline`,
`signal`. `snapshot` adds `files` (exact list) or `allowPatterns` /
`ignorePatterns` (fnmatch, as huggingface_hub; `"dir/"` = whole folder) and
`concurrency` (default 4). Browser functions also take `caches` (inject a
CacheStorage, e.g. in tests or workers).

## Behaviour (Node/Bun)

- **Cache dir:** `opts.cacheDir`, else `$HF_HUB_CACHE`, else
  `$HUGGINGFACE_HUB_CACHE`, else `$HF_HOME/hub`, else
  `$XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`.
- **Offline:** `opts.offline` or `$HF_HUB_OFFLINE` (`1`/`ON`/`YES`/`TRUE`):
  `refs/{revision}` → `snapshots/{commit}/{file}`; misses throw
  `OfflineCacheMiss`, cached 404s (`.no_exist/`) throw `EntryNotFound`.
  When the Hub is unreachable (transport error), online calls fall back to
  the cache the same way. `snapshot` offline lists the files present (using
  huggingface_hub 1.x's `trees/{commit}.json` listing when available).
- **Online resolution** mirrors `hf_hub_download`: a HEAD on the resolve URL
  (following same-origin redirects, stopping at the CDN redirect) yields
  `X-Repo-Commit` and the blob id (`X-Linked-Etag` = sha256 for LFS/Xet
  files, else the git blob sha1) → `refs/{revision}` is updated, an existing
  blob is reused (a moved branch costs HEAD requests only), otherwise the
  file is streamed to `blobs/{etag}.{random}.incomplete`, size-checked and
  renamed into place, and `snapshots/{commit}/{file}` becomes a relative
  symlink to it (copied where symlinks are not permitted). A commit-hash
  revision whose file is cached makes **no** request at all.
- **Tokens:** `opts.token`, else `$HF_TOKEN`, `$HUGGING_FACE_HUB_TOKEN`,
  else the token file `hf auth login` writes (`$HF_TOKEN_PATH` or
  `$HF_HOME/token`). Pass `token: ""` to send none. Redirects are followed
  manually and the token is never sent to another origin (the CDN).
- **404 on a file** with a known commit writes `.no_exist/{commit}/{file}`
  (as huggingface_hub does).

## Limitations

- No inter-process locking (huggingface_hub's `.locks/`): concurrent
  downloads of the same blob both download; atomic renames keep the cache
  consistent, but bandwidth is wasted.
- Downloads are not resumable (a failed download is discarded) and are not
  hash-verified beyond the byte count.
- Writes the classic per-repo `blobs/` layout, not huggingface_hub 1.x's
  cache-wide Xet shared-blob store (`{cache}/blobs/xx/{xet_hash}`); both
  layouts are read transparently (symlinks are followed). Does not write
  `trees/{commit}.json`.
- `snapshot` online asks `/api/{type}s/{repo}/revision/{rev}` for the file
  list, so it needs one API call even when everything is cached (except a
  commit-hash revision with an explicit `files` list).
- Browser store: Cache API only (no OPFS); a downloaded file is buffered
  into a Blob before being stored (large files need the memory, or the
  browser's disk-backed Blob support); an offline browser `snapshot` needs
  an explicit `files` list (Cache API has no listing). No `.no_exist`
  caching in the browser.
- Buckets and the Hub's `local_dir` download mode are not supported.

## Tests

`npm test -w @johnhenry/hf-cache` and `npm run test:bun -w @johnhenry/hf-cache`.

- `offline.test.ts` resolves `aac6fef/laya-mlx`, `laya-multilingual-mlx`
  and `laya-typed-decisions-mlx` from the real local cache (offline; skipped
  per repo when absent) and cross-checks `huggingface_hub.try_to_load_from_cache`
  (skipped without a Python that has `huggingface_hub`; `$MATH_PLUS_ORACLE_PYTHON`
  or `python3`).
- `layout.test.ts` drives a mocked Hub into a temp cache: exact files and
  symlinks, refs updates, blob reuse, `.no_exist`, failed/short downloads,
  token handling, network fallback, snapshot filters, and Python reading
  what we wrote.
- `network.test.ts` downloads small JSON files from huggingface.co into a
  temp cache and checks the commit, blob names and bytes equal what
  `hf download` stored (skipped when the Hub is unreachable).
- `browser.test.ts` runs the Cache API store against an in-memory
  CacheStorage.

## Family

Part of **[laya-js](https://github.com/johnhenry/laya-js#readme)**, Laya typed decisions in JavaScript on MLX, WebGPU and CPU — see its [package map](https://github.com/johnhenry/laya-js#which-package-do-i-want) and [results](https://github.com/johnhenry/laya-js#results).

- [`@johnhenry/laya`](https://github.com/johnhenry/laya-js/tree/main/packages/laya)'s `load()` resolves checkpoints through it (disk cache on Node/Bun, Cache API in browsers).
- `resolveFile` returns a path (Node) and `fetchFile` a Blob; [`@johnhenry/math-plus-safetensors`](https://github.com/johnhenry/math-plus/tree/main/packages/safetensors)'s `openSafetensors` reads either lazily (header first, tensors on demand).

## License

MIT.
