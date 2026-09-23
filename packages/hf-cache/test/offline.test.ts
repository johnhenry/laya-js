/**
 * Offline resolution against the REAL local cache, which Python's
 * `hf download` populated for the three Laya repos. Never touches the
 * network (offline: true) and never reads the large model files.
 */
import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const test = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import assert from "node:assert/strict";
import { existsSync, lstatSync, readFileSync, readlinkSync, realpathSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { HfCacheError, fetchFile, repoFolderName, resolveFile, snapshot, tryToLoadFromCache } from "../src/index.ts";
import { LAYA_REPOS, REAL_CACHE, pyTryToLoad, repoCached, pythonSkip } from "./helpers.ts";

for (const repo of LAYA_REPOS) {
  const skip = repoCached(repo) ? false : `${repo} is not in ${REAL_CACHE}`;
  const storage = join(REAL_CACHE, repoFolderName(repo));

  test(`${repo}: offline resolveFile follows refs/main to the snapshot symlink`, async () => {
    const commit = readFileSync(join(storage, "refs", "main"), "utf8").trim();
    const path = await resolveFile(repo, "mlx_config.json", { offline: true });
    assert.equal(path, join(storage, "snapshots", commit, "mlx_config.json"));
    assert.ok(lstatSync(path).isSymbolicLink());
    assert.equal(dirname(resolve(dirname(path), readlinkSync(path))), join(storage, "blobs"));
    JSON.parse(readFileSync(path, "utf8")); // real content
    // nested paths and the big weights resolve without reading them
    const weights = await resolveFile(repo, "model.safetensors", { offline: true });
    assert.ok(statSync(weights).size > 100_000_000);
    assert.ok(existsSync(await resolveFile(repo, "tokenizer/tokenizer.json", { offline: true })));
    // pinning the commit explicitly gives the same answer without refs
    assert.equal(await resolveFile(repo, "mlx_config.json", { offline: true, revision: commit }), path);
  }, { skip });

  test(`${repo}: HF_HUB_OFFLINE=1 is honoured, misses are OfflineCacheMiss`, async () => {
    const prev = process.env.HF_HUB_OFFLINE;
    process.env.HF_HUB_OFFLINE = "1";
    const noNetwork = (() => { throw new Error("network used"); }) as unknown as typeof fetch;
    try {
      assert.ok(existsSync(await resolveFile(repo, "mlx_config.json", { fetch: noNetwork })));
      await assert.rejects(resolveFile(repo, "does-not-exist.bin", { fetch: noNetwork }), (e: unknown) => e instanceof HfCacheError && e.code === "OfflineCacheMiss");
      await assert.rejects(resolveFile(repo, "mlx_config.json", { fetch: noNetwork, revision: "no-such-branch" }), (e: unknown) => e instanceof HfCacheError && e.code === "OfflineCacheMiss");
    } finally {
      if (prev === undefined) delete process.env.HF_HUB_OFFLINE;
      else process.env.HF_HUB_OFFLINE = prev;
    }
  }, { skip });

  test(`${repo}: offline snapshot lists cached files, filtered by patterns`, async () => {
    const snap = await snapshot(repo, { offline: true, allowPatterns: ["*.json"], ignorePatterns: ["tokenizer/*"] });
    assert.ok(snap.files.includes("mlx_config.json"));
    assert.ok(snap.files.includes("encoder/config.json"));
    assert.ok(!snap.files.some((f) => f.startsWith("tokenizer/") || f.endsWith(".safetensors")));
    for (const f of snap.files) assert.ok(existsSync(join(snap.dir, f)), f);
    const exact = await snapshot(repo, { offline: true, files: ["mlx_config.json", "model.safetensors"] });
    assert.deepEqual(exact.files, ["mlx_config.json", "model.safetensors"]);
    assert.equal(exact.commit, snap.commit);
    const folder = await snapshot(repo, { offline: true, allowPatterns: ["tokenizer/"] });
    assert.ok(folder.files.length >= 1 && folder.files.every((f) => f.startsWith("tokenizer/")));
  }, { skip });

  test(`${repo}: fetchFile gives a lazy Blob or an ArrayBuffer`, async () => {
    const blob = await fetchFile(repo, "mlx_config.json", { offline: true });
    const buf = await fetchFile(repo, "mlx_config.json", { offline: true, as: "arraybuffer" });
    assert.equal(blob.size, buf.byteLength);
    assert.deepEqual(new Uint8Array(await blob.arrayBuffer()), new Uint8Array(buf));
    const big = await fetchFile(repo, "model.safetensors", { offline: true });
    assert.ok(big.size > 100_000_000); // not read: openAsBlob is disk-backed
    const head = new Uint8Array(await big.slice(0, 8).arrayBuffer());
    assert.ok(new DataView(head.buffer).getBigUint64(0, true) > 0n);
  }, { skip });

  test(`${repo}: agrees with huggingface_hub.try_to_load_from_cache`, async () => {
    for (const file of ["mlx_config.json", "model.safetensors", "tokenizer/tokenizer.json", "nope.txt"]) {
      const ours = await tryToLoadFromCache(repo, file, { cacheDir: REAL_CACHE });
      const theirs = pyTryToLoad(repo, file, REAL_CACHE);
      assert.equal(ours ?? null, theirs, file);
      if (ours) assert.equal(realpathSync(ours), realpathSync(theirs as string));
    }
  }, { skip: skip || pythonSkip });
}

test("invalid file paths never escape the cache", async () => {
  for (const bad of ["../x", "/etc/passwd", "a//b", "a/./b", "a\\b"]) {
    await assert.rejects(resolveFile("aac6fef/laya-mlx", bad, { offline: true }), (e: unknown) => e instanceof HfCacheError && e.code === "InvalidFilename", bad);
  }
  await assert.rejects(resolveFile("../evil", "x", { offline: true }), HfCacheError);
});

