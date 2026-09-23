/**
 * Real Hugging Face Hub, small files only, into a TEMP cacheDir (never the
 * user's cache, never the large weights). Skips when the Hub is unreachable.
 * The blob names and bytes must equal what Python's `hf download` stored in
 * the real cache for the same files.
 */
import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const test = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import assert from "node:assert/strict";
import { readFileSync, readlinkSync, realpathSync } from "node:fs";
import { basename, join } from "node:path";
import { resolveFile, snapshot } from "../src/index.ts";
import { REAL_CACHE, hubReachable, repoCached, tempDir } from "./helpers.ts";

const REPO = "aac6fef/laya-mlx";
const online = await hubReachable();
const skip = online ? (repoCached(REPO) ? false : `${REPO} not cached locally (needed for comparison)`) : "huggingface.co unreachable";

test("resolveFile downloads mlx_config.json exactly like huggingface_hub did", async () => {
  const { dir, cleanup } = await tempDir();
  try {
    const opts = { cacheDir: dir, token: "" };
    const ours = await resolveFile(REPO, "mlx_config.json", opts);
    const theirs = await resolveFile(REPO, "mlx_config.json", { offline: true, cacheDir: REAL_CACHE });
    // same commit directory name, same blob (etag) name, same bytes
    assert.equal(basename(join(ours, "..")), basename(join(theirs, "..")));
    assert.equal(basename(readlinkSync(ours)), basename(realpathSync(theirs)));
    assert.deepEqual(readFileSync(ours), readFileSync(theirs));
    const nested = await resolveFile(REPO, "encoder/config.json", opts);
    assert.ok(readlinkSync(nested).startsWith("../../../blobs/"));
    assert.deepEqual(readFileSync(nested), readFileSync(await resolveFile(REPO, "encoder/config.json", { offline: true, cacheDir: REAL_CACHE })));
  } finally {
    await cleanup();
  }
}, { skip });

test("snapshot of the small JSON files pins main and reuses blobs on a second call", async () => {
  const { dir, cleanup } = await tempDir();
  try {
    const snap = await snapshot(REPO, { cacheDir: dir, token: "", allowPatterns: ["*.json"], ignorePatterns: ["tokenizer/tokenizer.json"] });
    const real = readFileSync(join(REAL_CACHE, "models--aac6fef--laya-mlx", "refs", "main"), "utf8").trim();
    assert.equal(snap.commit, real);
    assert.ok(snap.files.includes("mlx_config.json") && snap.files.includes("encoder/config.json"));
    assert.ok(!snap.files.includes("model.safetensors"));
    let got = 0;
    const again = await snapshot(REPO, { cacheDir: dir, token: "", files: snap.files, revision: snap.commit, onProgress: () => got++ });
    assert.deepEqual(again.files, snap.files);
    assert.equal(got, 0, "commit-pinned snapshot of cached files makes no requests");
  } finally {
    await cleanup();
  }
}, { skip });
