/**
 * The browser side of load(): io-browser.ts reading the tiny checkpoint from
 * a base URL (served over HTTP here, as a web app would), and a Bun browser
 * bundle of the package that must not contain node-only modules.
 */
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
import { makeTest } from "./harness.ts";
const test = makeTest(bun);
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, mkdtemp, readdir } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { fixturePath, loadJson } from "@johnhenry/laya-fixtures";
import { createCpuBackend } from "@johnhenry/backend-cpu";
import type { PredictResult, Questions } from "@johnhenry/laya-core";
import { readCheckpoint } from "../src/io-browser.ts";
import { createAgent } from "../src/index.ts";

test("io-browser: checkpoint from a base URL (Range reads for the weights) predicts like Python", async () => {
  const fx = await loadJson<{ state: any; questions: Questions; result: PredictResult }>("tiny", "predict.json");
  const ranges: string[] = [];
  const server = createServer(async (req, res) => {
    try {
      const body = await readFile(fixturePath("tiny", decodeURIComponent(req.url!.replace(/^\/models\/tiny\//, ""))));
      const range = req.headers.range?.match(/bytes=(\d+)-(\d*)/);
      if (range) {
        ranges.push(req.headers.range!);
        const start = Number(range[1]), end = range[2] ? Math.min(Number(range[2]), body.length - 1) : body.length - 1;
        res.writeHead(206, { "Content-Range": `bytes ${start}-${end}/${body.length}`, "Content-Length": end - start + 1, "Accept-Ranges": "bytes" });
        res.end(body.subarray(start, end + 1));
      } else {
        res.writeHead(200, { "Content-Length": body.length });
        res.end(body);
      }
    } catch {
      res.writeHead(404).end();
    }
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const port = (server.address() as { port: number }).port;
  try {
    const ckpt = await readCheckpoint(`http://127.0.0.1:${port}/models/tiny`);
    assert.equal(ckpt.location, `http://127.0.0.1:${port}/models/tiny/`);
    const weights = await ckpt.weights();
    const agent = await createAgent({ backend: createCpuBackend(), encoderConfig: ckpt.encoderConfig, agentConfig: ckpt.agentConfig, weights: weights.get, tokenizer: ckpt.tokenizer, batchSize: 2, warn: () => {} });
    assert.deepEqual(weights.remaining(), ["temperature"]);
    assert.deepEqual(await agent.predict(fx.state, fx.questions), fx.result);
    assert.ok(ranges.length >= 1, "weights were read with Range requests");
    agent.dispose();
    await assert.rejects(readCheckpoint(`http://127.0.0.1:${port}/models/nope`), /Not a complete Laya checkpoint/);
  } finally {
    server.close();
  }
});

const bunBin = spawnSync("bun", ["--version"], { encoding: "utf8" }).status === 0;
test("browser bundle (browser condition) pulls in io-browser and no node-only modules", async () => {
  const out = await mkdtemp(join(tmpdir(), "laya-bundle-"));
  const entry = fileURLToPath(new URL("../src/index.ts", import.meta.url));
  const r = spawnSync("bun", ["build", entry, "--target=browser", "--conditions=browser", "--conditions=source", "--outdir", out, "--splitting"], {
    encoding: "utf8",
    cwd: fileURLToPath(new URL("../../../", import.meta.url)), // bundle comments carry repo-relative paths
  });
  assert.equal(r.status, 0, r.stderr);
  let code = "";
  for (const f of await readdir(out)) code += await readFile(join(out, f), "utf8");
  assert.match(code, /packages\/laya\/src\/io-browser\.ts/);
  assert.match(code, /packages\/hf-cache\/src\/browser\.ts/);
  for (const bad of ["io-node.ts", "backend-mlx", "koffi", "laya-core/src/node.ts", "hf-cache/src/node.ts"]) assert.ok(!code.includes(bad), `bundle contains ${bad}`);
  assert.ok(!/from\s*["']node:/.test(code), "static node: import in the browser bundle");
}, { skip: bunBin ? false : "bun not on PATH" });
