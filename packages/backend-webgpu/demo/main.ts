/**
 * Browser smoke page: conformance cases + GEMM benchmark.
 * Build: npm run demo:build -w @johnhenry/backend-webgpu
 * Serve: from packages/ (e.g. `python3 -m http.server -d packages 8000`) and open
 *        http://localhost:8000/backend-webgpu/demo/
 */
import { runConformance, type OpCase, type TestApi } from "@johnhenry/tensor-backend/conformance";
import { createWebGpuBackend, type WebGpuBackend } from "../src/index.ts";

const logEl = document.getElementById("log")!;
const log = (msg: string, cls?: string) => {
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = msg;
  logEl.append(line);
};

/** Minimal describe/it runner: collects tests, then runs them in order. */
function makeRunner() {
  const tests: { name: string; fn: () => unknown }[] = [];
  const prefix: string[] = [];
  const api: TestApi = {
    describe: (name, fn) => {
      prefix.push(name);
      fn();
      prefix.pop();
    },
    it: (name, fn) => {
      tests.push({ name: [...prefix, name].join(" › "), fn });
    },
  };
  return { api, tests };
}

async function conformance(): Promise<{ pass: number; fail: number }> {
  const cases = fetch(new URL("../../tensor-backend/fixtures/ops.json", location.href)).then((r) => {
    if (!r.ok) throw new Error(`fetch ops.json: ${r.status}`);
    return r.json() as Promise<OpCase[]>;
  });
  const { api, tests } = makeRunner();
  const made: WebGpuBackend[] = [];
  let first = true;
  runConformance(async () => {
    const b = await createWebGpuBackend();
    made.push(b);
    if (first) {
      first = false;
      const { limits: _l, ...info } = b.adapterInfo;
      log(`adapter: ${JSON.stringify(info)}`);
      log(`f16 storage: ${b.supports("f16") ? "yes (shader-f16)" : "no (f32 only)"}`);
    }
    return b;
  }, cases, api);
  let pass = 0, fail = 0;
  for (const t of tests) {
    const t0 = performance.now();
    try {
      await t.fn();
      pass++;
      log(`PASS ${t.name} (${(performance.now() - t0).toFixed(0)} ms)`, "pass");
    } catch (e) {
      fail++;
      log(`FAIL ${t.name}\n${(e as Error).message}`, "fail");
    }
  }
  for (const b of made) b.destroy();
  return { pass, fail };
}

async function gemmBench(): Promise<void> {
  const b = await createWebGpuBackend();
  const M = 2048, K = 1024, N = 3072;
  const rnd = (n: number, s = 1) => Float32Array.from({ length: n }, () => (Math.random() * 2 - 1) * s);
  const x = await b.fromHost({ dtype: "f32", shape: [M, K], data: rnd(M * K) });
  const w = await b.fromHost({ dtype: "f32", shape: [N, K], data: rnd(N * K, 0.03) });
  for (const dt of b.supports("f16") ? (["f32", "f16"] as const) : (["f32"] as const)) {
    const xx = dt === "f32" ? x : b.cast(x, dt), ww = dt === "f32" ? w : b.cast(w, dt);
    for (let i = 0; i < 3; i++) b.dispose(b.linear(xx, ww));
    await b.sync();
    const iters = 20, t0 = performance.now();
    for (let i = 0; i < iters; i++) b.dispose(b.linear(xx, ww));
    await b.sync();
    const ms = (performance.now() - t0) / iters;
    log(`GEMM linear [${M},${K}]·[${N},${K}]ᵀ ${dt}: ${ms.toFixed(2)} ms, ${((2 * M * N * K) / ms / 1e6).toFixed(0)} GFLOP/s`);
  }
  b.destroy();
}

async function main() {
  logEl.textContent = "";
  const nav = navigator as Navigator & { gpu?: unknown };
  if (!nav.gpu) {
    log("SKIP: navigator.gpu is not available in this browser", "fail");
    (window as unknown as { __result: unknown }).__result = { skipped: true };
    return;
  }
  try {
    const r = await conformance();
    log(`conformance: ${r.pass} passed, ${r.fail} failed`, r.fail ? "fail" : "pass");
    await gemmBench();
    (window as unknown as { __result: unknown }).__result = r;
  } catch (e) {
    log(`ERROR ${(e as Error).stack ?? e}`, "fail");
    (window as unknown as { __result: unknown }).__result = { error: String(e) };
  }
}

document.getElementById("rerun")!.addEventListener("click", () => void main());
void main();
