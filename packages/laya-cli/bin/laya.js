#!/usr/bin/env node
// `laya` launcher. Published installs run dist/bin.js. In a workspace checkout
// without a build, re-run under the "source" export condition so the
// TypeScript sources of every @johnhenry package are used directly
// (Node >= 24 strips types; Bun runs TypeScript natively).
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const dist = new URL("../dist/bin.js", import.meta.url);
const hasSource = process.execArgv.some((a) => a === "--conditions=source" || a === "-C=source") ||
  (process.env.NODE_OPTIONS ?? "").includes("--conditions=source");
if (existsSync(fileURLToPath(dist)) && !process.env.LAYA_CLI_SOURCE) {
  await import(dist.href);
} else if (hasSource) {
  await import(new URL("../src/bin.ts", import.meta.url).href);
} else {
  const src = fileURLToPath(new URL("../src/bin.ts", import.meta.url));
  const r = spawnSync(process.execPath, ["--conditions=source", src, ...process.argv.slice(2)], { stdio: "inherit" });
  process.exit(r.status ?? 1);
}
