#!/usr/bin/env node
/**
 * Generates/refreshes every JSR-publishable package's jsr.json from its
 * package.json — one source of truth instead of two hand-maintained,
 * driftable manifests. Adapted from math-plus's scripts/sync-jsr-configs.mjs.
 *
 * - JSR publishes the TypeScript SOURCE, so each npm `exports` entry is
 *   translated to its `source` condition (the first condition in every
 *   package here). Conditional (`browser`) entries collapse to the
 *   top-level `source` target: JSR exports cannot be conditional, and the
 *   browser build stays reachable through its own subpath (hf-cache's
 *   `./browser`).
 * - Bare-specifier dependencies become an import map:
 *     - a workspace sibling that is itself on JSR  → `jsr:@johnhenry/<name>@^x`
 *     - `@johnhenry/math-plus-*` (published to JSR by math-plus) → `jsr:`
 *     - anything else (incl. siblings that are npm-only) → `npm:<name>@<range>`
 *   Deno/JSR accept ONE comparator per specifier (`^1.2.3`, `~1.2`, `1.2.3`,
 *   `1.x`, `*`); an npm union like `^0.0.0 || ^0.1.0` is rejected at publish
 *   ("Invalid package specifier ... Unexpected character", math-plus release
 *   run 35967422091), as are `>=a <b` ranges. `jsrRange` keeps the highest
 *   alternative of a `||` union and refuses anything else it cannot express.
 * - `publish.include` is `src` plus every non-build entry of the npm
 *   `files` list (fixtures, NOTICE, CHANGELOG.md), README.md and LICENSE.
 *
 * Packages NOT on JSR are listed in JSR_EXCLUDED with the reason; the root
 * manifest-drift test fails if a package is in neither list.
 *
 * Run after adding/bumping a dependency (`npm run sync:jsr`); the release
 * workflow runs it before `jsr publish`.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname;

const PACKAGE_DIRS = [
  "packages/backend-cpu",
  "packages/backend-mlx",
  "packages/backend-webgpu",
  "packages/hf-cache",
  "packages/langdetect-lite",
  "packages/laya-core",
  "packages/laya-presets",
  "packages/laya-router",
  "packages/modernbert",
  "packages/pyjson",
  "packages/tensor-backend",
];

/** Packages deliberately kept off JSR, with the reason (read by test/manifest-drift.test.ts). */
export const JSR_EXCLUDED = {
  "packages/backend-mlx-darwin-arm64": "npm platform package of native binaries (os/cpu-gated optionalDependency)",
  "packages/laya": "runtime split via package.json `imports` (#io browser/node), which JSR cannot express",
  "packages/laya-cli": "Node/Bun command-line tool (bin) on top of @johnhenry/laya",
  "packages/laya-fixtures": "private test fixtures",
};

const EXTERNAL_JSR_PREFIXES = ["@johnhenry/math-plus-"];

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function jsrNames() {
  return new Set(PACKAGE_DIRS.map((d) => readJson(join(ROOT, d, "package.json")).name));
}

function sourceTarget(entry) {
  if (typeof entry === "string") return entry;
  if (entry && typeof entry === "object") {
    if (typeof entry.source === "string") return entry.source;
    for (const v of Object.values(entry)) {
      const t = sourceTarget(v);
      if (t) return t;
    }
  }
  return undefined;
}

function buildExports(pkg) {
  const out = {};
  for (const [subpath, entry] of Object.entries(pkg.exports ?? {})) {
    if (subpath.includes("*") || subpath === "./package.json") continue;
    const target = sourceTarget(entry);
    if (!target || !target.endsWith(".ts")) throw new Error(`${pkg.name}: export ${subpath} has no TypeScript source target`);
    out[subpath] = target;
  }
  return out;
}

/**
 * Package-internal `#` imports (package.json `imports`, e.g. backend-webgpu's
 * `#dawn`) map to their top-level `source` target. Without this, Deno/JSR
 * falls back to package.json and resolves the `default` condition — a
 * `dist/` file JSR does not publish (the release dry-run failed with
 * `Module not found ".../dist/dawn-node.js"`). The top-level `source` is
 * the Node/Bun/Deno implementation; a nested `browser.source` stays an npm
 * bundler concern (JSR import maps cannot be conditional).
 */
export function internalImports(pkg) {
  const out = {};
  for (const [key, entry] of Object.entries(pkg.imports ?? {}).sort(([a], [b]) => a.localeCompare(b))) {
    const target = typeof entry === "string" ? entry : typeof entry?.source === "string" ? entry.source : sourceTarget(entry);
    if (!target || !target.endsWith(".ts")) throw new Error(`${pkg.name}: import ${key} has no TypeScript source target`);
    out[key] = target;
  }
  return out;
}

/** A version requirement Deno/JSR accept inside a `jsr:`/`npm:` specifier: one ^/~/exact/partial comparator, or `*`. */
export const SPECIFIER_RANGE = /^(?:\*|[\^~]?\d+(?:\.(?:\d+|x|\*)){0,2}(?:-[0-9A-Za-z.-]+)?)$/;

function rangeBase(r) {
  return r
    .replace(/^[\^~]/, "")
    .split("-")[0]
    .split(".")
    .map((p) => (p === "x" || p === "*" ? Number.POSITIVE_INFINITY : Number(p)));
}

/**
 * The package.json range as a Deno/JSR specifier range. A `||` union keeps
 * its highest alternative (`^0.0.0 || ^0.1.0 || ^0.2.0` → `^0.2.0`): the
 * lower alternatives only exist for npm dedupe across pre-1.0 minors, and
 * JSR resolves the newest match anyway.
 */
export function jsrRange(range) {
  const alts = String(range).split("||").map((r) => r.trim());
  for (const a of alts) {
    if (!SPECIFIER_RANGE.test(a)) throw new Error(`range "${range}": "${a}" is not a single ^/~/exact comparator, which a jsr:/npm: specifier requires`);
  }
  return alts.reduce((best, a) => {
    const x = rangeBase(a);
    const y = rangeBase(best);
    for (let i = 0; i < 3; i++) if ((x[i] ?? 0) !== (y[i] ?? 0)) return (x[i] ?? 0) > (y[i] ?? 0) ? a : best;
    return best;
  });
}

function buildImports(pkg, onJsr) {
  const deps = { ...pkg.dependencies, ...pkg.optionalDependencies, ...pkg.peerDependencies };
  const imports = internalImports(pkg);
  for (const [name, range] of Object.entries(deps).sort(([a], [b]) => a.localeCompare(b))) {
    const jsr = onJsr.has(name) || EXTERNAL_JSR_PREFIXES.some((p) => name.startsWith(p));
    imports[name] = `${jsr ? "jsr" : "npm"}:${name}@${jsrRange(range)}`;
  }
  return imports;
}

function buildInclude(dir, pkg) {
  const include = ["src"];
  for (const f of pkg.files ?? []) if (f !== "dist" && f !== "bin" && !include.includes(f)) include.push(f);
  for (const f of ["README.md", "LICENSE"]) if (existsSync(join(ROOT, dir, f)) && !include.includes(f)) include.push(f);
  return include;
}

export function jsrConfigFor(dir, onJsr = jsrNames()) {
  const pkg = readJson(join(ROOT, dir, "package.json"));
  const exports = buildExports(pkg);
  const imports = buildImports(pkg, onJsr);
  return {
    name: pkg.name,
    version: pkg.version,
    // JSR hard-requires a license (math-plus: error[missing-license] on the first real publish).
    license: pkg.license,
    exports: Object.keys(exports).length === 1 && exports["."] ? exports["."] : exports,
    ...(Object.keys(imports).length > 0 ? { imports } : {}),
    publish: { include: buildInclude(dir, pkg) },
  };
}

export { PACKAGE_DIRS };

if (import.meta.url === `file://${process.argv[1]}`) {
  const check = process.argv.includes("--check");
  const onJsr = jsrNames();
  let stale = 0;
  for (const dir of PACKAGE_DIRS) {
    const text = `${JSON.stringify(jsrConfigFor(dir, onJsr), null, 2)}\n`;
    const path = join(ROOT, dir, "jsr.json");
    if (check) {
      if (!existsSync(path) || readFileSync(path, "utf8") !== text) {
        console.error(`stale: ${dir}/jsr.json (run npm run sync:jsr)`);
        stale++;
      }
    } else {
      writeFileSync(path, text);
      console.log(`wrote ${dir}/jsr.json`);
    }
  }
  if (stale) process.exit(1);
}
