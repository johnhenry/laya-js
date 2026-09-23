#!/usr/bin/env node
/**
 * Post-build step for every package: rewrites relative `.ts` import
 * specifiers in emitted declaration files to `.js`.
 *
 * The sources import siblings as "./x.ts" (Node >= 24 and Bun run them
 * directly), and `rewriteRelativeImportExtensions` fixes the emitted .js,
 * but tsc leaves the specifiers in .d.ts files untouched. TypeScript
 * consumers happen to resolve "./x.ts" to x.d.ts, but other type checkers
 * do not: Deno's `npm:` type checking fails with TS2307 on every one.
 *
 * Usage: node ../../scripts/rewrite-dts-extensions.mjs [distDir=dist]
 */
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const dir = process.argv[2] ?? "dist";
const RE = /((?:from|import)\s*\(?\s*["'])(\.{1,2}\/[^"']+?)\.(m|c)?ts(["'])/g;
let files = 0;
let edits = 0;
function walk(d) {
  for (const e of readdirSync(d, { withFileTypes: true })) {
    const p = join(d, e.name);
    if (e.isDirectory()) walk(p);
    else if (/\.d\.(m|c)?ts$/.test(e.name)) {
      const src = readFileSync(p, "utf8");
      let n = 0;
      const out = src.replace(RE, (_m, pre, spec, mc, q) => (n++, `${pre}${spec}.${mc ?? ""}js${q}`));
      if (n) {
        writeFileSync(p, out);
        files++;
        edits += n;
      }
    }
  }
}
walk(dir);
if (process.env.DEBUG_DTS) console.log(`rewrote ${edits} specifier(s) in ${files} declaration file(s) under ${dir}`);
