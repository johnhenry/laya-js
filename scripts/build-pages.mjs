#!/usr/bin/env node
/**
 * Assemble the GitHub Pages site (https://johnhenry.github.io/laya-js/) from
 * the browser examples:
 *
 *   <out>/index.html      landing page (scripts/pages/index.html)
 *   <out>/playground/     examples/web-playground/dist
 *   <out>/snake/          examples/snake-web/dist
 *   <out>/flappy/         examples/flappy-web/dist
 *   <out>/checkers/       examples/checkers-web/dist
 *   <out>/tetris/         examples/tetris-web/dist
 *   <out>/dino/           examples/dino-web/dist
 *
 *   node scripts/build-pages.mjs [--out _site] [--no-build]
 *
 * Builds both examples first (bun; `--no-build` reuses existing dist/). The
 * apps reference their assets relatively (./main.js, ./styles.css, relative
 * chunk imports), so they work under any base path such as /laya-js/playground/
 * without a rebuild. The script fails if an app's HTML uses a root-absolute
 * URL, which would break under the /laya-js/ project-site prefix.
 */
import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname;
const args = process.argv.slice(2);
const out = join(ROOT, args.includes("--out") ? args[args.indexOf("--out") + 1] : "_site");

const APPS = [
  { workspace: "@johnhenry/example-web-playground", dir: "examples/web-playground", path: "playground" },
  { workspace: "@johnhenry/example-snake-web", dir: "examples/snake-web", path: "snake" },
  { workspace: "@johnhenry/example-flappy-web", dir: "examples/flappy-web", path: "flappy" },
  { workspace: "@johnhenry/example-checkers-web", dir: "examples/checkers-web", path: "checkers" },
  { workspace: "@johnhenry/example-tetris-web", dir: "examples/tetris-web", path: "tetris" },
  { workspace: "@johnhenry/example-dino-web", dir: "examples/dino-web", path: "dino" },
];

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

for (const app of APPS) {
  if (!args.includes("--no-build")) execFileSync("npm", ["run", "build", "-w", app.workspace], { cwd: ROOT, stdio: "inherit" });
  const dist = join(ROOT, app.dir, "dist");
  if (!existsSync(join(dist, "index.html"))) throw new Error(`${app.dir}/dist/index.html missing -- build the example first`);
  for (const file of readdirSync(dist).filter((f) => f.endsWith(".html") || f.endsWith(".css"))) {
    const text = readFileSync(join(dist, file), "utf8");
    const abs = /(?:src|href)\s*=\s*["']\/(?!\/)|url\(\s*["']?\/(?!\/)/.exec(text);
    if (abs) throw new Error(`${app.dir}/dist/${file}: root-absolute URL (${abs[0]}) breaks under the /laya-js/ base path`);
  }
  cpSync(dist, join(out, app.path), { recursive: true });
  console.log(`✔ ${app.dir}/dist → ${join(out, app.path)}`);
}

cpSync(join(ROOT, "scripts/pages/index.html"), join(out, "index.html"));
// No Jekyll processing on GitHub Pages (keeps files like _chunk.js servable).
writeFileSync(join(out, ".nojekyll"), "");
console.log(`✔ site assembled in ${out}`);
