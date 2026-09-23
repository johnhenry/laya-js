/**
 * Bundle a browser example with `Bun.build` and copy its static files.
 *   bun scripts/build.ts [--root <exampleDir>] [--watch]
 * Entry: <root>/src/main.ts -> <root>/dist/main.js; <root>/public/* -> <root>/dist/.
 * Fails if the bundle pulls in native / Node-only modules.
 */
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, watch } from "node:fs";
import { join, resolve } from "node:path";

declare const Bun: {
  build(options: Record<string, unknown>): Promise<{ success: boolean; logs: unknown[]; outputs: { path: string; size: number }[] }>;
};

const args = process.argv.slice(2);
const root = resolve(args.includes("--root") ? args[args.indexOf("--root") + 1]! : join(import.meta.dirname, ".."));
const out = join(root, "dist");

/**
 * Static or literal imports that must never reach a browser bundle. (Guarded
 * runtime-only imports behind a non-literal specifier, like math-plus-safetensors'
 * `import(spec("node:fs/promises"))` for file paths, never execute in a browser. The one
 * allowed literal is backend-webgpu's `import("webgpu")` Dawn fallback, kept external: it
 * only runs when `navigator.gpu` is missing, and then fails into the "no WebGPU" path.)
 */
const FORBIDDEN = [
  /(?:from\s*|import\s*\(\s*|require\s*\(\s*)["'](?:node:[\w/]+|fs|fs\/promises|child_process|os|path|module|worker_threads|koffi|bun:ffi|@johnhenry\/backend-mlx)["']/,
  /\blibmlxc\.dylib\b/,
  /dawn\.node/,
];

async function build(): Promise<boolean> {
  const started = performance.now();
  rmSync(out, { recursive: true, force: true });
  mkdirSync(out, { recursive: true });
  const result = await Bun.build({
    entrypoints: [join(root, "src/main.ts")],
    outdir: out,
    target: "browser",
    format: "esm",
    splitting: true,
    minify: true,
    sourcemap: "linked",
    conditions: ["browser", "source"],
    define: { "process.env.NODE_ENV": '"production"' },
    // backend-webgpu's Node/Bun fallback does `import("webgpu")` (the Dawn addon) behind a
    // variable that Bun constant-folds; it never runs in a browser (navigator.gpu wins).
    // Native backends reachable only through `@johnhenry/laya`'s backend: "auto" path are
    // kept external for the same reason; the check below proves none is bundled.
    external: ["webgpu", "koffi", "@johnhenry/backend-mlx", "@nielspeter/mlx-ts", "@nielspeter/mlx-ts-darwin-arm64"],
  });
  if (!result.success) {
    for (const log of result.logs) console.error(log);
    return false;
  }
  if (existsSync(join(root, "public"))) cpSync(join(root, "public"), out, { recursive: true });
  let ok = true;
  let bytes = 0;
  for (const file of readdirSync(out)) {
    if (!file.endsWith(".js")) continue;
    const text = readFileSync(join(out, file), "utf8");
    bytes += text.length;
    for (const re of FORBIDDEN) {
      const m = re.exec(text);
      if (m) {
        console.error(`✖ ${file}: bundle references Node/native module (${m[0]}) — not browser-safe`);
        ok = false;
      }
    }
  }
  console.log(`${ok ? "✔" : "✖"} built ${out} (${(bytes / 1024).toFixed(0)} KiB JS) in ${(performance.now() - started).toFixed(0)} ms; native/Node-module check ${ok ? "passed" : "FAILED"}`);
  return ok;
}

const ok = await build();
if (args.includes("--watch")) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  for (const dir of ["src", "public"].map((d) => join(root, d)).filter(existsSync)) {
    watch(dir, { recursive: true }, () => {
      clearTimeout(timer);
      timer = setTimeout(() => void build(), 100);
    });
  }
  console.log("watching src/ and public/ …");
} else if (!ok) process.exit(1);
