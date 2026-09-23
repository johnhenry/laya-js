/**
 * Locating libmlxc.dylib (mlx-c, Apple's C API over MLX).
 *
 * Resolution order (first existing file wins):
 *   1. the explicit `libPath` option
 *   2. `$LAYA_MLXC_PATH` — a libmlxc.dylib file, or a directory containing
 *      one (`$LAYA_MLXC_LIB` is accepted as a legacy alias). A set variable
 *      that points nowhere is an error, never a silent fallback.
 *   3. the platform package `@johnhenry/backend-mlx-darwin-arm64`
 *      (an optionalDependency: npm installs it only on darwin/arm64). It
 *      ships `lib/{libmlxc,libmlx,libjaccl}.dylib` + `lib/mlx.metallib`,
 *      built by `scripts/build-mlxc.sh` (mlx-c v0.6+ against the MLX 0.32.2
 *      wheel — the same MLX the Python reference uses)
 *   4. a local build in this package: `prebuilds/darwin-arm64/`
 *      (`npm run build:mlxc -w @johnhenry/backend-mlx`)
 *   5. `@nielspeter/mlx-ts-darwin-arm64` (npm; MLX 0.32.1, older mlx-c ABI)
 *   6. Homebrew (`brew install mlx-c`)
 *
 * Each bundle is relocatable: libmlxc sits beside libmlx.dylib and
 * mlx.metallib and loads them via @loader_path / @rpath.
 */
import { existsSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export interface LibCandidate {
  path: string;
  source: string;
  exists: boolean;
}

/** npm name of the prebuilt native bundle for darwin/arm64. */
export const PLATFORM_PACKAGE = "@johnhenry/backend-mlx-darwin-arm64";

type Env = Record<string, string | undefined>;
const processEnv = (): Env => (globalThis as { process?: { env?: Env } }).process?.env ?? {};

function packageFile(spec: string, ...rel: string[]): string | undefined {
  try {
    const req = createRequire(import.meta.url);
    return join(dirname(req.resolve(`${spec}/package.json`)), ...rel);
  } catch {
    return undefined;
  }
}

/** `$LAYA_MLXC_PATH` may name the dylib itself or the directory holding it. */
function envPath(value: string | undefined): string | undefined {
  if (!value) return undefined;
  try {
    if (statSync(value).isDirectory()) return join(value, "libmlxc.dylib");
  } catch {
    // missing path: reported as a non-existent candidate
  }
  return value;
}

export function libCandidates(explicit?: string): LibCandidate[] {
  const here = dirname(fileURLToPath(import.meta.url)); // src/ or dist/
  const env = processEnv();
  const list: [string | undefined, string][] = [
    [explicit, "option libPath"],
    [envPath(env.LAYA_MLXC_PATH), "$LAYA_MLXC_PATH"],
    [envPath(env.LAYA_MLXC_LIB), "$LAYA_MLXC_LIB"],
    [packageFile(PLATFORM_PACKAGE, "lib", "libmlxc.dylib"), PLATFORM_PACKAGE],
    [join(here, "..", "prebuilds", "darwin-arm64", "libmlxc.dylib"), "backend-mlx local build (npm run build:mlxc)"],
    [packageFile("@nielspeter/mlx-ts-darwin-arm64", "libmlxc.dylib"), "@nielspeter/mlx-ts-darwin-arm64"],
    ["/opt/homebrew/opt/mlx-c/lib/libmlxc.dylib", "Homebrew"],
    ["/opt/homebrew/lib/libmlxc.dylib", "Homebrew"],
  ];
  return list.filter((e): e is [string, string] => !!e[0]).map(([path, source]) => ({ path, source, exists: existsSync(path) }));
}

export function resolveLib(explicit?: string): LibCandidate {
  const cands = libCandidates(explicit);
  if (explicit) {
    const c = cands[0]!;
    if (!c.exists) throw new Error(`backend-mlx: libPath ${explicit} does not exist`);
    return c;
  }
  const pinned = cands.find((c) => c.source.startsWith("$LAYA_MLXC"));
  if (pinned && !pinned.exists) {
    throw new Error(`backend-mlx: ${pinned.source} is set but ${pinned.path} does not exist`);
  }
  const found = cands.find((c) => c.exists);
  if (!found) {
    throw new Error(
      `backend-mlx: libmlxc.dylib not found. Install ${PLATFORM_PACKAGE} (npm adds it automatically on ` +
        "darwin/arm64 unless optional dependencies are omitted), run `npm run build:mlxc -w @johnhenry/backend-mlx`, " +
        "`brew install mlx-c`, or set LAYA_MLXC_PATH. Tried:\n" +
        cands.map((c) => `  ${c.path} (${c.source})`).join("\n"),
    );
  }
  return found;
}

/** True when this process can plausibly run MLX (macOS on Apple Silicon). */
export function mlxPlatformSupported(): boolean {
  const p = (globalThis as { process?: { platform?: string; arch?: string } }).process;
  return p?.platform === "darwin" && p?.arch === "arm64";
}
