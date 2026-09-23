/**
 * Locating libmlxc.dylib (mlx-c, Apple's C API over MLX).
 *
 * Resolution order (first existing file wins):
 *   1. the explicit `libPath` option
 *   2. `$LAYA_MLXC_LIB`
 *   3. this package's `prebuilds/darwin-arm64/libmlxc.dylib`
 *      (produced by `scripts/build-mlxc.sh`: mlx-c v0.6+ built against the
 *      `mlx-metal` 0.32.2 wheel — the same MLX the Python reference uses)
 *   4. `@nielspeter/mlx-ts-darwin-arm64/libmlxc.dylib` (npm; MLX 0.32.1)
 *   5. Homebrew (`brew install mlx-c`)
 *
 * Each bundle is relocatable: libmlxc sits beside libmlx.dylib and
 * mlx.metallib and loads them via @loader_path / @rpath.
 */
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export interface LibCandidate {
  path: string;
  source: string;
  exists: boolean;
}

function platformPackage(): string | undefined {
  try {
    const req = createRequire(import.meta.url);
    return join(dirname(req.resolve("@nielspeter/mlx-ts-darwin-arm64/package.json")), "libmlxc.dylib");
  } catch {
    return undefined;
  }
}

export function libCandidates(explicit?: string): LibCandidate[] {
  const here = dirname(fileURLToPath(import.meta.url)); // src/ or dist/
  const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env ?? {};
  const list: [string | undefined, string][] = [
    [explicit, "option libPath"],
    [env.LAYA_MLXC_LIB, "$LAYA_MLXC_LIB"],
    [join(here, "..", "prebuilds", "darwin-arm64", "libmlxc.dylib"), "backend-mlx prebuilds (scripts/build-mlxc.sh)"],
    [platformPackage(), "@nielspeter/mlx-ts-darwin-arm64"],
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
  const found = cands.find((c) => c.exists);
  if (!found) {
    throw new Error(
      "backend-mlx: libmlxc.dylib not found. Run `npm run build:mlxc -w @johnhenry/backend-mlx`, " +
        "install @nielspeter/mlx-ts-darwin-arm64, `brew install mlx-c`, or set LAYA_MLXC_LIB. Tried:\n" +
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
