import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { hubCacheDir } from "../src/index.ts";

/** The three Laya repos that must already be in the local cache (downloaded by `hf download`). */
export const LAYA_REPOS = ["aac6fef/laya-mlx", "aac6fef/laya-multilingual-mlx", "aac6fef/laya-typed-decisions-mlx"] as const;

/** Real cache dir, independent of env overrides tests may set. */
export const REAL_CACHE = hubCacheDir();

export function repoCached(repo: string): boolean {
  return existsSync(join(REAL_CACHE, `models--${repo.replace("/", "--")}`, "refs", "main"));
}

export async function tempDir(prefix = "hf-cache-"): Promise<{ dir: string; cleanup: () => Promise<void> }> {
  const dir = await mkdtemp(join(tmpdir(), prefix));
  return { dir, cleanup: () => rm(dir, { recursive: true, force: true }) };
}

/** Python with huggingface_hub, for cross-checking the layout ($MATH_PLUS_ORACLE_PYTHON, else python3). */
function findPython(): string | undefined {
  for (const c of [process.env.MATH_PLUS_ORACLE_PYTHON, "python3"].filter(Boolean) as string[]) {
    try {
      execFileSync(c, ["-c", "import huggingface_hub"], { stdio: "ignore" });
      return c;
    } catch {
      // next
    }
  }
  return undefined;
}
export const PYTHON = findPython();
export const pythonSkip = PYTHON ? false : "no python with huggingface_hub (set MATH_PLUS_ORACLE_PYTHON)";

/** huggingface_hub.try_to_load_from_cache(repo, file, cache_dir, revision) -> path | "<NO_EXIST>" | null. */
export function pyTryToLoad(repo: string, file: string, cacheDir: string, revision = "main"): string | null {
  const script = [
    "import sys, json",
    "from huggingface_hub import try_to_load_from_cache",
    "r = try_to_load_from_cache(sys.argv[1], sys.argv[2], cache_dir=sys.argv[3], revision=sys.argv[4])",
    "print(json.dumps(r if isinstance(r, str) or r is None else '<NO_EXIST>'))",
  ].join("\n");
  const env = { ...process.env, HF_HUB_OFFLINE: "1" };
  return JSON.parse(execFileSync(PYTHON as string, ["-c", script, repo, file, cacheDir, revision], { encoding: "utf8", env })) as string | null;
}

/** Whether huggingface.co answers (network tests skip, never fail, without it). */
export async function hubReachable(): Promise<boolean> {
  try {
    const res = await fetch("https://huggingface.co/api/models/aac6fef/laya-mlx/revision/main", { signal: AbortSignal.timeout(8000) });
    await res.body?.cancel();
    return res.ok;
  } catch {
    return false;
  }
}
