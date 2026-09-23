/**
 * Node/Bun/Deno side of `load()` (selected through the `#io` import map;
 * browsers get io-browser.ts): local directories and the huggingface_hub
 * disk cache, and every backend including native MLX.
 */
import { existsSync, statSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { isAbsolute, join, normalize, resolve } from "node:path";
import type { Backend } from "@johnhenry/tensor-backend";
import type { AgentConfig } from "@johnhenry/laya-core";
import { loadTokenizerFromDir } from "@johnhenry/laya-core/node";
import { snapshot } from "@johnhenry/hf-cache";
import { readWeights } from "./weights.ts";
import { CHECKPOINT_FILES, REQUIRED_FILES, checkSubfolder, type BackendRequest, type Checkpoint, type ResolveOptions } from "./common.ts";

export const RUNTIME = "node" as const;

const isFile = (p: string) => {
  try {
    return statSync(p).isFile();
  } catch {
    return false;
  }
};

/** `resolve_model`: local directory, else a Hub repo through the HF cache. Returns the checkpoint directory. */
export async function resolveModelDir(modelIdOrPath: string, opts: ResolveOptions = {}): Promise<string> {
  const sub = opts.subfolder ? checkSubfolder(opts.subfolder) : "";
  const value = String(modelIdOrPath);
  let dir = value.startsWith("~") ? join(homedir(), value.slice(1)) : value;
  if (!existsSync(dir)) {
    if (value.startsWith("/") || value.startsWith("./") || value.startsWith("../") || value.startsWith("~") || isAbsolute(value)) {
      throw new Error(`Local model directory does not exist: ${value}`);
    }
    const prefix = sub ? sub.replace(/\/+$/, "") + "/" : "";
    const res = await snapshot(value, {
      files: CHECKPOINT_FILES.map((f) => prefix + f),
      ...(opts.revision !== undefined ? { revision: opts.revision } : {}),
      ...(opts.token !== undefined ? { token: opts.token } : {}),
      ...(opts.offline !== undefined ? { offline: opts.offline } : {}),
      ...(opts.fetch ? { fetch: opts.fetch } : {}),
      ...(opts.onProgress ? { onProgress: (e) => opts.onProgress!({ file: e.file, loaded: e.loaded, ...(e.total !== undefined ? { total: e.total } : {}) }) } : {}),
    });
    dir = res.dir;
  }
  dir = resolve(normalize(dir));
  if (sub) dir = join(dir, sub);
  for (const name of REQUIRED_FILES) {
    if (!isFile(join(dir, name))) throw new Error(`Not a complete Laya checkpoint: ${join(dir, name)} is missing`);
  }
  return dir;
}

/** Everything `createAgent` needs, read from disk. */
export async function readCheckpoint(modelIdOrPath: string, opts: ResolveOptions = {}): Promise<Checkpoint> {
  const dir = await resolveModelDir(modelIdOrPath, opts);
  const json = async (name: string) => JSON.parse(await readFile(join(dir, name), "utf8")) as Record<string, unknown>;
  const [agentConfig, encoderConfig, tokenizer] = await Promise.all([
    json("rl_agent_config.json"),
    json("encoder/config.json"),
    loadTokenizerFromDir(join(dir, "tokenizer")),
  ]);
  return {
    location: dir,
    agentConfig: agentConfig as AgentConfig,
    encoderConfig,
    tokenizer,
    weights: () => readWeights(join(dir, "model.safetensors")),
  };
}

/** Why "auto" skipped mlx/webgpu on the last call (diagnostics). */
export const autoSkipped: string[] = [];

/** Creates the requested backend; "auto" = mlx (macOS arm64 + libmlxc) → webgpu (adapter) → cpu. */
export async function createBackend(req: BackendRequest, device?: "gpu" | "cpu"): Promise<Backend> {
  const mlx = async () => {
    const m = await import("@johnhenry/backend-mlx");
    if (!m.mlxPlatformSupported()) throw new Error("MLX needs macOS on Apple Silicon");
    m.resolveLib(); // throws a helpful message when libmlxc.dylib is missing
    return m.createMlxBackend(device ? { device } : {}) as unknown as Backend;
  };
  const webgpu = async () => {
    const w = await import("@johnhenry/backend-webgpu");
    return (await w.createWebGpuBackend()) as unknown as Backend;
  };
  const cpu = async () => (await import("@johnhenry/backend-cpu")).createCpuBackend() as unknown as Backend;
  if (req === "mlx") return mlx();
  if (req === "webgpu") return webgpu();
  if (req === "cpu") return cpu();
  if (req !== "auto") throw new Error(`backend must be a Backend or one of "auto", "mlx", "webgpu", "cpu"; got ${String(req)}`);
  autoSkipped.length = 0;
  for (const [name, make] of [["mlx", mlx], ["webgpu", webgpu]] as const) {
    try {
      return await make();
    } catch (e) {
      autoSkipped.push(`${name}: ${(e as Error).message.split("\n")[0]}`);
    }
  }
  return cpu();
}
