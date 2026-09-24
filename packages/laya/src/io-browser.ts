/**
 * Browser side of `load()` (selected through the `#io` import map's
 * `browser` condition). No node: imports and no native backends: the model
 * comes from the Hub through the Cache API (`@johnhenry/hf-cache/browser`)
 * or from any base URL; backends are WebGPU or the CPU reference.
 */
import type { Backend } from "@johnhenry/tensor-backend";
import type { AgentConfig } from "@johnhenry/laya-core";
import { loadTokenizer } from "@johnhenry/laya-core";
import { fetchFile, snapshot } from "@johnhenry/hf-cache/browser";
import { readWeights } from "./weights.ts";
import { isAbsoluteUrl, readCheckpointFromUrl } from "./url.ts";
import { CHECKPOINT_FILES, checkSubfolder, type BackendRequest, type Checkpoint, type ResolveOptions } from "./common.ts";

export const RUNTIME = "browser" as const;

/** Strings treated as a base URL rather than a Hub repo id. */
const isUrlLike = (s: string) => isAbsoluteUrl(s) || s.startsWith("/") || s.startsWith("./") || s.startsWith("../");

export async function readCheckpoint(modelIdOrPath: string, opts: ResolveOptions = {}): Promise<Checkpoint> {
  const sub = opts.subfolder ? checkSubfolder(opts.subfolder) : "";
  const prefix = sub ? sub + "/" : "";
  if (isUrlLike(modelIdOrPath)) return readCheckpointFromUrl(modelIdOrPath, prefix, opts);
  const hub = {
    ...(opts.revision !== undefined ? { revision: opts.revision } : {}),
    ...(opts.token !== undefined ? { token: opts.token } : {}),
    ...(opts.offline !== undefined ? { offline: opts.offline } : {}),
    ...(opts.fetch ? { fetch: opts.fetch } : {}),
    ...(opts.onProgress ? { onProgress: (e: { file: string; loaded: number; total: number | undefined }) => opts.onProgress!({ file: e.file, loaded: e.loaded, ...(e.total !== undefined ? { total: e.total } : {}) }) } : {}),
  };
  const snap = await snapshot(modelIdOrPath, { ...hub, files: CHECKPOINT_FILES.map((f) => prefix + f) });
  const pinned = { ...hub, revision: snap.commit };
  const text = async (f: string) => (await fetchFile(modelIdOrPath, prefix + f, pinned)).text();
  const [agentConfig, encoderConfig, tokJson, tokConfig] = await Promise.all([
    text("rl_agent_config.json").then(JSON.parse),
    text("encoder/config.json").then(JSON.parse),
    text("tokenizer/tokenizer.json"),
    text("tokenizer/tokenizer_config.json"),
  ]);
  return {
    location: snap.dir + prefix,
    agentConfig: agentConfig as AgentConfig,
    encoderConfig: encoderConfig as Record<string, unknown>,
    tokenizer: loadTokenizer(tokJson, tokConfig),
    weights: async (w) => {
      const src = await fetchFile(modelIdOrPath, prefix + "model.safetensors", pinned);
      return readWeights(src, { ...(opts.fetch ? { fetch: opts.fetch } : {}), ...(w?.dtype ? { dtype: w.dtype } : {}) });
    },
  };
}

/** "auto" = webgpu (adapter available) → cpu. MLX is native-only. */
export async function createBackend(req: BackendRequest, _device?: "gpu" | "cpu"): Promise<Backend> {
  const webgpu = async () => (await (await import("@johnhenry/backend-webgpu")).createWebGpuBackend()) as unknown as Backend;
  const cpu = async () => (await import("@johnhenry/backend-cpu")).createCpuBackend() as unknown as Backend;
  if (req === "mlx") throw new Error("The mlx backend needs Node or Bun on macOS/arm64; use \"webgpu\" or \"cpu\" in browsers");
  if (req === "webgpu") return webgpu();
  if (req === "cpu") return cpu();
  if (req !== "auto") throw new Error(`backend must be a Backend or one of "auto", "mlx", "webgpu", "cpu"; got ${String(req)}`);
  try {
    return await webgpu();
  } catch {
    return cpu();
  }
}
