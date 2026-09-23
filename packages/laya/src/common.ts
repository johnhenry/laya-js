/** Runtime-neutral pieces shared by io-node.ts and io-browser.ts. */
import type { Backend } from "@johnhenry/tensor-backend";
import type { AgentConfig, LayaTokenizer } from "@johnhenry/laya-core";
import type { ConsumingWeights } from "./weights.ts";

export type BackendName = "mlx" | "webgpu" | "cpu";
export type BackendRequest = "auto" | BackendName;

export interface ProgressInfo {
  file: string;
  loaded: number;
  total?: number;
}

export interface ResolveOptions {
  revision?: string;
  token?: string;
  subfolder?: string;
  offline?: boolean;
  onProgress?: (e: ProgressInfo) => void;
  fetch?: typeof fetch;
}

/** A resolved checkpoint; weights are read on demand (after the backend exists). */
export interface Checkpoint {
  /** Directory (Node) or base URL (browser) the files came from. */
  location: string;
  agentConfig: AgentConfig;
  encoderConfig: Record<string, unknown>;
  tokenizer: LayaTokenizer & { encodeWithSpecialTokens(text: string): number[] };
  weights: () => Promise<ConsumingWeights>;
}

/** Files fetched from the Hub (resolve_model's allow_patterns minus mlx_config.json). */
export const CHECKPOINT_FILES = [
  "model.safetensors",
  "rl_agent_config.json",
  "encoder/config.json",
  "tokenizer/tokenizer.json",
  "tokenizer/tokenizer_config.json",
] as const;

/** Files whose absence makes a directory "not a complete Laya checkpoint". */
export const REQUIRED_FILES = ["model.safetensors", "rl_agent_config.json", "encoder/config.json"] as const;

/** `subfolder must be a relative path inside the model repository` (PurePosixPath check). */
export function checkSubfolder(subfolder: string): string {
  const parts = subfolder.split("/");
  if (subfolder.startsWith("/") || parts.includes("..")) {
    throw new Error("subfolder must be a relative path inside the model repository");
  }
  return parts.filter((p) => p !== "" && p !== ".").join("/");
}

export type { Backend };
