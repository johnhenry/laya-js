/**
 * `load()`: resolve a checkpoint (local directory, Hub repo through the HF
 * cache, or a base URL in browsers), pick a backend, and build the agent.
 * Runtime-specific I/O and backend imports live behind the `#io` import map
 * (io-node.ts / io-browser.ts), so browser bundles never see node: modules,
 * koffi or libmlxc.
 */
import type { Backend } from "@johnhenry/tensor-backend";
import { hasNativeQuantized } from "@johnhenry/tensor-backend";
import { readCheckpoint, createBackend } from "#io";
import { parseModernBertConfig } from "@johnhenry/modernbert";
import { createAgent, validateConfig, type AgentOptions, type Dtype, type LayaAgent } from "./agent.ts";
import type { BackendRequest, ProgressInfo } from "./common.ts";

export interface LoadOptions extends AgentOptions {
  /** A Backend instance (you own it), or which one to create (default "auto": mlx → webgpu → cpu). */
  backend?: Backend | BackendRequest;
  /** Default "f16"; the cpu backend always computes in f32. */
  dtype?: Dtype;
  /** Python's `device`: "gpu"/"metal" or "cpu" selects the MLX device (with backend "auto" or "mlx"). */
  device?: "gpu" | "metal" | "cpu";
  revision?: string;
  token?: string;
  subfolder?: string;
  /** Never touch the network (default: $HF_HUB_OFFLINE). */
  offline?: boolean;
  onProgress?: (e: ProgressInfo) => void;
  /** fetch for Hub/URL downloads (browsers, proxies, tests). */
  fetch?: typeof fetch;
  /** Where the temperature-clamping warning goes (default console.warn). */
  warn?: (message: string) => void;
  /**
   * Quantized checkpoints only. "device" (default): keep the int8/int4
   * weights quantized on the backend (less device memory) when it has native
   * quantized kernels (MLX, WebGPU); otherwise, as with "dequantize", rebuild
   * float weights on the host while loading (the CPU backend always does).
   */
  quantized?: "device" | "dequantize";
}

/**
 * Loads a Laya checkpoint and returns a ready agent
 * (`laya_mlx.load(model_id_or_path, ...)`).
 */
export async function load(modelIdOrPath: string, opts: LoadOptions = {}): Promise<LayaAgent> {
  if (opts.dtype !== undefined && opts.dtype !== "f16" && opts.dtype !== "f32") throw new Error(`dtype must be one of ["f16", "f32"]`);
  if (opts.device !== undefined && !["gpu", "metal", "cpu"].includes(opts.device)) throw new Error("MLX device must be 'gpu', 'metal', or 'cpu'");
  if (opts.quantized !== undefined && opts.quantized !== "device" && opts.quantized !== "dequantize") throw new Error(`quantized must be "device" or "dequantize"`);
  const batchSize = opts.batchSize ?? 16;
  if (typeof batchSize !== "number" || !Number.isInteger(batchSize) || batchSize < 1) throw new Error("batch_size must be a positive integer");
  const pad = opts.padToMultiple ?? null;
  if (pad !== null && (typeof pad !== "number" || !Number.isInteger(pad) || pad < 1)) throw new Error("pad_to_multiple must be a positive integer or None");

  const ckpt = await readCheckpoint(modelIdOrPath, opts);
  validateConfig(ckpt.agentConfig, parseModernBertConfig(ckpt.encoderConfig)); // before reading weights
  const req = opts.backend ?? "auto";
  const owns = typeof req === "string";
  const device = opts.device === undefined ? undefined : opts.device === "cpu" ? "cpu" : "gpu";
  const backend = typeof req === "string" ? await createBackend(req, device) : req;
  let agent: LayaAgent | undefined;
  try {
    // quantized checkpoints dequantize straight to the dtype the agent will compute in (createAgent's rule)
    const computeDtype = (opts.dtype ?? "f16") === "f16" && backend.name !== "cpu" && backend.supports("f16") ? "f16" : "f32";
    const quantized = (opts.quantized ?? "device") === "device" && hasNativeQuantized(backend) ? "device" : "dequantize";
    const weights = await ckpt.weights({ dtype: computeDtype, quantized });
    agent = await createAgent({
      backend,
      encoderConfig: ckpt.encoderConfig,
      agentConfig: ckpt.agentConfig,
      weights: weights.get,
      tokenizer: ckpt.tokenizer,
      modelId: String(modelIdOrPath),
      ownsBackend: owns,
      ...(opts.dtype ? { dtype: opts.dtype } : {}),
      batchSize,
      padToMultiple: pad,
      ...(opts.cachePrompts !== undefined ? { cachePrompts: opts.cachePrompts } : {}),
      ...(opts.compile !== undefined ? { compile: opts.compile } : {}),
      ...(opts.warn ? { warn: opts.warn } : {}),
    });
    // `temperature` is a checkpoint buffer; calibration uses the JSON config (model.py)
    const extra = weights.remaining().filter((n) => n !== "temperature");
    if (extra.length) {
      agent.dispose();
      throw new Error(`Received parameters not in model: ${extra.sort().join(", ")}`);
    }
    return agent;
  } catch (e) {
    if (owns && !agent) backend.destroy?.();
    throw e;
  }
}
