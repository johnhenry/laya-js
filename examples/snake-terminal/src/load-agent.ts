/**
 * Load a `@johnhenry/laya` agent for the Snake policy (Node/Bun).
 * Loads offline from the local Hugging Face cache by default, like the Python demo.
 */
import { load, type LayaAgent, type LoadOptions } from "@johnhenry/laya";

export type BackendName = "auto" | "mlx" | "webgpu" | "cpu";

const PRETTY: Record<string, string> = { mlx: "MLX", webgpu: "WebGPU", cpu: "CPU" };

export async function loadSnakeAgent(
  model: string,
  {
    backend = "auto",
    dtype = "f16",
    offline = true,
    optimize = false,
  }: { backend?: BackendName; dtype?: "f16" | "f32"; offline?: boolean; optimize?: boolean } = {},
): Promise<{ agent: LayaAgent; engine: string }> {
  const options: LoadOptions = { backend, dtype, batchSize: 3, offline };
  // Python `--optimize`: compile + 16-token buckets + bounded prefix reuse.
  if (optimize) Object.assign(options, { padToMultiple: 16, cachePrompts: true, compile: true });
  const agent = await load(model, options);
  const name = agent.backend.name;
  return { agent, engine: `${PRETTY[name] ?? name} · ${agent.dtype.toUpperCase()}` };
}
