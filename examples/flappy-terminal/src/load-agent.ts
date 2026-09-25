/**
 * Load a `@johnhenry/laya` agent for the Flappy Bird policy (Node/Bun).
 * Loads offline from the local Hugging Face cache by default. Same shape
 * as snake-terminal's `load-agent.ts` -- any Laya checkpoint that answers
 * choice + noul questions works here, there is nothing Snake-specific
 * about the loaded agent.
 */
import { load, type LayaAgent, type LoadOptions } from "@johnhenry/laya";

export type BackendName = "auto" | "mlx" | "webgpu" | "cpu";

const PRETTY: Record<string, string> = { mlx: "MLX", webgpu: "WebGPU", cpu: "CPU" };

export async function loadFlappyAgent(
  model: string,
  {
    backend = "auto",
    dtype = "f16",
    offline = true,
    optimize = false,
  }: { backend?: BackendName; dtype?: "f16" | "f32"; offline?: boolean; optimize?: boolean } = {},
): Promise<{ agent: LayaAgent; engine: string }> {
  const options: LoadOptions = { backend, dtype, batchSize: 3, offline };
  if (optimize) Object.assign(options, { padToMultiple: 16, cachePrompts: true, compile: true });
  const agent = await load(model, options);
  const name = agent.backend.name;
  return { agent, engine: `${PRETTY[name] ?? name} · ${agent.dtype.toUpperCase()}` };
}
