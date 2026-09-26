/**
 * A thin `fetch()` wrapper around ../../../laya-server's `/predict` and
 * `/health` endpoints -- MLX, ONNX, and Jev all need a Node process (native
 * FFI, onnxruntime-node, and a server-side-only API key respectively), never
 * the browser. The returned object is shaped like `BrowserAgent.predict`, so
 * it plugs into the existing compare-mode run/render pipeline unchanged.
 */
import type { AnswerLike } from "./queue.ts";
import type { Json } from "./state-fields.ts";

export type RemoteBackendName = "mlx" | "onnx" | "jev";

export interface Availability {
  available: boolean;
  reason?: string;
}

export interface HealthReport {
  mlx: Availability;
  onnx: Availability;
  jev: Availability;
}

function trimSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

export async function checkHealth(serverUrl: string): Promise<HealthReport> {
  const res = await fetch(`${trimSlash(serverUrl)}/health`);
  if (!res.ok) throw new Error(`Health check failed: HTTP ${res.status}`);
  return (await res.json()) as HealthReport;
}

export interface RemotePredictResult {
  answers: Record<string, AnswerLike>;
  usage?: { input_tokens?: number };
}

/** Matches `BrowserAgent.predict`'s call shape -- the only method `runOn()` uses. */
export function createRemoteAgent(serverUrl: string, backend: RemoteBackendName, opts: { repo?: string; dtype?: "f16" | "f32" } = {}): { predict(state: Json, questions: Record<string, unknown>): Promise<RemotePredictResult> } {
  return {
    async predict(state, questions) {
      const res = await fetch(`${trimSlash(serverUrl)}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ backend, repo: opts.repo, dtype: opts.dtype, state, questions }),
      });
      const body = (await res.json()) as { result: RemotePredictResult; seconds: number; engine: string } | { error: string };
      if (!res.ok || "error" in body) throw new Error("error" in body ? body.error : `Request failed: HTTP ${res.status}`);
      return body.result;
    },
  };
}
