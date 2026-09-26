/**
 * Local HTTP server proxying MLX, ONNX (@receptron/laya), and Jev (TypeSafe)
 * predictions for web-playground's compare mode. Runs in Node only -- MLX is
 * native FFI, ONNX runs on onnxruntime-node, and TYPESAFE_API_KEY must never
 * reach a browser. Zero HTTP-framework dependency, matching
 * ../web-playground/scripts/serve.ts's precedent.
 */
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { load, type LayaAgent } from "@johnhenry/laya";
import type { Questions, State } from "@johnhenry/laya-core";
import type { Question as OnnxQuestion } from "@receptron/laya";
import type { EntryType as JevEntryType } from "@typesafe-ai/sdk";
import { mapOnnxResult, mapJevResult, toJevQuestions, type OnnxSystemOneResult, type JevSystemOneResult, type RemotePredictResult } from "./mapping.ts";

const PORT = Number(process.env.PORT ?? 5199);
const DEFAULT_MLX_REPO = "aac6fef/laya-mlx";

type BackendName = "mlx" | "onnx" | "jev";

interface Availability {
  available: boolean;
  reason?: string;
}

interface PredictBody {
  backend: BackendName;
  repo?: string;
  dtype?: "f16" | "f32";
  state: State;
  questions: Questions;
}

interface PredictResponse {
  result: RemotePredictResult;
  seconds: number;
  engine: string;
}

// ---------------------------------------------------------------- availability

function mlxAvailability(): Availability {
  if (process.platform !== "darwin" || process.arch !== "arm64") {
    return { available: false, reason: "MLX requires Apple Silicon (darwin/arm64)" };
  }
  return { available: true };
}

async function onnxAvailability(): Promise<Availability> {
  try {
    await import("@receptron/laya");
    return { available: true };
  } catch {
    return { available: false, reason: "@receptron/laya is not installed (npm install @receptron/laya)" };
  }
}

function jevAvailability(): Availability {
  if (!process.env.TYPESAFE_API_KEY) return { available: false, reason: "TYPESAFE_API_KEY is not set" };
  return { available: true };
}

// ---------------------------------------------------------------- agent caches
// One in-memory instance per backend config, reused across requests (e.g. from
// web-playground's queue) so only the first call per config pays load time.

const mlxAgents = new Map<string, Promise<LayaAgent>>();
function getMlxAgent(repo: string, dtype: "f16" | "f32"): Promise<LayaAgent> {
  const key = `${repo}:${dtype}`;
  let agent = mlxAgents.get(key);
  if (!agent) {
    agent = load(repo, { backend: "mlx", dtype, offline: true });
    agent.catch(() => mlxAgents.delete(key));
    mlxAgents.set(key, agent);
  }
  return agent;
}

let onnxAgent: Promise<import("@receptron/laya").Laya> | undefined;
async function getOnnxAgent(): Promise<import("@receptron/laya").Laya> {
  if (!onnxAgent) {
    const { Laya } = await import("@receptron/laya");
    onnxAgent = Laya.load();
    onnxAgent.catch(() => (onnxAgent = undefined));
  }
  return onnxAgent;
}

let jevClient: import("@typesafe-ai/sdk").TypeSafeClient | undefined;
async function getJevClient(): Promise<import("@typesafe-ai/sdk").TypeSafeClient> {
  if (!jevClient) {
    const { TypeSafeClient } = await import("@typesafe-ai/sdk");
    jevClient = new TypeSafeClient();
  }
  return jevClient;
}

// ---------------------------------------------------------------- predict

async function predict(body: PredictBody): Promise<PredictResponse> {
  const t0 = performance.now();
  if (body.backend === "mlx") {
    const repo = body.repo || DEFAULT_MLX_REPO;
    const dtype = body.dtype ?? "f16";
    const agent = await getMlxAgent(repo, dtype);
    const result = await agent.predict(body.state, body.questions);
    return { result, seconds: (performance.now() - t0) / 1000, engine: `MLX · ${dtype.toUpperCase()}` };
  }
  if (body.backend === "onnx") {
    // The one checkpoint receptron/laya-onnx publishes today (English/ModernBERT-large,
    // matching aac6fef/laya-mlx) -- Laya.load()'s own default, so no repo option to pass.
    const availability = await onnxAvailability();
    if (!availability.available) throw new Error(availability.reason);
    const agent = await getOnnxAgent();
    const raw = (await agent.systemOne(body.state, body.questions as unknown as Record<string, OnnxQuestion>)) as unknown as OnnxSystemOneResult;
    return { result: mapOnnxResult(raw), seconds: (performance.now() - t0) / 1000, engine: "ONNX (receptron/laya) · English" };
  }
  if (body.backend === "jev") {
    const availability = jevAvailability();
    if (!availability.available) throw new Error(availability.reason);
    const client = await getJevClient();
    const raw = (await client.systemOne({ state: body.state as unknown as JevEntryType, questions: toJevQuestions(body.questions) })) as unknown as JevSystemOneResult;
    return { result: mapJevResult(raw), seconds: (performance.now() - t0) / 1000, engine: "Jev (TypeSafe)" };
  }
  throw new Error(`Unknown backend "${(body as { backend: string }).backend}"`);
}

// ---------------------------------------------------------------- http

function withCors(res: ServerResponse): ServerResponse {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  return res;
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  withCors(res)
    .writeHead(status, { "Content-Type": "application/json" })
    .end(JSON.stringify(body));
}

async function readJsonBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  const text = Buffer.concat(chunks).toString("utf8");
  return text ? JSON.parse(text) : {};
}

const server = createServer((req, res) => {
  void (async () => {
    const url = new URL(req.url ?? "/", "http://localhost");
    if (req.method === "OPTIONS") {
      withCors(res).writeHead(204).end();
      return;
    }
    if (req.method === "GET" && url.pathname === "/health") {
      const [mlx, onnx, jev] = await Promise.all([mlxAvailability(), onnxAvailability(), jevAvailability()]);
      sendJson(res, 200, { mlx, onnx, jev });
      return;
    }
    if (req.method === "POST" && url.pathname === "/predict") {
      let body: PredictBody;
      try {
        body = (await readJsonBody(req)) as PredictBody;
      } catch {
        sendJson(res, 400, { error: "Invalid JSON body" });
        return;
      }
      try {
        sendJson(res, 200, await predict(body));
      } catch (e) {
        sendJson(res, 500, { error: (e as Error).message });
      }
      return;
    }
    sendJson(res, 404, { error: "Not found" });
  })();
});

if (import.meta.url === `file://${process.argv[1]}`) {
  server.listen(PORT, () => console.log(`laya-server listening on http://localhost:${PORT}/`));
}

export { server };
