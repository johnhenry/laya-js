/**
 * Golden fixtures generated from Python laya-mlx by
 * `laya-mlx/scripts/dump_js_fixtures.py` (run `npm run fixtures` at the repo root).
 * Node/Bun/Deno only (reads from disk). Never hand-edit the data.
 */
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

export const DATA_DIR: string = fileURLToPath(new URL("../data/", import.meta.url));

/** Absolute path inside the data directory, e.g. fixturePath("tiny", "model.safetensors"). */
export function fixturePath(...parts: string[]): string {
  return DATA_DIR + parts.join("/");
}

export async function loadJson<T = unknown>(...parts: string[]): Promise<T> {
  return JSON.parse(await readFile(fixturePath(...parts), "utf8")) as T;
}

export const MODELS = ["english", "multilingual", "typed-decisions"] as const;
export type ModelName = (typeof MODELS)[number];
export const REPOS: Readonly<Record<ModelName, string>> = {
  english: "aac6fef/laya-mlx",
  multilingual: "aac6fef/laya-multilingual-mlx",
  "typed-decisions": "aac6fef/laya-typed-decisions-mlx",
};

export interface EncodedTensor { dtype: "f32" | "i32" | "bool"; shape: number[]; b64: string }

/** data/real/<model>.json */
export interface RealFixture {
  repo: string;
  model_dir: string; // local HF cache snapshot used to generate it
  config: Record<string, unknown>;
  cases: Array<{
    case: string;
    state: unknown;
    questions: Record<string, unknown>;
    items: Array<{ ids: number[]; markers: number[]; qtype: 0 | 1 | 2 }>;
    outputs: Array<{ logits: number[]; act: number[] }>; // fp32, logits trimmed to k
    result_fp32: any;
    result_fp16: any;
  }>;
  embeddings: { texts: string[]; vectors: EncodedTensor };
}

export const loadReal = (m: ModelName) => loadJson<RealFixture>("real", `${m}.json`);
