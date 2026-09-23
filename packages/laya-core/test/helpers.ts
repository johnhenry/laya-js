import { existsSync } from "node:fs";
import { fixturePath, loadReal, MODELS, type ModelName, type RealFixture } from "@johnhenry/laya-fixtures";
import { loadTokenizerFromDir } from "@johnhenry/laya-core/node";
import type { HFLayaTokenizer } from "@johnhenry/laya-core";

export { MODELS, type ModelName };

const reals = new Map<ModelName, Promise<RealFixture>>();
export const real = (m: ModelName) => {
  if (!reals.has(m)) reals.set(m, loadReal(m));
  return reals.get(m)!;
};

/** Tokenizer dir for a checkpoint from the local HF cache (the fixture's model_dir), or null. */
export async function tokenizerDir(m: ModelName): Promise<string | null> {
  const dir = (await real(m)).model_dir + "/tokenizer";
  return existsSync(dir + "/tokenizer.json") ? dir : null;
}

const toks = new Map<string, Promise<HFLayaTokenizer>>();
export function tokenizer(dir: string): Promise<HFLayaTokenizer> {
  if (!toks.has(dir)) toks.set(dir, loadTokenizerFromDir(dir));
  return toks.get(dir)!;
}

export const tinyTokenizer = () => tokenizer(fixturePath("tiny", "tokenizer"));

/** Tokenizer dirs for every checkpoint (null when not cached), resolved once. */
export const DIRS: Record<ModelName, string | null> = Object.fromEntries(
  await Promise.all(MODELS.map(async (m) => [m, await tokenizerDir(m)] as const)),
) as Record<ModelName, string | null>;
export const skipReason = (m: ModelName) => (DIRS[m] ? false : `checkpoint tokenizer for ${m} not in the local HF cache`);
