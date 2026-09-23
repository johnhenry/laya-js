/**
 * Node/Bun/Deno helpers (filesystem). Import from `@johnhenry/laya-core/node`.
 */
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { loadTokenizer, type HFLayaTokenizer } from "./tokenizer.ts";

/** Load `tokenizer.json` + `tokenizer_config.json` from a directory (e.g. `<checkpoint>/tokenizer`). */
export async function loadTokenizerFromDir(dir: string): Promise<HFLayaTokenizer> {
  const [json, config] = await Promise.all([
    readFile(join(dir, "tokenizer.json"), "utf8"),
    readFile(join(dir, "tokenizer_config.json"), "utf8"),
  ]);
  return loadTokenizer(json, config);
}
