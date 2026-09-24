/**
 * A checkpoint served from any base URL (static host, CDN, object store):
 * config and tokenizer files are fetched, `model.safetensors` is read with
 * HTTP Range requests by `openSafetensors`. Shared by io-browser.ts (base
 * URLs and relative paths) and io-node.ts (http(s) URLs). No caching: put a
 * cache (HTTP, service worker) in front, or use a Hub repo id.
 */
import type { AgentConfig } from "@johnhenry/laya-core";
import { loadTokenizer } from "@johnhenry/laya-core";
import { readWeights } from "./weights.ts";
import type { Checkpoint, ResolveOptions } from "./common.ts";

/** `http(s)://…` and other scheme URLs. */
export const isAbsoluteUrl = (s: string): boolean => /^[a-z][a-z0-9+.-]*:\/\//i.test(s);

export async function readCheckpointFromUrl(baseUrl: string | URL, prefix: string, opts: ResolveOptions = {}): Promise<Checkpoint> {
  const doFetch = opts.fetch ?? globalThis.fetch.bind(globalThis);
  const base = new URL(String(baseUrl).replace(/\/*$/, "/") + prefix, (globalThis as { location?: { href: string } }).location?.href);
  const url = (file: string) => new URL(file, base).href;
  const text = async (file: string) => {
    const res = await doFetch(url(file));
    if (!res.ok) throw new Error(`Not a complete Laya checkpoint: ${url(file)} is missing (HTTP ${res.status})`);
    return res.text();
  };
  const [agentConfig, encoderConfig, tokJson, tokConfig] = await Promise.all([
    text("rl_agent_config.json").then(JSON.parse),
    text("encoder/config.json").then(JSON.parse),
    text("tokenizer/tokenizer.json"),
    text("tokenizer/tokenizer_config.json"),
  ]);
  return {
    location: base.href,
    agentConfig: agentConfig as AgentConfig,
    encoderConfig: encoderConfig as Record<string, unknown>,
    tokenizer: loadTokenizer(tokJson, tokConfig),
    // openSafetensors reads URLs with Range requests
    weights: (w) => readWeights(url("model.safetensors"), { ...(opts.fetch ? { fetch: opts.fetch } : {}), ...(w?.dtype ? { dtype: w.dtype } : {}), ...(w?.quantized ? { quantized: w.quantized } : {}) }),
  };
}
