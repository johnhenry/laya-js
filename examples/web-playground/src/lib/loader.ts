/**
 * Load a `@johnhenry/laya` agent in the browser on WebGPU, with aggregated
 * download progress. Weights come from https://huggingface.co/<repo>/resolve/…
 * through hf-cache's browser store (Cache API), so a reload is served locally.
 */
import { load, type LayaAgent } from "@johnhenry/laya";

export interface LoadProgress {
  loaded: number;
  total: number;
  file?: string;
  /** Files seen so far with a known size. */
  files: number;
}

export type BrowserAgent = LayaAgent;

export async function loadBrowserAgent(
  repo: string,
  { onProgress, dtype = "f16" }: { onProgress?: (p: LoadProgress) => void; dtype?: "f16" | "f32" } = {},
): Promise<{ agent: BrowserAgent; seconds: number; downloaded: number }> {
  const perFile = new Map<string, { loaded: number; total: number }>();
  let downloaded = 0;
  const report = (file?: string) => {
    let loaded = 0;
    let total = 0;
    for (const v of perFile.values()) {
      loaded += v.loaded;
      total += v.total;
    }
    onProgress?.({ loaded, total, file, files: perFile.size });
  };
  const started = performance.now();
  const agent = await load(repo, {
    backend: "webgpu",
    dtype,
    onProgress: (e) => {
      const key = e.file ?? "model";
      const prev = perFile.get(key);
      downloaded += e.loaded - (prev?.loaded ?? 0);
      perFile.set(key, { loaded: e.loaded, total: e.total ?? Math.max(e.loaded, prev?.total ?? 0) });
      report(key);
    },
  });
  return { agent, seconds: (performance.now() - started) / 1000, downloaded };
}
