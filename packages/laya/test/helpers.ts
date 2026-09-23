/** Test-only helpers (not a public API). */
import type { Batch, PreparedItem } from "@johnhenry/laya-core";

/** Port of laya-mlx `collate_items` (laya-core's collate may replace this). */
export function collateItems(items: readonly PreparedItem[], padId: number): Batch {
  if (!items.length) throw new Error("Cannot collate an empty batch");
  const B = items.length;
  const L = Math.max(...items.map((i) => i.ids.length));
  const M = Math.max(2, ...items.map((i) => i.markers.length));
  const batch: Batch = {
    size: B, length: L, markerCount: M,
    inputIds: new Int32Array(B * L).fill(padId),
    attentionMask: new Uint8Array(B * L),
    markerPos: new Int32Array(B * M),
    markerMask: new Uint8Array(B * M),
    qtype: Int32Array.from(items.map((i) => i.qtype)),
  };
  items.forEach((it, r) => {
    batch.inputIds.set(it.ids, r * L);
    batch.attentionMask.fill(1, r * L, r * L + it.ids.length);
    batch.markerPos.set(it.markers, r * M);
    batch.markerMask.fill(1, r * M, r * M + it.markers.length);
  });
  return batch;
}

export interface ErrStats { maxAbs: number; maxRel: number; worst: number; n: number }

/** Max errors over `idx` (default: all); `worst` = max |a-e| / (atol + rtol|e|). */
export function errStats(a: ArrayLike<number>, e: ArrayLike<number>, atol: number, rtol: number, idx?: Iterable<number>): ErrStats {
  let maxAbs = 0, maxRel = 0, worst = 0, n = 0;
  const it = idx ?? Array.from({ length: e.length }, (_, i) => i);
  for (const i of it) {
    const d = Math.abs(a[i]! - e[i]!);
    if (!(d <= Number.MAX_VALUE)) return { maxAbs: Infinity, maxRel: Infinity, worst: Infinity, n };
    maxAbs = Math.max(maxAbs, d);
    maxRel = Math.max(maxRel, d / Math.max(Math.abs(e[i]!), 1e-30));
    worst = Math.max(worst, d / (atol + rtol * Math.abs(e[i]!)));
    n++;
  }
  return { maxAbs, maxRel, worst, n };
}

export const fmt = (s: ErrStats) => `maxAbs ${s.maxAbs.toExponential(2)} maxRel ${s.maxRel.toExponential(2)} (n=${s.n})`;
