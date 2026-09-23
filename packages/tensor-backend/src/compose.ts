import type { Backend, Tensor } from "./index.ts";

/** gelu(value) · gate, where [value, gate] = split(x, 2, -1). */
export function geglu<T extends Tensor>(b: Backend<T>, x: T): T {
  if (b.geglu) return b.geglu(x);
  return b.scope(() => {
    const [value, gate] = b.split(x, 2, x.shape.length - 1);
    return b.mul(b.gelu(value!), gate!);
  });
}

/** Masked mean over axis 1 in f32: x [B, L, D], mask bool [B, L] → [B, D]. */
export function meanPool<T extends Tensor>(b: Backend<T>, x: T, mask: T): T {
  if (b.meanPool) return b.meanPool(x, mask);
  return b.scope(() => {
    const [B, L] = x.shape as [number, number, number];
    const m = b.cast(b.reshape(mask, [B, L, 1]), "f32");
    const total = b.sum(b.mul(b.cast(x, "f32"), m), 1);
    const count = b.maximum(b.sum(m, 1), b.fromHost({ dtype: "f32", shape: [1], data: new Float32Array([1]) }));
    return b.div(total, count);
  });
}
