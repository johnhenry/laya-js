/**
 * Checkpoint weights from a safetensors source, read once and handed to the
 * backend one tensor at a time. Browser-safe (no fs import of our own).
 */
import { openSafetensors, type LazySafetensors } from "@johnhenry/math-plus-safetensors";
import type { HostTensor, DType } from "@johnhenry/tensor-backend";
import type { WeightGetter } from "@johnhenry/modernbert";

type Source = Parameters<typeof openSafetensors>[0];
type OpenOptions = Parameters<typeof openSafetensors>[1];

const DTYPES: Record<string, DType> = { F16: "f16", BF16: "bf16", F32: "f32", I32: "i32", BOOL: "bool", U8: "bool" };

/** Maps upstream PyTorch parameter names to laya-mlx names (`sanitize_weights`). */
export function sanitizeName(name: string): string {
  let n = name.replace(".in_proj_weight", ".in_proj.weight").replace(".in_proj_bias", ".in_proj.bias");
  for (const prefix of ["scorer", "act_head"]) {
    if (n.startsWith(prefix + ".") && !n.startsWith(prefix + ".layers.")) n = prefix + ".layers." + n.slice(prefix.length + 1);
  }
  return n;
}

/**
 * A one-shot WeightGetter over every tensor of a safetensors file. Each
 * tensor is released (`get` forgets it) as soon as it is handed out, so after
 * the backend upload only the backend's copy stays alive: the file is never
 * held twice (host bytes + device copy) for longer than one tensor.
 * `remaining()` lists tensors nobody asked for (`strict` loading).
 */
export interface ConsumingWeights {
  readonly get: WeightGetter;
  remaining(): string[];
}

export function consumingWeights(tensors: Map<string, HostTensor>): ConsumingWeights {
  const map = new Map<string, HostTensor>();
  for (const [name, t] of tensors) {
    const key = sanitizeName(name);
    if (map.has(key)) throw new Error(`Duplicate checkpoint parameter after conversion: ${key}`);
    map.set(key, t);
  }
  return {
    get: (name) => {
      const t = map.get(name);
      if (t) map.delete(name);
      return t;
    },
    remaining: () => [...map.keys()],
  };
}

/** Reads every tensor of an opened lazy file (coalesced reads) into host tensors. */
export async function readAllTensors(file: LazySafetensors): Promise<Map<string, HostTensor>> {
  const views = await file.readMany();
  const out = new Map<string, HostTensor>();
  for (const [name, data] of views) {
    const info = file.info(name);
    const dtype = DTYPES[info.dtype];
    if (!dtype) throw new TypeError(`laya: unsupported safetensors dtype ${info.dtype} for ${name}`);
    out.set(name, { dtype, shape: [...info.shape], data: data as HostTensor["data"] });
  }
  return out;
}

/**
 * Opens a safetensors checkpoint (path in Node/Bun, URL, Blob/File, bytes)
 * and returns a consuming weight source for `createAgent`.
 */
export async function readWeights(source: Source, options?: OpenOptions): Promise<ConsumingWeights> {
  const file = await openSafetensors(source, options);
  try {
    return consumingWeights(await readAllTensors(file));
  } finally {
    await file.close();
  }
}
