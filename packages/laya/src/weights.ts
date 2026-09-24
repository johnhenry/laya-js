/**
 * Checkpoint weights from a safetensors source, read once and handed to the
 * backend one tensor at a time. Browser-safe (no fs import of our own).
 */
import { openSafetensors, type LazySafetensors } from "@johnhenry/math-plus-safetensors";
import type { HostTensor, DType } from "@johnhenry/tensor-backend";
import type { WeightGetter } from "@johnhenry/modernbert";
import { dequantizeMatrix, groupQuantized, quantMetadata, type DequantDtype, type QuantizedMatrix, type RawTensor } from "./quant.ts";

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
 * A consuming weight source over a quantized checkpoint's tensors
 * (see quant.ts). Quantized matrices stay packed until asked for; `get`
 * dequantizes one to `dtype` (f16 or f32) and forgets the packed copy, so
 * at most one dequantized tensor exists on the host at a time (the backend
 * upload follows immediately). Plain tensors are handed out as-is.
 */
export function dequantizingWeights(tensors: ReadonlyMap<string, RawTensor>, dtype: DequantDtype = "f16"): ConsumingWeights {
  const { quantized, plain } = groupQuantized(tensors);
  const q = new Map<string, QuantizedMatrix>();
  for (const [name, m] of quantized) {
    const key = sanitizeName(name);
    if (q.has(key)) throw new Error(`Duplicate checkpoint parameter after conversion: ${key}`);
    q.set(key, m);
  }
  const hosts = new Map<string, HostTensor>();
  for (const [name, t] of plain) {
    const d = DTYPES[t.dtype];
    if (!d) throw new TypeError(`laya: unsupported safetensors dtype ${t.dtype} for ${name}`);
    hosts.set(name, { dtype: d, shape: [...t.shape], data: t.data as HostTensor["data"] });
  }
  const rest = consumingWeights(hosts);
  const plainKeys = new Set(rest.remaining());
  for (const key of q.keys()) if (plainKeys.has(key)) throw new Error(`Duplicate checkpoint parameter after conversion: ${key}`);
  return {
    get: (name) => {
      const m = q.get(name);
      if (!m) return rest.get(name);
      q.delete(name);
      return { dtype, shape: [m.rows, m.cols], data: dequantizeMatrix(m, dtype) };
    },
    remaining: () => [...q.keys(), ...rest.remaining()],
  };
}

/** Options for `readWeights`: safetensors open options plus the dequantization target. */
export type ReadWeightsOptions = OpenOptions & {
  /** Values dtype for dequantized tensors of a quantized checkpoint (default "f16"; ignored otherwise). */
  dtype?: DequantDtype;
};

/**
 * Opens a safetensors checkpoint (path in Node/Bun, URL, Blob/File, bytes)
 * and returns a consuming weight source for `createAgent`. Quantized
 * checkpoints (`laya_quant` in the file's metadata, see quant.ts) are
 * detected and dequantized tensor by tensor as the model asks for them.
 */
export async function readWeights(source: Source, options?: ReadWeightsOptions): Promise<ConsumingWeights> {
  const { dtype, ...open } = options ?? {};
  const file = await openSafetensors(source, open);
  try {
    const meta = quantMetadata(file.metadata);
    if (!meta) return consumingWeights(await readAllTensors(file));
    const views = await file.readMany();
    const raw = new Map<string, RawTensor>();
    for (const [name, data] of views) {
      const info = file.info(name);
      raw.set(name, { dtype: info.dtype, shape: info.shape, data });
    }
    return dequantizingWeights(raw, dtype ?? "f16");
  } finally {
    await file.close();
  }
}
