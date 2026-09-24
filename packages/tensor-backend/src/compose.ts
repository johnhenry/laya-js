/**
 * Call-site helpers for the optional ops of `Backend`: each uses the
 * backend's native kernel when it has one, and a default composition from
 * required ops otherwise. Call optional ops through these (`sqrt(b, x)`,
 * not `b.sqrt!(x)`) and code runs on every backend.
 *
 * Compositions never upload: constants are derived on the device from an
 * input tensor (`zerosLike` / `onesLike`, below), so every helper stays
 * synchronous and traceable by `compile`, now that `fromHost` is async.
 *
 * Composition caveats (native kernels do not have them):
 * - Comparisons compose from `sub` + `relu` + `cast(…, "bool")`: exact for
 *   finite inputs (IEEE subtraction of distinct floats is nonzero), but
 *   NaN/±inf inputs and i32 differences that overflow give backend-defined
 *   results, and flush-to-zero hardware can call subnormal-apart values equal.
 * - `pow` composes as exp(b·log a): defined for a > 0 (and a = 0 with b > 0);
 *   a negative base gives NaN.
 * - `sqrt`/`rsqrt` compose as exp(±½·log x) (a few ulp).
 * - `erf` composes the canonical math-plus algorithm (series below 1,
 *   continued fraction above) from elementwise ops: ≈1e-7 absolute, but
 *   ~150 dispatches.
 * - `argmax`/`argmin` compose from `max`/`min`, `equal`, `where` and an iota
 *   built with `cumsum`, so they need a native `cumsum`.
 * - `cumsum` has no composition (it needs an iota or a triangular constant,
 *   which in turn need an upload or a scan): it is required-if-used.
 * - Quantized weights (`uploadQuantized` → `quantizedLinear` /
 *   `quantizedEmbedding`) compose by storing the integer values as floats and
 *   dequantizing on the device per call: correct on every backend, but it
 *   saves no memory — only the native ops keep weights packed.
 */
import type { Backend, DType, HostQuantized, HostTensor, QuantizedTensor, Tensor } from "./index.ts";
import { toF32, unpackQuantized, validateQuantized } from "./host.ts";

/** The optional "general numerics" ops (math-plus RFC 0001 §12 Q7). */
export const NUMERICS_OPS = [
  "equal", "notEqual", "less", "lessEqual", "greater", "greaterEqual",
  "logicalAnd", "logicalOr", "logicalNot",
  "sqrt", "rsqrt", "pow", "neg", "abs", "tanh", "sigmoid", "erf",
  "argmax", "argmin", "mean", "min", "cumsum",
] as const;
export type NumericsOp = (typeof NUMERICS_OPS)[number];

/** Optional ops with no default composition: a backend must implement them for callers that use them. */
export const NATIVE_ONLY_OPS: readonly NumericsOp[] = ["cumsum"];

/** Optional ops whose composition needs another op natively (`argmax`/`argmin` need `cumsum`). */
export const COMPOSITION_NEEDS: Readonly<Partial<Record<NumericsOp, readonly NumericsOp[]>>> = { argmax: ["cumsum"], argmin: ["cumsum"] };

/** Whether `b` implements optional op `op` natively (else the helper composes it). */
export function hasNative(b: Backend<any>, op: NumericsOp | QuantizedOp | "geglu" | "meanPool" | "compile"): boolean {
  return typeof (b as unknown as Record<string, unknown>)[op] === "function";
}

const isFloat = (d: DType) => d === "f32" || d === "f16" || d === "bf16";
const floatOf = (d: DType): DType => (isFloat(d) ? d : "f32");

function asFloat<T extends Tensor>(b: Backend<T>, x: T): T {
  return isFloat(x.dtype) ? x : b.cast(x, "f32");
}

/** 0/1 as f32 (nonzero-is-true), from any dtype. */
function bit<T extends Tensor>(b: Backend<T>, x: T): T {
  return b.cast(b.cast(x, "bool"), "f32");
}

/** Zeros shaped like `x` in `dtype`, computed on the device (valid for any x, including NaN/±inf). */
export function zerosLike<T extends Tensor>(b: Backend<T>, x: T, dtype: DType = x.dtype): T {
  return b.scope(() => b.cast(b.scale(bit(b, x), 0), dtype));
}

/** Ones shaped like `x` in `dtype`, computed on the device (valid for any x). */
export function onesLike<T extends Tensor>(b: Backend<T>, x: T, dtype: DType = x.dtype): T {
  return b.scope(() => b.cast(b.cast(b.exp(b.scale(bit(b, x), 0)), "bool"), dtype));
}

/** `value` broadcast to x's shape, in `dtype` (default: x's float dtype), computed on the device. */
export function fullLike<T extends Tensor>(b: Backend<T>, x: T, value: number, dtype: DType = floatOf(x.dtype)): T {
  return b.scope(() => b.cast(b.scale(onesLike(b, x, "f32"), value), dtype));
}

// ---------------------------------------------------------------- fused

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
    const n = b.sum(m, 1);
    return b.div(total, b.maximum(n, onesLike(b, n)));
  });
}

// ---------------------------------------------------------------- logical

export function logicalNot<T extends Tensor>(b: Backend<T>, x: T): T {
  if (b.logicalNot) return b.logicalNot(x);
  return b.scope(() => {
    const v = bit(b, x);
    return b.cast(b.sub(onesLike(b, v), v), "bool");
  });
}

export function logicalAnd<T extends Tensor>(b: Backend<T>, x: T, y: T): T {
  if (b.logicalAnd) return b.logicalAnd(x, y);
  return b.scope(() => b.cast(b.mul(bit(b, x), bit(b, y)), "bool"));
}

export function logicalOr<T extends Tensor>(b: Backend<T>, x: T, y: T): T {
  if (b.logicalOr) return b.logicalOr(x, y);
  return b.scope(() => b.cast(b.maximum(bit(b, x), bit(b, y)), "bool"));
}

// ---------------------------------------------------------------- comparisons

/** bool operands subtract as i32 (MLX rejects bool arithmetic). */
function diff<T extends Tensor>(b: Backend<T>, x: T, y: T): T {
  const i = (t: T) => (t.dtype === "bool" ? b.cast(t, "i32") : t);
  return b.sub(i(x), i(y));
}

export function greater<T extends Tensor>(b: Backend<T>, x: T, y: T): T {
  if (b.greater) return b.greater(x, y);
  return b.scope(() => b.cast(b.relu(diff(b, x, y)), "bool"));
}

export function less<T extends Tensor>(b: Backend<T>, x: T, y: T): T {
  if (b.less) return b.less(x, y);
  return greater(b, y, x);
}

export function notEqual<T extends Tensor>(b: Backend<T>, x: T, y: T): T {
  if (b.notEqual) return b.notEqual(x, y);
  return b.scope(() => b.cast(diff(b, x, y), "bool"));
}

export function equal<T extends Tensor>(b: Backend<T>, x: T, y: T): T {
  if (b.equal) return b.equal(x, y);
  return b.scope(() => logicalNot(b, notEqual(b, x, y)));
}

export function lessEqual<T extends Tensor>(b: Backend<T>, x: T, y: T): T {
  if (b.lessEqual) return b.lessEqual(x, y);
  return b.scope(() => logicalNot(b, greater(b, x, y)));
}

export function greaterEqual<T extends Tensor>(b: Backend<T>, x: T, y: T): T {
  if (b.greaterEqual) return b.greaterEqual(x, y);
  return b.scope(() => logicalNot(b, less(b, x, y)));
}

// ---------------------------------------------------------------- unary math

export function neg<T extends Tensor>(b: Backend<T>, x: T): T {
  if (b.neg) return b.neg(x);
  if (isFloat(x.dtype)) return b.scale(x, -1);
  return b.scope(() => b.sub(zerosLike(b, x, "i32"), b.cast(x, "i32")));
}

export function abs<T extends Tensor>(b: Backend<T>, x: T): T {
  if (b.abs) return b.abs(x);
  return b.scope(() => b.maximum(x, neg(b, x)));
}

export function sqrt<T extends Tensor>(b: Backend<T>, x: T): T {
  if (b.sqrt) return b.sqrt(x);
  return b.scope(() => b.exp(b.scale(b.log(asFloat(b, x)), 0.5)));
}

export function rsqrt<T extends Tensor>(b: Backend<T>, x: T): T {
  if (b.rsqrt) return b.rsqrt(x);
  return b.scope(() => b.exp(b.scale(b.log(asFloat(b, x)), -0.5)));
}

export function pow<T extends Tensor>(b: Backend<T>, x: T, y: T): T {
  if (b.pow) return b.pow(x, y);
  return b.scope(() => b.exp(b.mul(asFloat(b, y), b.log(asFloat(b, x)))));
}

export function sigmoid<T extends Tensor>(b: Backend<T>, x: T): T {
  if (b.sigmoid) return b.sigmoid(x);
  return b.scope(() => {
    const f = asFloat(b, x);
    const one = onesLike(b, f);
    return b.div(one, b.add(one, b.exp(neg(b, f))));
  });
}

export function tanh<T extends Tensor>(b: Backend<T>, x: T): T {
  if (b.tanh) return b.tanh(x);
  return b.scope(() => {
    const f = asFloat(b, x);
    return b.sub(b.scale(sigmoid(b, b.scale(f, 2)), 2), onesLike(b, f));
  });
}

/** Truncation of the canonical erf's f32 lowering (math-plus tensor-core ERF_F32_PARAMS). */
const ERF_SERIES_TERMS = 10, ERF_CF_DEPTH = 28;
const TWO_OVER_SQRT_PI = 1.1283791670955126;

/**
 * erf. The composition evaluates math-plus tensor-core's canonical algorithm
 * (derived from backend-cpu's double-precision erf): Maclaurin series for
 * |x| < 1, erfc by the even contraction of Laplace's continued fraction above.
 */
export function erf<T extends Tensor>(b: Backend<T>, x: T): T {
  if (b.erf) return b.erf(x);
  return b.scope(() => {
    const f = asFloat(b, x);
    const one = onesLike(b, f);
    const c = (v: number) => b.scale(one, v);
    const x2 = b.mul(f, f);
    // series: erf(x) = 2/√π · Σ (−1)ⁿ x^(2n+1) / (n!(2n+1))
    let term = f, sum = f;
    for (let n = 1; n <= ERF_SERIES_TERMS; n++) {
      term = b.mul(term, b.scale(x2, -1 / n));
      sum = b.add(sum, b.scale(term, 1 / (2 * n + 1)));
    }
    const series = b.scale(sum, TWO_OVER_SQRT_PI);
    // continued fraction on z = |x|: erfc(z) = e^(−z²)/√π · 2z / (t − 1·2/(t+4 − 3·4/(t+8 − …))), t = 2z² + 1
    const z = abs(b, f);
    const t = b.add(b.scale(x2, 2), one);
    let fr = zerosLike(b, f);
    for (let n = ERF_CF_DEPTH; n >= 1; n--) fr = b.div(c((2 * n - 1) * (2 * n)), b.sub(b.add(t, c(4 * n)), fr));
    const erfc = b.div(b.mul(b.exp(neg(b, x2)), b.scale(z, TWO_OVER_SQRT_PI)), b.sub(t, fr));
    const tail = b.sub(one, erfc);
    const signed = b.where(less(b, f, zerosLike(b, f)), neg(b, tail), tail);
    return b.where(less(b, z, one), series, signed);
  });
}

// ---------------------------------------------------------------- reductions

function axisLen(x: Tensor, axis: number): number {
  const r = x.shape.length;
  const a = axis < 0 ? axis + r : axis;
  if (a < 0 || a >= r) throw new RangeError(`axis ${axis} out of range for rank ${r}`);
  return x.shape[a]!;
}

export function mean<T extends Tensor>(b: Backend<T>, x: T, axis: number, keepDims = false): T {
  if (b.mean) return b.mean(x, axis, keepDims);
  const n = axisLen(x, axis);
  return b.scope(() => b.scale(b.sum(asFloat(b, x), axis, keepDims), 1 / n));
}

export function min<T extends Tensor>(b: Backend<T>, x: T, axis: number, keepDims = false): T {
  if (b.min) return b.min(x, axis, keepDims);
  if (x.dtype === "bool") return b.scope(() => logicalNot(b, b.max(logicalNot(b, x), axis, keepDims)));
  return b.scope(() => neg(b, b.max(neg(b, x), axis, keepDims)));
}

/** Inclusive prefix sum. No composition: throws when the backend has no native `cumsum`. */
export function cumsum<T extends Tensor>(b: Backend<T>, x: T, axis: number): T {
  if (b.cumsum) return b.cumsum(x, axis);
  throw new Error(`cumsum: the ${b.name} backend has no native cumsum and there is no default composition`);
}

function argReduce<T extends Tensor>(b: Backend<T>, x: T, axis: number, keepDims: boolean, which: "argmax" | "argmin"): T {
  if (!b.cumsum) throw new Error(`${which}: the ${b.name} backend has neither a native ${which} nor a native cumsum to compose it from`);
  const n = axisLen(x, axis);
  return b.scope(() => {
    const best = which === "argmax" ? b.max(x, axis, true) : min(b, x, axis, true);
    const ones = onesLike(b, x, "i32");
    const iota = b.sub(cumsum(b, ones, axis), ones);
    const cand = b.where(equal(b, x, best), iota, b.cast(b.scale(onesLike(b, x, "f32"), n), "i32"));
    return b.cast(min(b, cand, axis, keepDims), "i32");
  });
}

export function argmax<T extends Tensor>(b: Backend<T>, x: T, axis: number, keepDims = false): T {
  if (b.argmax) return b.argmax(x, axis, keepDims);
  return argReduce(b, x, axis, keepDims, "argmax");
}

export function argmin<T extends Tensor>(b: Backend<T>, x: T, axis: number, keepDims = false): T {
  if (b.argmin) return b.argmin(x, axis, keepDims);
  return argReduce(b, x, axis, keepDims, "argmin");
}

// ---------------------------------------------------------------- quantized

/** The optional quantized-weight ops (a backend implements all three or none). */
export const QUANTIZED_OPS = ["fromHostQuantized", "quantizedLinear", "quantizedEmbedding"] as const;
export type QuantizedOp = (typeof QUANTIZED_OPS)[number];

/** Whether `b` keeps quantized weights quantized on the device (has the native quantized ops). */
export function hasNativeQuantized(b: Backend<any>): boolean {
  return QUANTIZED_OPS.every((op) => typeof (b as unknown as Record<string, unknown>)[op] === "function");
}

/** Whether `x` is a `QuantizedTensor` (from `uploadQuantized`) rather than a plain tensor. */
export function isQuantized<T extends Tensor>(x: T | QuantizedTensor<T> | null | undefined): x is QuantizedTensor<T> {
  return !!x && typeof x === "object" && "scales" in x && "native" in x;
}

/** Uploads a float host tensor and casts it to `dtype` on the device (when it differs). */
async function uploadFloat<T extends Tensor>(b: Backend<T>, h: HostTensor, dtype: DType): Promise<T> {
  const t = await b.fromHost(h.dtype === "f32" || b.supports(h.dtype) ? h : { dtype: "f32", shape: h.shape, data: toF32(h) });
  if (t.dtype === dtype) return t;
  const c = b.cast(t, dtype);
  b.dispose(t);
  return c;
}

/**
 * Uploads a quantized matrix. With the backend's native quantized ops
 * (`fromHostQuantized`) it stays packed on the device (bits/8 bytes per value
 * plus the per-group parameters). Otherwise — or when the backend declines
 * this configuration — the default composition uploads the integer values
 * as `dtype` floats [out, in] and the scales/biases as `dtype`, and
 * `quantizedLinear` dequantizes on the device per call: correct everywhere,
 * but no memory saving (a caller that only wants the float matrix should
 * dequantize on the host instead). `dtype` is the float dtype the values
 * dequantize to (`quantizedEmbedding` results; `quantizedLinear` follows x).
 */
export async function uploadQuantized<T extends Tensor>(b: Backend<T>, h: HostQuantized, dtype: DType): Promise<QuantizedTensor<T>> {
  validateQuantized(h);
  if (!isFloat(dtype)) throw new TypeError(`uploadQuantized: dtype must be float, got ${dtype}`);
  if (b.fromHostQuantized && b.quantizedLinear && b.quantizedEmbedding) {
    const q = await b.fromHostQuantized(h, dtype);
    if (q) return q;
  }
  const [N, K] = h.shape;
  const q = unpackQuantized(h);
  const jobs = [
    uploadFloat(b, { dtype: "f32", shape: [N, K], data: Float32Array.from(q) }, dtype),
    uploadFloat(b, h.scales, dtype),
    ...(h.mode === "affine" ? [uploadFloat(b, h.biases!, dtype)] : []),
  ];
  const r = await Promise.allSettled(jobs);
  const bad = r.find((x): x is PromiseRejectedResult => x.status === "rejected");
  const ok = r.flatMap((x) => (x.status === "fulfilled" ? [x.value] : []));
  if (bad) {
    for (const t of ok) b.dispose(t);
    throw bad.reason;
  }
  const { bits, groupSize, mode } = h;
  return { shape: [N, K], bits, groupSize, mode, dtype, native: false, w: ok[0]!, scales: ok[1]!, biases: ok[2] ?? null };
}

/** Frees a `QuantizedTensor`'s device tensors. */
export function disposeQuantized<T extends Tensor>(b: Backend<T>, q: QuantizedTensor<T>): void {
  b.dispose(q.w);
  b.dispose(q.scales);
  if (q.biases) b.dispose(q.biases);
}

/**
 * Compose layout: rows of q values as floats w [R, in], per-group scales /
 * biases [R, G] → dequantized [R, in] in `dtype`, computed in f32 and rounded
 * once. A partial last group is handled separately.
 */
function dequantRows<T extends Tensor>(b: Backend<T>, w: T, scales: T, biases: T | null, groupSize: number, dtype: DType): T {
  return b.scope(() => {
    const [R, K] = w.shape as [number, number];
    const G = Math.ceil(K / groupSize), full = Math.floor(K / groupSize);
    const f = (t: T) => b.cast(t, "f32");
    const part = (w0: number, w1: number, g0: number, g1: number): T => {
      const n = g1 - g0, gs = (w1 - w0) / n;
      const wq = b.reshape(f(b.slice(w, [0, w0], [R, w1])), [R, n, gs]);
      const s = b.reshape(f(b.slice(scales, [0, g0], [R, g1])), [R, n, 1]);
      let v = b.mul(wq, s);
      if (biases) v = b.add(v, b.reshape(f(b.slice(biases, [0, g0], [R, g1])), [R, n, 1]));
      return b.reshape(v, [R, w1 - w0]);
    };
    const pieces: T[] = [];
    if (full) pieces.push(part(0, full * groupSize, 0, full));
    if (G > full) pieces.push(part(full * groupSize, K, full, G));
    const v = pieces.length === 1 ? pieces[0]! : b.concat(pieces, 1);
    return v.dtype === dtype ? v : b.cast(v, dtype);
  });
}

/** Dequantized matrix [out, in] in `q.dtype` (compose-layout tensors only). */
export function dequantize<T extends Tensor>(b: Backend<T>, q: QuantizedTensor<T>): T {
  if (q.native) throw new Error(`dequantize: ${b.name} keeps this matrix in its native layout; use quantizedLinear / quantizedEmbedding`);
  return dequantRows(b, q.w, q.scales, q.biases, q.groupSize, q.dtype);
}

/** y = x · dequant(q)ᵀ (+ bias), in x's dtype, f32 accumulation. */
export function quantizedLinear<T extends Tensor>(b: Backend<T>, x: T, q: QuantizedTensor<T>, bias?: T | null): T {
  if (x.shape[x.shape.length - 1] !== q.shape[1]) throw new Error(`quantizedLinear: x [${x.shape}] vs w [${q.shape}]`);
  if (q.native) {
    if (!b.quantizedLinear) throw new Error(`quantizedLinear: the ${b.name} backend has no native quantizedLinear for its native-layout weight`);
    return b.quantizedLinear(x, q.w, q.scales, q.biases, q, bias ?? null);
  }
  return b.scope(() => b.linear(x, dequantRows(b, q.w, q.scales, q.biases, q.groupSize, isFloat(x.dtype) ? x.dtype : "f32"), bias ?? null));
}

/** Row gather of dequant(q) by ids i32 [...] → [..., in] in `q.dtype`. */
export function quantizedEmbedding<T extends Tensor>(b: Backend<T>, q: QuantizedTensor<T>, ids: T): T {
  if (q.native) {
    if (!b.quantizedEmbedding) throw new Error(`quantizedEmbedding: the ${b.name} backend has no native quantizedEmbedding for its native-layout weight`);
    return b.quantizedEmbedding(q.w, q.scales, q.biases, q, ids);
  }
  return b.scope(() => {
    const n = ids.shape.reduce((a, d) => a * d, 1);
    const flat = b.reshape(ids, [n]);
    const rows = dequantRows(b, b.embedding(q.w, flat), b.embedding(q.scales, flat), q.biases ? b.embedding(q.biases, flat) : null, q.groupSize, q.dtype);
    return b.reshape(rows, [...ids.shape, q.shape[1]]);
  });
}

/** `linear` for a weight that may be quantized: plain tensors go to `b.linear`. */
export function linearAny<T extends Tensor>(b: Backend<T>, x: T, w: T | QuantizedTensor<T>, bias?: T | null): T {
  return isQuantized(w) ? quantizedLinear(b, x, w, bias) : b.linear(x, w, bias ?? null);
}

/** `embedding` for a table that may be quantized. */
export function embeddingAny<T extends Tensor>(b: Backend<T>, table: T | QuantizedTensor<T>, ids: T): T {
  return isQuantized(table) ? quantizedEmbedding(b, table, ids) : b.embedding(table, ids);
}
