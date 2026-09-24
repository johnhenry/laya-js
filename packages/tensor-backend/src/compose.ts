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
 */
import type { Backend, DType, Tensor } from "./index.ts";

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
export function hasNative(b: Backend<any>, op: NumericsOp | "geglu" | "meanPool" | "compile"): boolean {
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
