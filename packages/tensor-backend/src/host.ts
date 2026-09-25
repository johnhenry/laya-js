import type { DType, HostData, HostQuantized, HostTensor, QuantBits, Shape } from "./index.ts";

export function sizeOf(shape: Shape): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

/** dtypes whose HostData elements are JS `bigint`, not `number` (matches @johnhenry/math-plus-tensor-core's `isBigIntDType`). */
export function isBigIntDType(dtype: DType): boolean {
  return dtype === "u64" || dtype === "i64";
}

export function allocHost(dtype: DType, length: number): HostData {
  switch (dtype) {
    case "f32":
      return new Float32Array(length);
    case "f16":
      return new Float16Array(length);
    case "bf16":
      return new Uint16Array(length);
    case "i32":
      return new Int32Array(length);
    case "bool":
      return new Uint8Array(length);
    case "u8":
      return new Uint8Array(length);
    case "i8":
      return new Int8Array(length);
    case "u16":
      return new Uint16Array(length);
    case "i16":
      return new Int16Array(length);
    case "u32":
      return new Uint32Array(length);
    case "f64":
      return new Float64Array(length);
    case "u64":
      return new BigUint64Array(length);
    case "i64":
      return new BigInt64Array(length);
  }
}

export function host(dtype: DType, shape: Shape, data?: ArrayLike<number | bigint>): HostTensor {
  const out = allocHost(dtype, sizeOf(shape));
  if (data) {
    if (data.length !== out.length) throw new RangeError(`host(): ${data.length} values for shape [${shape}]`);
    if (dtype === "bf16") {
      const u = out as Uint16Array;
      for (let i = 0; i < u.length; i++) u[i] = f32ToBf16Bits(data[i] as number);
    } else if (isBigIntDType(dtype)) {
      const b = out as BigUint64Array | BigInt64Array;
      for (let i = 0; i < b.length; i++) b[i] = BigInt(data[i]!);
    } else (out as Exclude<HostData, BigUint64Array | BigInt64Array>).set(data as ArrayLike<number>);
  }
  return { dtype, shape: [...shape], data: out };
}

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

/** Round-to-nearest-even f32 → bf16 bits. */
export function f32ToBf16Bits(x: number): number {
  f32Scratch[0] = x;
  const u = u32Scratch[0]!;
  if ((u & 0x7fffffff) > 0x7f800000) return 0x7fc0; // NaN
  return ((u + 0x7fff + ((u >>> 16) & 1)) >>> 16) & 0xffff;
}

export function bf16BitsToF32(bits: number): number {
  u32Scratch[0] = bits << 16;
  return f32Scratch[0]!;
}

/** Any HostTensor → Float32Array of its values. */
export function toF32(t: HostTensor): Float32Array {
  if (t.dtype === "f32") return t.data as Float32Array;
  if (t.dtype === "bf16") {
    const u = t.data as Uint16Array;
    const out = new Float32Array(u.length);
    for (let i = 0; i < u.length; i++) out[i] = bf16BitsToF32(u[i]!);
    return out;
  }
  if (isBigIntDType(t.dtype)) {
    const b = t.data as BigUint64Array | BigInt64Array;
    const out = new Float32Array(b.length);
    for (let i = 0; i < b.length; i++) out[i] = Number(b[i]!);
    return out;
  }
  return Float32Array.from(t.data as ArrayLike<number>);
}

/**
 * Any HostTensor → Float64Array of its values, at full precision for f64
 * (unlike `toF32`, which would round f64 values down -- this is what the
 * conformance suite's `assertClose` uses for f64 cases so a tight tolerance
 * actually verifies f64 precision instead of comparing two already-rounded
 * f32 values).
 */
export function toF64(t: HostTensor): Float64Array {
  if (t.dtype === "f64") return t.data as Float64Array;
  if (t.dtype === "bf16") {
    const u = t.data as Uint16Array;
    const out = new Float64Array(u.length);
    for (let i = 0; i < u.length; i++) out[i] = bf16BitsToF32(u[i]!);
    return out;
  }
  if (isBigIntDType(t.dtype)) {
    const b = t.data as BigUint64Array | BigInt64Array;
    const out = new Float64Array(b.length);
    for (let i = 0; i < b.length; i++) out[i] = Number(b[i]!);
    return out;
  }
  return Float64Array.from(t.data as ArrayLike<number>);
}

/**
 * Structural bridge to @johnhenry/math-plus-tensor-core without a
 * dependency: pass the returned tuple to `Tensor.fromTypedArray`.
 * (math-plus stores f16/bf16 as Uint16Array bits.)
 */
export function toMathPlusArgs(t: HostTensor): { data: HostData | Uint16Array; shape: number[]; dtype: DType } {
  const data = t.dtype === "f16" ? new Uint16Array((t.data as Float16Array).buffer, t.data.byteOffset, t.data.length) : t.data;
  return { data, shape: [...t.shape], dtype: t.dtype };
}

// ---------------------------------------------------------------- quantized

/** Groups per row of a quantized matrix with `inFeatures` columns: ⌈in / groupSize⌉. */
export function quantGroups(inFeatures: number, groupSize: number): number {
  return Math.ceil(inFeatures / groupSize);
}

/** Throws unless `h` is a well-formed `HostQuantized` (see its docs). */
export function validateQuantized(h: HostQuantized): void {
  const [N, K] = h.shape;
  const where = `quantized [${N}, ${K}] q${h.bits}/${h.mode}/g${h.groupSize}`;
  if (h.bits !== 4 && h.bits !== 8) throw new RangeError(`${where}: bits must be 4 or 8`);
  if (h.mode !== "symmetric" && h.mode !== "affine") throw new RangeError(`${where}: mode must be "symmetric" or "affine"`);
  if (!(Number.isInteger(N) && N > 0 && Number.isInteger(K) && K > 0)) throw new RangeError(`${where}: bad shape`);
  if (!(Number.isInteger(h.groupSize) && h.groupSize > 0)) throw new RangeError(`${where}: bad group size`);
  if ((K * h.bits) % 32) throw new RangeError(`${where}: in · bits must be a multiple of 32`);
  if (h.data.length !== (N * K * h.bits) / 8) throw new RangeError(`${where}: data has ${h.data.length} bytes, want ${(N * K * h.bits) / 8}`);
  const G = quantGroups(K, h.groupSize);
  const check = (t: HostTensor | null, name: string) => {
    if (!t) throw new RangeError(`${where}: missing ${name}`);
    if (t.dtype !== "f32" && t.dtype !== "f16" && t.dtype !== "bf16") throw new TypeError(`${where}: ${name} must be float`);
    if (t.shape.length !== 2 || t.shape[0] !== N || t.shape[1] !== G) throw new RangeError(`${where}: ${name} has shape [${t.shape}], want [${N}, ${G}]`);
  };
  check(h.scales, "scales");
  if (h.mode === "affine") check(h.biases, "biases");
  else if (h.biases) throw new RangeError(`${where}: symmetric quantization takes no biases`);
}

/** The integer values q [out · in] of a packed quantized matrix (signed for "symmetric"). */
export function unpackQuantized(h: HostQuantized): Int32Array {
  const [N, K] = h.shape;
  const n = N * K, d = h.data;
  const q = new Int32Array(n);
  const signed = h.mode === "symmetric";
  if (h.bits === 8) {
    for (let i = 0; i < n; i++) q[i] = signed ? (d[i]! << 24) >> 24 : d[i]!;
  } else {
    for (let i = 0; i < n; i += 2) {
      const lo = d[i >> 1]! & 15, hi = d[i >> 1]! >> 4;
      q[i] = signed ? (lo << 28) >> 28 : lo;
      q[i + 1] = signed ? (hi << 28) >> 28 : hi;
    }
  }
  return q;
}

/** Packs integer values q [out · in] (row-major) into `HostQuantized.data` bytes. */
export function packQuantized(q: ArrayLike<number>, bits: QuantBits): Uint8Array {
  if (bits === 8) {
    const out = new Uint8Array(q.length);
    for (let i = 0; i < q.length; i++) out[i] = q[i]! & 255;
    return out;
  }
  if (q.length % 2) throw new RangeError("packQuantized: 4-bit needs an even value count");
  const out = new Uint8Array(q.length / 2);
  for (let i = 0; i < q.length; i += 2) out[i >> 1] = (q[i]! & 15) | ((q[i + 1]! & 15) << 4);
  return out;
}
