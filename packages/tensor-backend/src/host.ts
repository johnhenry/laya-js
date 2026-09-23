import type { DType, HostData, HostTensor, Shape } from "./index.ts";

export function sizeOf(shape: Shape): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
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
  }
}

export function host(dtype: DType, shape: Shape, data?: ArrayLike<number>): HostTensor {
  const out = allocHost(dtype, sizeOf(shape));
  if (data) {
    if (data.length !== out.length) throw new RangeError(`host(): ${data.length} values for shape [${shape}]`);
    if (dtype === "bf16") {
      const u = out as Uint16Array;
      for (let i = 0; i < u.length; i++) u[i] = f32ToBf16Bits(data[i]!);
    } else out.set(data as ArrayLike<number>);
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
  return Float32Array.from(t.data as ArrayLike<number>);
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
