/**
 * @johnhenry/backend-cpu — pure-TypeScript f32 reference implementation of
 * the @johnhenry/tensor-backend `Backend` contract.
 *
 * - Float tensors are always stored as Float32Array (f16/bf16 host data is
 *   widened on `fromHost`); i32 → Int32Array; bool → Uint8Array (0/1).
 *   `supports("f16" | "bf16")` is false and `cast` to them throws.
 * - Contiguous row-major storage; `reshape` and same-dtype `cast` share the
 *   buffer (tensors are immutable, so sharing is safe).
 * - Evaluation is eager and synchronous; `read` resolves immediately.
 * - Reductions, softmax, LayerNorm and matmul accumulate in f64.
 */
import type { Backend, DType, HostTensor, Shape, Tensor } from "@johnhenry/tensor-backend";
import { sizeOf } from "@johnhenry/tensor-backend";
import { geluScalar } from "./erf.ts";
import { gemmNT, transpose2d } from "./gemm.ts";

export { erf, erfc, geluScalar } from "./erf.ts";
export { gemmNT } from "./gemm.ts";

type CpuDType = "f32" | "i32" | "bool";
type Data = Float32Array | Int32Array | Uint8Array;

export class CpuTensor implements Tensor {
  readonly shape: readonly number[];
  readonly dtype: DType;
  #data: Data | null;
  constructor(shape: readonly number[], dtype: CpuDType, data: Data) {
    this.shape = shape;
    this.dtype = dtype;
    this.#data = data;
  }
  /** Backing storage (row-major, possibly shared with other tensors: never mutate). Throws after dispose. */
  get data(): Data {
    const d = this.#data;
    if (d === null) throw new Error("backend-cpu: tensor used after dispose");
    return d;
  }
  get disposed(): boolean {
    return this.#data === null;
  }
  /** @internal */
  _release(): void {
    this.#data = null;
  }
}

// ---------------------------------------------------------------- helpers
function normAxis(axis: number, rank: number): number {
  const a = axis < 0 ? axis + rank : axis;
  if (a < 0 || a >= rank) throw new RangeError(`backend-cpu: axis ${axis} out of range for rank ${rank}`);
  return a;
}

function stridesOf(shape: readonly number[]): number[] {
  const s = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= shape[i]!;
  }
  return s;
}

function broadcastShapes(...shapes: (readonly number[])[]): number[] {
  const rank = Math.max(0, ...shapes.map((s) => s.length));
  const out = new Array<number>(rank).fill(1);
  for (const s of shapes) {
    for (let i = 0; i < s.length; i++) {
      const o = rank - s.length + i;
      const d = s[i]!;
      if (d === 1) continue;
      if (out[o] === 1) out[o] = d;
      else if (out[o] !== d) throw new RangeError(`backend-cpu: cannot broadcast ${shapes.map((x) => `[${x}]`).join(", ")}`);
    }
  }
  return out;
}

/** Strides of `shape` aligned to `outShape` (0 on broadcast axes). */
function broadcastStrides(shape: readonly number[], outShape: readonly number[]): number[] {
  const st = stridesOf(shape);
  const out = new Array<number>(outShape.length).fill(0);
  for (let i = 0; i < shape.length; i++) {
    const o = outShape.length - shape.length + i;
    out[o] = shape[i] === 1 ? 0 : st[i]!;
  }
  return out;
}

/** For each "row" (all axes but the last) of outShape, the base offset into a tensor with `strides`. */
function rowOffsets(outShape: readonly number[], strides: readonly number[]): Int32Array {
  const rank = outShape.length;
  if (rank <= 1) return new Int32Array(1);
  let rows = 1;
  for (let i = 0; i < rank - 1; i++) rows *= outShape[i]!;
  const offs = new Int32Array(rows);
  const idx = new Array<number>(rank - 1).fill(0);
  let off = 0;
  for (let r = 0; r < rows; r++) {
    offs[r] = off;
    for (let d = rank - 2; d >= 0; d--) {
      idx[d]!++;
      off += strides[d]!;
      if (idx[d]! < outShape[d]!) break;
      off -= strides[d]! * outShape[d]!;
      idx[d] = 0;
    }
  }
  return offs;
}

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function alloc(dtype: CpuDType, n: number): Data {
  return dtype === "f32" ? new Float32Array(n) : dtype === "i32" ? new Int32Array(n) : new Uint8Array(n);
}

const ADD = 0, SUB = 1, MUL = 2, DIV = 3, MAX = 4;

function binaryKernel(op: number, A: Data, oa: number, sa: number, B: Data, ob: number, sb: number, O: Data, oo: number, n: number): void {
  switch (op) {
    case ADD: for (let j = 0; j < n; j++) O[oo + j] = A[oa + j * sa]! + B[ob + j * sb]!; break;
    case SUB: for (let j = 0; j < n; j++) O[oo + j] = A[oa + j * sa]! - B[ob + j * sb]!; break;
    case MUL: for (let j = 0; j < n; j++) O[oo + j] = A[oa + j * sa]! * B[ob + j * sb]!; break;
    case DIV: for (let j = 0; j < n; j++) O[oo + j] = A[oa + j * sa]! / B[ob + j * sb]!; break;
    default:
      for (let j = 0; j < n; j++) {
        const x = A[oa + j * sa]!, y = B[ob + j * sb]!;
        O[oo + j] = x !== x || y !== y ? NaN : x > y ? x : y;
      }
  }
}

// ---------------------------------------------------------------- backend
class CpuBackend implements Backend<CpuTensor> {
  readonly name = "cpu";
  readonly #scopes: Set<CpuTensor>[] = [];

  supports(dtype: DType): boolean {
    return dtype === "f32" || dtype === "i32" || dtype === "bool";
  }

  #make(shape: readonly number[], dtype: CpuDType, data: Data): CpuTensor {
    const t = new CpuTensor(Object.freeze([...shape]), dtype, data);
    const top = this.#scopes[this.#scopes.length - 1];
    if (top) top.add(t);
    return t;
  }

  // ---- transfer / lifetime
  fromHost(t: HostTensor): CpuTensor {
    const n = sizeOf(t.shape);
    if (t.data.length !== n) throw new RangeError(`backend-cpu: fromHost ${t.data.length} values for shape [${t.shape}]`);
    switch (t.dtype) {
      case "f32":
        return this.#make(t.shape, "f32", (t.data as Float32Array).slice());
      case "f16":
        return this.#make(t.shape, "f32", new Float32Array(t.data as Float16Array));
      case "bf16": {
        const u = t.data as Uint16Array;
        const bits = new Uint32Array(n);
        for (let i = 0; i < n; i++) bits[i] = u[i]! << 16;
        return this.#make(t.shape, "f32", new Float32Array(bits.buffer));
      }
      case "i32":
        return this.#make(t.shape, "i32", Int32Array.from(t.data as Int32Array));
      case "bool": {
        const s = t.data as Uint8Array, out = new Uint8Array(n);
        for (let i = 0; i < n; i++) out[i] = s[i] ? 1 : 0;
        return this.#make(t.shape, "bool", out);
      }
    }
  }

  async read(t: CpuTensor): Promise<HostTensor> {
    return { dtype: t.dtype, shape: [...t.shape], data: t.data.slice() };
  }

  dispose(t: CpuTensor): void {
    t._release();
  }

  scope<R>(fn: () => R): R {
    const set = new Set<CpuTensor>();
    this.#scopes.push(set);
    let result: R;
    try {
      result = fn();
    } catch (e) {
      this.#scopes.pop();
      for (const t of set) t._release();
      throw e;
    }
    this.#scopes.pop();
    const keep = new Set<CpuTensor>();
    const visit = (v: unknown) => {
      if (v instanceof CpuTensor) keep.add(v);
    };
    if (result instanceof CpuTensor) keep.add(result);
    else if (Array.isArray(result)) result.forEach(visit);
    else if (result && typeof result === "object") Object.values(result as object).forEach(visit);
    const parent = this.#scopes[this.#scopes.length - 1];
    for (const t of set) {
      if (keep.has(t)) parent?.add(t);
      else t._release();
    }
    return result;
  }

  flush(): void {}

  destroy(): void {
    for (const s of this.#scopes) for (const t of s) t._release();
    this.#scopes.length = 0;
  }

  // ---- shape
  reshape(x: CpuTensor, shape: Shape): CpuTensor {
    const n = sizeOf(x.shape);
    const out = [...shape];
    const neg = out.indexOf(-1);
    if (neg >= 0) {
      let known = 1;
      out.forEach((d, i) => { if (i !== neg) known *= d; });
      out[neg] = n / known;
    }
    if (sizeOf(out) !== n) throw new RangeError(`backend-cpu: reshape [${x.shape}] -> [${shape}]`);
    return this.#make(out, x.dtype as CpuDType, x.data);
  }

  transpose(x: CpuTensor, perm: readonly number[]): CpuTensor {
    const rank = x.shape.length;
    if (perm.length !== rank) throw new RangeError(`backend-cpu: transpose perm [${perm}] for rank ${rank}`);
    const p = perm.map((a) => normAxis(a, rank));
    const outShape = p.map((a) => x.shape[a]!);
    const inStr = stridesOf(x.shape);
    const src = x.data;
    const out = alloc(x.dtype as CpuDType, src.length);
    if (rank === 0 || src.length === 0) {
      out.set(src);
      return this.#make(outShape, x.dtype as CpuDType, out);
    }
    const pStr = p.map((a) => inStr[a]!);
    const offs = rowOffsets(outShape, pStr);
    const n = outShape[rank - 1]!, s = pStr[rank - 1]!;
    for (let r = 0; r < offs.length; r++) {
      const o = offs[r]!, oo = r * n;
      if (s === 1) out.set(src.subarray(o, o + n), oo);
      else for (let j = 0; j < n; j++) out[oo + j] = src[o + j * s]!;
    }
    return this.#make(outShape, x.dtype as CpuDType, out);
  }

  slice(x: CpuTensor, begin: readonly number[], end: readonly number[]): CpuTensor {
    const rank = x.shape.length;
    const b: number[] = [], outShape: number[] = [];
    for (let i = 0; i < rank; i++) {
      const d = x.shape[i]!;
      let lo = i < begin.length ? begin[i]! : 0;
      let hi = i < end.length ? end[i]! : d;
      if (lo < 0) lo += d;
      if (hi < 0) hi += d;
      lo = Math.min(Math.max(lo, 0), d);
      hi = Math.min(Math.max(hi, lo), d);
      b.push(lo);
      outShape.push(hi - lo);
    }
    const inStr = stridesOf(x.shape);
    let base = 0;
    for (let i = 0; i < rank; i++) base += b[i]! * inStr[i]!;
    const src = x.data;
    const out = alloc(x.dtype as CpuDType, sizeOf(outShape));
    if (out.length === 0) return this.#make(outShape, x.dtype as CpuDType, out);
    if (rank === 0) {
      out[0] = src[0]!;
      return this.#make(outShape, x.dtype as CpuDType, out);
    }
    const offs = rowOffsets(outShape, inStr);
    const n = outShape[rank - 1]!;
    for (let r = 0; r < offs.length; r++) {
      const o = base + offs[r]!;
      out.set(src.subarray(o, o + n), r * n);
    }
    return this.#make(outShape, x.dtype as CpuDType, out);
  }

  split(x: CpuTensor, parts: number, axis: number): CpuTensor[] {
    const a = normAxis(axis, x.shape.length);
    const d = x.shape[a]!;
    if (d % parts) throw new RangeError(`backend-cpu: cannot split ${d} into ${parts}`);
    const step = d / parts;
    const res: CpuTensor[] = [];
    for (let p = 0; p < parts; p++) {
      const begin = x.shape.map((_, i) => (i === a ? p * step : 0));
      const end = x.shape.map((s, i) => (i === a ? (p + 1) * step : s));
      res.push(this.slice(x, begin, end));
    }
    return res;
  }

  concat(xs: readonly CpuTensor[], axis: number): CpuTensor {
    if (xs.length === 0) throw new RangeError("backend-cpu: concat of nothing");
    const first = xs[0]!;
    const rank = first.shape.length;
    const a = normAxis(axis, rank);
    let dtype = first.dtype as CpuDType;
    for (const x of xs) {
      if (x.shape.length !== rank) throw new RangeError("backend-cpu: concat rank mismatch");
      for (let i = 0; i < rank; i++) if (i !== a && x.shape[i] !== first.shape[i]) throw new RangeError("backend-cpu: concat shape mismatch");
      if (x.dtype === "f32") dtype = "f32";
    }
    const outShape = [...first.shape];
    outShape[a] = xs.reduce((s, x) => s + x.shape[a]!, 0);
    let outer = 1;
    for (let i = 0; i < a; i++) outer *= outShape[i]!;
    let inner = 1;
    for (let i = a + 1; i < rank; i++) inner *= outShape[i]!;
    const out = alloc(dtype, sizeOf(outShape));
    const rowLen = outShape[a]! * inner;
    let colOff = 0;
    for (const x of xs) {
      const chunk = x.shape[a]! * inner;
      const src = x.data;
      for (let o = 0; o < outer; o++) {
        const s = o * chunk, d = o * rowLen + colOff;
        for (let j = 0; j < chunk; j++) out[d + j] = src[s + j]!;
      }
      colOff += chunk;
    }
    return this.#make(outShape, dtype, out);
  }

  cast(x: CpuTensor, dtype: DType): CpuTensor {
    if (dtype === "f16" || dtype === "bf16") throw new TypeError(`backend-cpu: ${dtype} is not supported (f32 reference backend)`);
    if (dtype === x.dtype) return this.#make(x.shape, dtype, x.data);
    const src = x.data;
    const n = src.length;
    if (dtype === "f32") return this.#make(x.shape, "f32", Float32Array.from(src));
    if (dtype === "i32") {
      const out = new Int32Array(n);
      for (let i = 0; i < n; i++) out[i] = Math.trunc(src[i]!);
      return this.#make(x.shape, "i32", out);
    }
    const out = new Uint8Array(n);
    for (let i = 0; i < n; i++) out[i] = src[i] !== 0 ? 1 : 0;
    return this.#make(x.shape, "bool", out);
  }

  // ---- elementwise
  #binary(a: CpuTensor, b: CpuTensor, op: number): CpuTensor {
    const dtype: CpuDType = a.dtype === "f32" || b.dtype === "f32" || op === DIV ? "f32" : "i32";
    const A = a.data, B = b.data;
    if (sameShape(a.shape, b.shape)) {
      const out = alloc(dtype, A.length);
      binaryKernel(op, A, 0, 1, B, 0, 1, out, 0, A.length);
      return this.#make(a.shape, dtype, out);
    }
    const outShape = broadcastShapes(a.shape, b.shape);
    const out = alloc(dtype, sizeOf(outShape));
    if (out.length === 0) return this.#make(outShape, dtype, out);
    const rank = outShape.length;
    const sa = broadcastStrides(a.shape, outShape), sb = broadcastStrides(b.shape, outShape);
    const oa = rowOffsets(outShape, sa), ob = rowOffsets(outShape, sb);
    const n = rank ? outShape[rank - 1]! : 1;
    const la = rank ? sa[rank - 1]! : 0, lb = rank ? sb[rank - 1]! : 0;
    for (let r = 0; r < oa.length; r++) binaryKernel(op, A, oa[r]!, la, B, ob[r]!, lb, out, r * n, n);
    return this.#make(outShape, dtype, out);
  }

  add(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#binary(a, b, ADD); }
  sub(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#binary(a, b, SUB); }
  mul(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#binary(a, b, MUL); }
  div(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#binary(a, b, DIV); }
  maximum(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#binary(a, b, MAX); }

  where(cond: CpuTensor, a: CpuTensor, b: CpuTensor): CpuTensor {
    const outShape = broadcastShapes(cond.shape, a.shape, b.shape);
    const dtype: CpuDType = a.dtype === "f32" || b.dtype === "f32" ? "f32" : (a.dtype as CpuDType);
    const out = alloc(dtype, sizeOf(outShape));
    if (out.length === 0) return this.#make(outShape, dtype, out);
    const rank = outShape.length;
    const sc = broadcastStrides(cond.shape, outShape), sa = broadcastStrides(a.shape, outShape), sb = broadcastStrides(b.shape, outShape);
    const oc = rowOffsets(outShape, sc), oa = rowOffsets(outShape, sa), ob = rowOffsets(outShape, sb);
    const n = rank ? outShape[rank - 1]! : 1;
    const lc = rank ? sc[rank - 1]! : 0, la = rank ? sa[rank - 1]! : 0, lb = rank ? sb[rank - 1]! : 0;
    const C = cond.data, A = a.data, B = b.data;
    for (let r = 0; r < oc.length; r++) {
      const c0 = oc[r]!, a0 = oa[r]!, b0 = ob[r]!, o0 = r * n;
      for (let j = 0; j < n; j++) out[o0 + j] = C[c0 + j * lc] ? A[a0 + j * la]! : B[b0 + j * lb]!;
    }
    return this.#make(outShape, dtype, out);
  }

  scale(x: CpuTensor, s: number): CpuTensor {
    const src = x.data, out = new Float32Array(src.length);
    for (let i = 0; i < out.length; i++) out[i] = src[i]! * s;
    return this.#make(x.shape, "f32", out);
  }
  exp(x: CpuTensor): CpuTensor {
    const src = x.data, out = new Float32Array(src.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.exp(src[i]!);
    return this.#make(x.shape, "f32", out);
  }
  log(x: CpuTensor): CpuTensor {
    const src = x.data, out = new Float32Array(src.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.log(src[i]!);
    return this.#make(x.shape, "f32", out);
  }
  relu(x: CpuTensor): CpuTensor {
    const src = x.data, out = new Float32Array(src.length);
    for (let i = 0; i < out.length; i++) {
      const v = src[i]!;
      out[i] = v > 0 ? v : 0;
    }
    return this.#make(x.shape, "f32", out);
  }
  gelu(x: CpuTensor): CpuTensor {
    const src = x.data, out = new Float32Array(src.length);
    for (let i = 0; i < out.length; i++) out[i] = geluScalar(src[i]!);
    return this.#make(x.shape, "f32", out);
  }

  // ---- reductions
  #axes(x: CpuTensor, axis: number): { a: number; outer: number; len: number; inner: number } {
    const rank = x.shape.length;
    const a = normAxis(axis, rank);
    let outer = 1, inner = 1;
    for (let i = 0; i < a; i++) outer *= x.shape[i]!;
    for (let i = a + 1; i < rank; i++) inner *= x.shape[i]!;
    return { a, outer, len: x.shape[a]!, inner };
  }

  #reduce(x: CpuTensor, axis: number, keepDims: boolean, isMax: boolean): CpuTensor {
    const { a, outer, len, inner } = this.#axes(x, axis);
    const outShape = keepDims ? x.shape.map((d, i) => (i === a ? 1 : d)) : x.shape.filter((_, i) => i !== a);
    const dtype: CpuDType = isMax ? (x.dtype as CpuDType) : x.dtype === "f32" ? "f32" : "i32";
    const out = alloc(dtype, outer * inner);
    const src = x.data;
    const acc = new Float64Array(inner);
    for (let o = 0; o < outer; o++) {
      const base = o * len * inner;
      acc.fill(isMax ? -Infinity : 0);
      for (let j = 0; j < len; j++) {
        const row = base + j * inner;
        if (isMax) {
          for (let i = 0; i < inner; i++) {
            const v = src[row + i]!;
            if (v > acc[i]! || v !== v) acc[i] = v;
          }
        } else {
          for (let i = 0; i < inner; i++) acc[i]! += src[row + i]!;
        }
      }
      out.set(acc, o * inner);
    }
    return this.#make(outShape, dtype, out);
  }

  sum(x: CpuTensor, axis: number, keepDims = false): CpuTensor { return this.#reduce(x, axis, keepDims, false); }
  max(x: CpuTensor, axis: number, keepDims = false): CpuTensor { return this.#reduce(x, axis, keepDims, true); }

  softmax(x: CpuTensor, axis: number): CpuTensor {
    const { outer, len, inner } = this.#axes(x, axis);
    const src = x.data;
    const out = new Float32Array(src.length);
    const tmp = new Float64Array(len);
    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < inner; i++) {
        const base = o * len * inner + i;
        let m = -Infinity;
        for (let j = 0; j < len; j++) {
          const v = src[base + j * inner]!;
          if (v > m) m = v;
        }
        let s = 0;
        for (let j = 0; j < len; j++) {
          const e = Math.exp(src[base + j * inner]! - m);
          tmp[j] = e;
          s += e;
        }
        const inv = 1 / s;
        for (let j = 0; j < len; j++) out[base + j * inner] = tmp[j]! * inv;
      }
    }
    return this.#make(x.shape, "f32", out);
  }

  sort(x: CpuTensor, axis: number): CpuTensor {
    const { outer, len, inner } = this.#axes(x, axis);
    const src = x.data;
    const out = alloc(x.dtype as CpuDType, src.length);
    const lane = alloc(x.dtype as CpuDType, len);
    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < inner; i++) {
        const base = o * len * inner + i;
        for (let j = 0; j < len; j++) lane[j] = src[base + j * inner]!;
        lane.sort();
        for (let j = 0; j < len; j++) out[base + j * inner] = lane[j]!;
      }
    }
    return this.#make(x.shape, x.dtype as CpuDType, out);
  }

  // ---- linear algebra & NN
  #f32(x: CpuTensor, op: string): Float32Array {
    const d = x.data;
    if (!(d instanceof Float32Array)) throw new TypeError(`backend-cpu: ${op} needs a float tensor, got ${x.dtype}`);
    return d;
  }

  matmul(a: CpuTensor, b: CpuTensor): CpuTensor {
    if (a.shape.length < 2 || b.shape.length < 2) throw new RangeError("backend-cpu: matmul needs rank >= 2");
    const A = this.#f32(a, "matmul"), Bd = this.#f32(b, "matmul");
    const m = a.shape[a.shape.length - 2]!, k = a.shape[a.shape.length - 1]!;
    const k2 = b.shape[b.shape.length - 2]!, n = b.shape[b.shape.length - 1]!;
    if (k !== k2) throw new RangeError(`backend-cpu: matmul [${a.shape}] @ [${b.shape}]`);
    const batchA = a.shape.slice(0, -2), batchB = b.shape.slice(0, -2);
    const batch = broadcastShapes(batchA, batchB);
    const outShape = [...batch, m, n];
    const out = new Float32Array(sizeOf(outShape));
    const nb = sizeOf(batch);
    // matrix index (not element offset) of each batch entry in a and b
    const offA = rowOffsets([...batch, 1], [...broadcastStrides(batchA, batch), 0]);
    const offB = rowOffsets([...batch, 1], [...broadcastStrides(batchB, batch), 0]);
    const bt = new Float32Array(k * n);
    let lastB = -1;
    for (let i = 0; i < nb; i++) {
      const ob = offB[i]!;
      if (ob !== lastB) {
        transpose2d(Bd, ob * k * n, k, n, bt);
        lastB = ob;
      }
      gemmNT(A, offA[i]! * m * k, k, bt, 0, k, out, i * m * n, n, m, n, k);
    }
    return this.#make(outShape, "f32", out);
  }

  linear(x: CpuTensor, w: CpuTensor, b?: CpuTensor | null): CpuTensor {
    const X = this.#f32(x, "linear"), W = this.#f32(w, "linear");
    if (w.shape.length !== 2) throw new RangeError("backend-cpu: linear weight must be [out, in]");
    const [nOut, nIn] = w.shape as [number, number];
    if (x.shape[x.shape.length - 1] !== nIn) throw new RangeError(`backend-cpu: linear [${x.shape}] with weight [${w.shape}]`);
    const rows = X.length / nIn;
    const out = new Float32Array(rows * nOut);
    gemmNT(X, 0, nIn, W, 0, nIn, out, 0, nOut, rows, nOut, nIn);
    if (b) {
      const bias = this.#f32(b, "linear");
      for (let r = 0; r < rows; r++) {
        const o = r * nOut;
        for (let j = 0; j < nOut; j++) out[o + j]! += bias[j]!;
      }
    }
    return this.#make([...x.shape.slice(0, -1), nOut], "f32", out);
  }

  layerNorm(x: CpuTensor, weight: CpuTensor | null, bias: CpuTensor | null, eps: number): CpuTensor {
    const X = this.#f32(x, "layerNorm");
    const D = x.shape[x.shape.length - 1]!;
    const W = weight ? this.#f32(weight, "layerNorm") : null;
    const Bb = bias ? this.#f32(bias, "layerNorm") : null;
    const rows = X.length / D;
    const out = new Float32Array(X.length);
    for (let r = 0; r < rows; r++) {
      const o = r * D;
      let mean = 0;
      for (let j = 0; j < D; j++) mean += X[o + j]!;
      mean /= D;
      let v = 0;
      for (let j = 0; j < D; j++) {
        const d = X[o + j]! - mean;
        v += d * d;
      }
      const inv = 1 / Math.sqrt(v / D + eps);
      if (W && Bb) for (let j = 0; j < D; j++) out[o + j] = (X[o + j]! - mean) * inv * W[j]! + Bb[j]!;
      else if (W) for (let j = 0; j < D; j++) out[o + j] = (X[o + j]! - mean) * inv * W[j]!;
      else if (Bb) for (let j = 0; j < D; j++) out[o + j] = (X[o + j]! - mean) * inv + Bb[j]!;
      else for (let j = 0; j < D; j++) out[o + j] = (X[o + j]! - mean) * inv;
    }
    return this.#make(x.shape, "f32", out);
  }

  embedding(table: CpuTensor, ids: CpuTensor): CpuTensor {
    const T = this.#f32(table, "embedding");
    const [V, D] = table.shape as [number, number];
    const I = ids.data;
    const out = new Float32Array(I.length * D);
    for (let i = 0; i < I.length; i++) {
      const id = I[i]!;
      if (id < 0 || id >= V) throw new RangeError(`backend-cpu: embedding id ${id} out of range [0, ${V})`);
      out.set(T.subarray(id * D, id * D + D), i * D);
    }
    return this.#make([...ids.shape, D], "f32", out);
  }

  gatherRows(x: CpuTensor, idx: CpuTensor): CpuTensor {
    const X = this.#f32(x, "gatherRows");
    const [B, L, D] = x.shape as [number, number, number];
    const [B2, M] = idx.shape as [number, number];
    if (B2 !== B) throw new RangeError(`backend-cpu: gatherRows [${x.shape}] idx [${idx.shape}]`);
    const I = idx.data;
    const out = new Float32Array(B * M * D);
    for (let b = 0; b < B; b++) {
      for (let m = 0; m < M; m++) {
        let r = I[b * M + m]!;
        if (r < 0) r += L;
        if (r < 0 || r >= L) throw new RangeError(`backend-cpu: gatherRows index ${I[b * M + m]} out of range`);
        const s = (b * L + r) * D;
        out.set(X.subarray(s, s + D), (b * M + m) * D);
      }
    }
    return this.#make([B, M, D], "f32", out);
  }

  rope(x: CpuTensor, base: number): CpuTensor {
    const X = this.#f32(x, "rope");
    const [B, H, L, Dh] = x.shape as [number, number, number, number];
    const half = Dh >> 1;
    // Emulates the f32 angle computation of MLX / PyTorch (inv_freq and pos·inv_freq rounded to f32).
    const lb = Math.fround(Math.log2(base));
    const cos = new Float64Array(L * half), sin = new Float64Array(L * half);
    for (let i = 0; i < half; i++) {
      const invf = Math.fround(2 ** Math.fround(-Math.fround(i / half) * lb));
      for (let p = 0; p < L; p++) {
        const th = Math.fround(p * invf);
        cos[p * half + i] = Math.cos(th);
        sin[p * half + i] = Math.sin(th);
      }
    }
    const out = new Float32Array(X.length);
    for (let bh = 0; bh < B * H; bh++) {
      for (let p = 0; p < L; p++) {
        const o = (bh * L + p) * Dh, t = p * half;
        for (let i = 0; i < half; i++) {
          const x1 = X[o + i]!, x2 = X[o + half + i]!, c = cos[t + i]!, s = sin[t + i]!;
          out[o + i] = x1 * c - x2 * s;
          out[o + half + i] = x2 * c + x1 * s;
        }
      }
    }
    return this.#make(x.shape, "f32", out);
  }

  sdpa(q: CpuTensor, k: CpuTensor, v: CpuTensor, mask: CpuTensor | null, scale: number): CpuTensor {
    const Q = this.#f32(q, "sdpa"), K = this.#f32(k, "sdpa"), V = this.#f32(v, "sdpa");
    const [B, H, Lq, Dh] = q.shape as [number, number, number, number];
    const Hk = k.shape[1]!, Lk = k.shape[2]!, Dv = v.shape[3]!;
    if (H % Hk) throw new RangeError("backend-cpu: sdpa head count mismatch");
    const rep = H / Hk;
    let M: Data | null = null;
    let msB = 0, msH = 0, msI = 0, msJ = 0;
    let boolMask = true;
    if (mask) {
      M = mask.data;
      boolMask = mask.dtype === "bool";
      const ms = [...mask.shape];
      while (ms.length < 4) ms.unshift(1);
      [msB, msH, msI, msJ] = broadcastStrides(ms, [B, H, Lq, Lk]) as [number, number, number, number];
    }
    const out = new Float32Array(B * H * Lq * Dv);
    const S = new Float32Array(Lq * Lk);
    const Vt = new Float32Array(Dv * Lk);
    for (let b = 0; b < B; b++) {
      for (let h = 0; h < H; h++) {
        const kh = Math.floor(h / rep);
        const qo = (b * H + h) * Lq * Dh, ko = (b * Hk + kh) * Lk * Dh, vo = (b * Hk + kh) * Lk * Dv;
        gemmNT(Q, qo, Dh, K, ko, Dh, S, 0, Lk, Lq, Lk, Dh);
        for (let i = 0; i < Lq; i++) {
          const row = i * Lk;
          const mo = b * msB + h * msH + i * msI;
          let m = -Infinity;
          for (let j = 0; j < Lk; j++) {
            let s = S[row + j]! * scale;
            if (M) {
              const mv = M[mo + j * msJ]!;
              if (boolMask) {
                if (!mv) s = -Infinity;
              } else s += mv;
            }
            S[row + j] = s;
            if (s > m) m = s;
          }
          if (m === -Infinity) {
            S.fill(0, row, row + Lk); // fully masked row (undefined by contract): zeros
            continue;
          }
          let sum = 0;
          for (let j = 0; j < Lk; j++) {
            const e = Math.exp(S[row + j]! - m);
            S[row + j] = e;
            sum += e;
          }
          const inv = 1 / sum;
          for (let j = 0; j < Lk; j++) S[row + j]! *= inv;
        }
        transpose2d(V, vo, Lk, Dv, Vt);
        gemmNT(S, 0, Lk, Vt, 0, Lk, out, (b * H + h) * Lq * Dv, Dv, Lq, Dv, Lk);
      }
    }
    return this.#make([B, H, Lq, Dv], "f32", out);
  }

  geglu(x: CpuTensor): CpuTensor {
    const X = this.#f32(x, "geglu");
    const D2 = x.shape[x.shape.length - 1]!, D = D2 >> 1;
    const rows = X.length / D2;
    const out = new Float32Array(rows * D);
    for (let r = 0; r < rows; r++) {
      const s = r * D2, o = r * D;
      for (let j = 0; j < D; j++) out[o + j] = geluScalar(X[s + j]!) * X[s + D + j]!;
    }
    return this.#make([...x.shape.slice(0, -1), D], "f32", out);
  }

  meanPool(x: CpuTensor, mask: CpuTensor): CpuTensor {
    const X = x.data;
    const [B, L, D] = x.shape as [number, number, number];
    const Mk = mask.data;
    const out = new Float32Array(B * D);
    const acc = new Float64Array(D);
    for (let b = 0; b < B; b++) {
      acc.fill(0);
      let cnt = 0;
      for (let l = 0; l < L; l++) {
        if (!Mk[b * L + l]) continue;
        cnt++;
        const s = (b * L + l) * D;
        for (let j = 0; j < D; j++) acc[j]! += X[s + j]!;
      }
      const inv = 1 / Math.max(cnt, 1);
      for (let j = 0; j < D; j++) out[b * D + j] = acc[j]! * inv;
    }
    return this.#make([B, D], "f32", out);
  }
}

/** Creates an independent CPU backend (no shared global state). */
export function createCpuBackend(): Backend<CpuTensor> {
  return new CpuBackend();
}
