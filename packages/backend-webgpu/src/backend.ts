import type { Backend, DType, HostTensor, Shape, Tensor } from "@johnhenry/tensor-backend";
import { f32ToBf16Bits } from "@johnhenry/tensor-backend";
import { Runtime, Storage, type CompiledKernel, type KernelSource } from "./runtime.ts";
import {
  ERF_HELPERS,
  GEMM_DEFAULT,
  POW_HELPERS,
  argReduceKernel,
  copyKernel,
  cumsumKernel,
  gatherKernel,
  gegluKernel,
  gemmKernel,
  gemmDirectKernel,
  gemmSkinnyKernel,
  gemmSgKernel,
  grid,
  layerNormKernel,
  meanPoolKernel,
  naryKernel,
  reduceKernel,
  ropeKernel,
  sdpaConfig,
  sdpaKernel,
  sdpaFastKernel,
  softmaxColKernel,
  softmaxRowKernel,
  sortKernel,
  sortSlowKernel,
  type CType,
  type GemmConfig,
  type Kind,
  kindKey,
  type NaryInput,
} from "./kernels.ts";
import type { AdapterSummary } from "./device.ts";

/** GPU-resident tensor: a contiguous view (element offset) into a refcounted buffer. */
export class WebGpuTensor implements Tensor {
  disposed = false;
  readonly shape: readonly number[];
  readonly dtype: DType;
  readonly storage: Storage;
  /** Element offset into `storage`. */
  readonly offset: number;
  constructor(shape: readonly number[], dtype: DType, storage: Storage, offset: number) {
    this.shape = shape;
    this.dtype = dtype;
    this.storage = storage;
    this.offset = offset;
  }
  get size(): number {
    return numel(this.shape);
  }
}

export interface WebGpuBackendOptions {
  /** Dispatches per command buffer before an automatic submit (default 128). */
  maxBatch?: number;
  /** Dispatches in the first submit after the GPU went idle (default 24), so the GPU starts while the rest is encoded. */
  firstBatch?: number;
  /** Idle bytes kept in the buffer pool (default 1 GiB). */
  maxPooledBytes?: number;
  /** Override the GEMM tile configuration (benchmarking). */
  gemm?: GemmConfig;
  /**
   * Before awaiting a readback, sleep for most of the GPU time the same
   * amount of work took last time, instead of letting the runtime poll
   * (Dawn-node polls in a busy loop). Default: true for Node/Bun (Dawn),
   * false for navigator.gpu.
   */
  sleepWhileWaiting?: boolean;
}

const numel = (s: readonly number[]) => s.reduce((a, b) => a * b, 1);
const contiguousStrides = (s: readonly number[]) => {
  const st = new Array<number>(s.length);
  let acc = 1;
  for (let i = s.length - 1; i >= 0; i--) {
    st[i] = acc;
    acc *= s[i]!;
  }
  return st;
};
/** Right-align to rank 8 (pad leading dims with size 1 / stride 0). */
const pad8 = (xs: readonly number[], fill: number) => {
  if (xs.length > 8) throw new Error(`webgpu: rank ${xs.length} > 8 not supported`);
  return [...new Array<number>(8 - xs.length).fill(fill), ...xs];
};
/** Cheap pipeline-cache key fragment for kinds / configs / scalars. */
const keyOf = (...xs: unknown[]): string =>
  xs
    .map((v) => (v == null ? "-" : typeof v === "object" ? ("st" in v ? kindKey(v as Kind) : Object.values(v).join("x")) : String(v)))
    .join(",");
const isFloat = (d: DType) => d === "f32" || d === "f16" || d === "bf16";

function broadcastShapes(...shapes: (readonly number[])[]): number[] {
  const rank = Math.max(...shapes.map((s) => s.length));
  const out: number[] = [];
  for (let i = 0; i < rank; i++) {
    let d = 1;
    for (const s of shapes) {
      const v = s[s.length - rank + i] ?? 1;
      if (v !== 1) {
        if (d !== 1 && d !== v) throw new Error(`webgpu: cannot broadcast [${shapes.map((x) => `[${x}]`).join(", ")}]`);
        d = v;
      }
    }
    out.push(d);
  }
  return out;
}

/** Strides of `s` broadcast into `out` shape (0 on broadcast axes). */
function broadcastStrides(s: readonly number[], out: readonly number[]): number[] {
  const st = contiguousStrides(s);
  const r = out.length - s.length;
  return out.map((d, i) => (i < r || s[i - r] === 1 ? 0 : st[i - r]!));
}

function promote(a: DType, b: DType): DType {
  if (a === b) return a;
  if (isFloat(a) && isFloat(b)) return "f32"; // f16+bf16 or f*+f32
  if (isFloat(a)) return a;
  if (isFloat(b)) return b;
  return "i32";
}

export class WebGpuBackend implements Backend<WebGpuTensor> {
  readonly name = "webgpu";
  readonly rt: Runtime;
  readonly hasF16: boolean;
  /** f32 8×8×8 subgroup matrices usable (Dawn chromium-experimental-subgroup-matrix, subgroup size 32). */
  readonly hasSubgroupMatrix: boolean;
  gemmConfig: GemmConfig;
  private scopes: Set<WebGpuTensor>[] = [];
  private ropeTables = new Map<string, Storage>();
  private destroyed = false;

  readonly device: GPUDevice;
  readonly adapterInfo: AdapterSummary;
  private readonly ownsDevice: boolean;

  constructor(device: GPUDevice, adapterInfo: AdapterSummary, opts: WebGpuBackendOptions & { f16: boolean; ownsDevice: boolean; subgroupMatrix?: boolean }) {
    this.device = device;
    this.adapterInfo = adapterInfo;
    this.hasF16 = opts.f16;
    this.hasSubgroupMatrix = opts.subgroupMatrix ?? false;
    this.ownsDevice = opts.ownsDevice;
    this.rt = new Runtime(device, opts.maxBatch ?? 128, opts.maxPooledBytes ?? 2 ** 30, opts.firstBatch ?? 24);
    this.gemmConfig = opts.gemm ?? GEMM_DEFAULT;
    this.rt.sleepWhileWaiting = opts.sleepWhileWaiting ?? adapterInfo.source !== "navigator.gpu";
  }

  supports(dtype: DType): boolean {
    return dtype === "f16" ? this.hasF16 : true;
  }

  // ---- storage kinds -------------------------------------------------------

  kind(d: DType): Kind {
    switch (d) {
      case "f32":
        return { st: "f32" };
      case "bf16":
        return { st: "f32", bf16: true };
      case "f16":
        return this.hasF16 ? { st: "f16" } : { st: "f32" };
      case "i32":
        return { st: "i32" };
      case "bool":
        return { st: "u32" };
    }
  }
  private bytesPer(d: DType): number {
    return this.kind(d).st === "f16" ? 2 : 4;
  }

  // ---- lifetime ------------------------------------------------------------

  private track(t: WebGpuTensor): WebGpuTensor {
    this.scopes[this.scopes.length - 1]?.add(t);
    return t;
  }
  private alloc(shape: readonly number[], dtype: DType): WebGpuTensor {
    const bytes = Math.max(4, numel(shape) * this.bytesPer(dtype));
    const { buffer, bytes: cls } = this.rt.acquire(bytes);
    return this.track(new WebGpuTensor([...shape], dtype, new Storage(buffer, cls), 0));
  }
  private view(x: WebGpuTensor, shape: readonly number[], offset = x.offset, dtype = x.dtype): WebGpuTensor {
    this.live(x);
    x.storage.refs++;
    return this.track(new WebGpuTensor([...shape], dtype, x.storage, offset));
  }
  private live(x: WebGpuTensor): void {
    if (x.disposed) throw new Error("webgpu: tensor used after dispose");
  }

  dispose(t: WebGpuTensor): void {
    if (t.disposed) return;
    t.disposed = true;
    if (--t.storage.refs === 0) this.rt.release(t.storage.buffer, t.storage.bytes);
  }

  scope<R>(fn: () => R): R {
    const s = new Set<WebGpuTensor>();
    this.scopes.push(s);
    let r: R;
    try {
      r = fn();
    } catch (e) {
      this.scopes.pop();
      for (const t of s) this.dispose(t);
      throw e;
    }
    this.scopes.pop();
    const keep = new Set<unknown>();
    const visit = (v: unknown, depth: number) => {
      if (v instanceof WebGpuTensor) keep.add(v);
      else if (depth < 2 && v && typeof v === "object") for (const x of Array.isArray(v) ? v : Object.values(v)) visit(x, depth + 1);
    };
    visit(r, 0);
    const parent = this.scopes[this.scopes.length - 1];
    for (const t of s) {
      if (keep.has(t)) parent?.add(t);
      else this.dispose(t);
    }
    return r;
  }

  flush(): void {
    this.rt.flush();
  }

  /** Resolves when all submitted GPU work is done (benchmarking). */
  async sync(): Promise<void> {
    await this.rt.onIdle();
    this.rt.checkErrors();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const s of this.ropeTables.values()) s.buffer.destroy();
    this.ropeTables.clear();
    this.rt.destroyAll();
    if (this.ownsDevice) this.device.destroy();
  }

  // ---- transfer ------------------------------------------------------------

  /** Uploads via `queue.writeBuffer`, which copies the host data at call time; resolves once queued. */
  async fromHost(h: HostTensor): Promise<WebGpuTensor> {
    return this.upload(h);
  }

  private upload(h: HostTensor): WebGpuTensor {
    const n = numel(h.shape);
    if (h.data.length !== n) throw new RangeError(`fromHost: ${h.data.length} values for shape [${h.shape}]`);
    const k = this.kind(h.dtype);
    let data: ArrayBufferView;
    const d = h.data;
    switch (h.dtype) {
      case "f32":
        data = d instanceof Float32Array ? d : Float32Array.from(d as ArrayLike<number>);
        break;
      case "bf16": {
        const out = new Float32Array(n);
        const u = new Uint32Array(out.buffer);
        const src = d as Uint16Array;
        for (let i = 0; i < n; i++) u[i] = src[i]! << 16;
        data = out;
        break;
      }
      case "f16":
        if (k.st === "f16") {
          data = d instanceof Uint16Array || d instanceof Float16Array ? d : Float16Array.from(d as ArrayLike<number>);
        } else {
          data = d instanceof Uint16Array ? Float32Array.from(new Float16Array(d.buffer, d.byteOffset, d.length)) : Float32Array.from(d as ArrayLike<number>);
        }
        break;
      case "i32":
        data = d instanceof Int32Array ? d : Int32Array.from(d as ArrayLike<number>);
        break;
      case "bool": {
        // A plain loop: Uint32Array.from with a map callback is ~10× slower
        // (the B×L×L sliding-window masks reach millions of elements).
        const out = new Uint32Array(n);
        const src = d as ArrayLike<number>;
        for (let i = 0; i < n; i++) out[i] = src[i] ? 1 : 0;
        data = out;
        break;
      }
    }
    const bytes = Math.max(4, n * this.bytesPer(h.dtype));
    const { buffer, bytes: cls, writeHazard } = this.rt.acquire(bytes);
    if (n) this.rt.write(buffer, writeHazard, data);
    return this.track(new WebGpuTensor([...h.shape], h.dtype, new Storage(buffer, cls), 0));
  }

  async read(t: WebGpuTensor): Promise<HostTensor> {
    this.live(t);
    const n = t.size;
    const k = this.kind(t.dtype);
    const bpe = this.bytesPer(t.dtype);
    let src = t;
    let tmp: WebGpuTensor | null = null;
    if ((t.offset * bpe) % 4 !== 0) {
      tmp = this.copyContig(t);
      src = tmp;
    }
    const raw = n ? await this.rt.readBytes(src.storage.buffer, src.offset * bpe, n * bpe) : new ArrayBuffer(0);
    if (tmp) this.dispose(tmp);
    const shape = [...t.shape];
    switch (t.dtype) {
      case "f32":
        return { dtype: "f32", shape, data: new Float32Array(raw) };
      case "f16":
        return { dtype: "f16", shape, data: k.st === "f16" ? new Float16Array(raw) : Float16Array.from(new Float32Array(raw)) };
      case "bf16": {
        const f = new Float32Array(raw);
        const out = new Uint16Array(n);
        for (let i = 0; i < n; i++) out[i] = f32ToBf16Bits(f[i]!);
        return { dtype: "bf16", shape, data: out };
      }
      case "i32":
        return { dtype: "i32", shape, data: new Int32Array(raw) };
      case "bool":
        return { dtype: "bool", shape, data: Uint8Array.from(new Uint32Array(raw)) };
    }
  }

  // ---- dispatch helpers ------------------------------------------------------

  private k(src: () => KernelSource, key: string): CompiledKernel {
    return this.rt.kernel(src, key);
  }
  private run(
    key: string,
    src: () => KernelSource,
    bufs: GPUBuffer[],
    params: Record<string, number | readonly number[]>,
    groups: readonly [number, number?, number?],
  ): void {
    this.rt.dispatch(this.k(src, key), bufs, params, groups);
  }
  private flatGroups(n: number, wg = 256) {
    return grid(Math.ceil(n / wg));
  }

  /** out = contiguous copy of a strided view described by (shape, strides, offset). */
  private stridedCopy(
    x: WebGpuTensor,
    shape: readonly number[],
    inStrides: readonly number[],
    inOffset: number,
    out: WebGpuTensor,
    outStrides: readonly number[],
    outOffset: number,
  ): void {
    const n = numel(shape);
    if (!n) return;
    const ik = this.kind(x.dtype), ok = this.kind(out.dtype);
    // Collapse: drop unit axes, merge axes that are contiguous in both views.
    const sh: number[] = [], is: number[] = [], os: number[] = [];
    for (let d = shape.length - 1; d >= 0; d--) {
      if (shape[d] === 1) continue;
      if (sh.length && inStrides[d] === is[0]! * sh[0]! && outStrides[d] === os[0]! * sh[0]!) {
        sh[0] = sh[0]! * shape[d]!;
      } else {
        sh.unshift(shape[d]!);
        is.unshift(inStrides[d]!);
        os.unshift(outStrides[d]!);
      }
    }
    if (!sh.length) sh.push(1), is.push(1), os.push(1);
    const R = sh.length;
    const V = is[R - 1] === 1 && os[R - 1] === 1 && sh[R - 1]! % 4 === 0 ? 4 : 1;
    if (V === 4) sh[R - 1] = sh[R - 1]! / 4;
    const groups = n / V;
    const src = () => copyKernel(ik, ok, R, V);
    this.run(`copy:${keyOf(ik, ok)}:${R}:${V}`, src, [x.storage.buffer, out.storage.buffer], {
      n: groups, io: inOffset, oo: outOffset, sh: pad8(sh, 1), ist: pad8(is.map((v, i) => (i === R - 1 ? v * V : v)), 0), ost: pad8(os.map((v, i) => (i === R - 1 ? v * V : v)), 0),
    }, this.flatGroups(groups));
  }

  private copyContig(x: WebGpuTensor, dtype: DType = x.dtype): WebGpuTensor {
    const out = this.alloc(x.shape, dtype);
    const st = contiguousStrides(x.shape);
    this.stridedCopy(x, x.shape, st, x.offset, out, st, 0);
    return out;
  }

  private nary(op: string, expr: string, xs: WebGpuTensor[], outDtype: DType, c: CType, s = 0, helpers = ""): WebGpuTensor {
    for (const x of xs) this.live(x);
    const shape = broadcastShapes(...xs.map((x) => x.shape));
    const n = numel(shape);
    const out = this.alloc(shape, outDtype);
    if (!n) return out;
    const ins: NaryInput[] = xs.map((x) => ({ kind: this.kind(x.dtype), flat: x.size === n && x.shape.length === shape.length }));
    // Same-shape check for "flat": sizes equal and shapes equal after broadcast.
    ins.forEach((inp, j) => {
      if (inp.flat) inp.flat = xs[j]!.shape.every((d, i) => d === shape[i]);
    });
    const params: Record<string, number | readonly number[]> = { n, s };
    if (ins.some((i) => !i.flat)) params.sh = pad8(shape, 1);
    xs.forEach((x, j) => {
      params[`o${j}`] = x.offset;
      if (!ins[j]!.flat) params[`st${j}`] = pad8(broadcastStrides(x.shape, shape), 0);
    });
    const ok = this.kind(outDtype);
    const key = `nary:${op}:${ins.map((i) => (i.kind.st + (i.kind.bf16 ? "b" : "") + (i.flat ? "f" : "s"))).join(",")}:${ok.st}${ok.bf16 ? "b" : ""}:${c}`;
    this.run(key, () => naryKernel(op, expr, ins, ok, c, helpers), [...xs.map((x) => x.storage.buffer), out.storage.buffer], params, this.flatGroups(n));
    return out;
  }

  // ---- shape ---------------------------------------------------------------

  reshape(x: WebGpuTensor, shape: Shape): WebGpuTensor {
    const s = [...shape];
    const neg = s.indexOf(-1);
    if (neg >= 0) s[neg] = x.size / numel(s.filter((_, i) => i !== neg));
    if (numel(s) !== x.size) throw new Error(`reshape: [${x.shape}] → [${shape}]`);
    return this.view(x, s);
  }

  transpose(x: WebGpuTensor, perm: readonly number[]): WebGpuTensor {
    this.live(x);
    const r = x.shape.length;
    const p = perm.map((a) => (a < 0 ? a + r : a));
    const shape = p.map((a) => x.shape[a]!);
    // Identity up to size-1 axes → free view.
    const nonUnit = p.filter((a) => x.shape[a] !== 1);
    if (nonUnit.every((a, i) => i === 0 || a > nonUnit[i - 1]!)) return this.view(x, shape);
    const st = contiguousStrides(x.shape);
    const out = this.alloc(shape, x.dtype);
    this.stridedCopy(x, shape, p.map((a) => st[a]!), x.offset, out, contiguousStrides(shape), 0);
    return out;
  }

  slice(x: WebGpuTensor, begin: readonly number[], end: readonly number[]): WebGpuTensor {
    this.live(x);
    const r = x.shape.length;
    const b: number[] = [], shape: number[] = [];
    for (let i = 0; i < r; i++) {
      const d = x.shape[i]!;
      let lo = begin[i] ?? 0, hi = end[i] ?? d;
      if (lo < 0) lo += d;
      if (hi < 0) hi += d;
      lo = Math.min(Math.max(lo, 0), d);
      hi = Math.min(Math.max(hi, lo), d);
      b.push(lo);
      shape.push(hi - lo);
    }
    const st = contiguousStrides(x.shape);
    const offset = x.offset + b.reduce((a, v, i) => a + v * st[i]!, 0);
    // Contiguous iff: leading axes of extent 1, one partial axis, then full axes.
    let k = 0;
    while (k < r && shape[k] === 1) k++;
    let contiguous = true;
    for (let i = k + 1; i < r; i++) if (shape[i] !== x.shape[i]) contiguous = false;
    if (contiguous) return this.view(x, shape, offset);
    const out = this.alloc(shape, x.dtype);
    this.stridedCopy(x, shape, st, offset, out, contiguousStrides(shape), 0);
    return out;
  }

  split(x: WebGpuTensor, parts: number, axis: number): WebGpuTensor[] {
    const r = x.shape.length;
    const a = axis < 0 ? axis + r : axis;
    const d = x.shape[a]!;
    if (d % parts) throw new Error(`split: axis ${a} of size ${d} not divisible by ${parts}`);
    const step = d / parts;
    const out: WebGpuTensor[] = [];
    for (let p = 0; p < parts; p++) {
      const b = new Array<number>(r).fill(0), e = [...x.shape];
      b[a] = p * step;
      e[a] = (p + 1) * step;
      out.push(this.slice(x, b, e));
    }
    return out;
  }

  concat(xs: readonly WebGpuTensor[], axis: number): WebGpuTensor {
    const first = xs[0]!;
    const r = first.shape.length;
    const a = axis < 0 ? axis + r : axis;
    const dtype = xs.map((x) => x.dtype).reduce(promote);
    const shape = [...first.shape];
    shape[a] = xs.reduce((s, x) => s + x.shape[a]!, 0);
    const out = this.alloc(shape, dtype);
    const ost = contiguousStrides(shape);
    let at = 0;
    for (const x of xs) {
      this.live(x);
      this.stridedCopy(x, x.shape, contiguousStrides(x.shape), x.offset, out, ost, at * ost[a]!);
      at += x.shape[a]!;
    }
    return out;
  }

  cast(x: WebGpuTensor, dtype: DType): WebGpuTensor {
    this.live(x);
    if (dtype === x.dtype) return this.view(x, x.shape);
    return this.copyContig(x, dtype);
  }

  // ---- elementwise ---------------------------------------------------------

  private binary(op: string, expr: string, a: WebGpuTensor, b: WebGpuTensor, floatOnly = false): WebGpuTensor {
    let out = promote(a.dtype, b.dtype);
    if (floatOnly && !isFloat(out)) out = "f32";
    const c: CType = isFloat(out) || out === "bool" ? "f32" : "i32";
    return this.nary(op, expr, [a, b], out, c);
  }
  add(a: WebGpuTensor, b: WebGpuTensor) { return this.binary("add", "a + b", a, b); }
  sub(a: WebGpuTensor, b: WebGpuTensor) { return this.binary("sub", "a - b", a, b); }
  mul(a: WebGpuTensor, b: WebGpuTensor) { return this.binary("mul", "a * b", a, b); }
  div(a: WebGpuTensor, b: WebGpuTensor) { return this.binary("div", "a / b", a, b, true); }
  maximum(a: WebGpuTensor, b: WebGpuTensor) { return this.binary("maximum", "max(a, b)", a, b); }

  where(cond: WebGpuTensor, a: WebGpuTensor, b: WebGpuTensor): WebGpuTensor {
    const out = promote(a.dtype, b.dtype);
    const c: CType = isFloat(out) || out === "bool" ? "f32" : "i32";
    // cond is loaded into the compute type; nonzero = true.
    return this.nary("where", c === "f32" ? "select(c, b, a != 0.0)" : "select(c, b, a != 0)", [cond, a, b], out, c);
  }

  private unaryFloat(op: string, expr: string, x: WebGpuTensor, s = 0, helpers = ""): WebGpuTensor {
    const out = isFloat(x.dtype) ? x.dtype : "f32";
    return this.nary(op, expr, [x], out, "f32", s, helpers);
  }
  scale(x: WebGpuTensor, s: number): WebGpuTensor {
    if (x.dtype === "i32") return this.nary("scale", "a * P.s", [x], "f32", "f32", s);
    return this.unaryFloat("scale", "a * P.s", x, s);
  }
  exp(x: WebGpuTensor) { return this.unaryFloat("exp", "exp(a)", x); }
  log(x: WebGpuTensor) { return this.unaryFloat("log", "log(a)", x); }
  relu(x: WebGpuTensor) {
    if (x.dtype === "i32") return this.nary("relu", "max(a, 0)", [x], "i32", "i32");
    return this.unaryFloat("relu", "max(a, 0.0)", x);
  }
  gelu(x: WebGpuTensor) { return this.unaryFloat("gelu", "gelu_(a)", x); }

  // ---- reductions ----------------------------------------------------------

  private axis3(x: WebGpuTensor, axis: number): { a: number; outer: number; R: number; inner: number } {
    const r = x.shape.length;
    const a = axis < 0 ? axis + r : axis;
    if (a < 0 || a >= r) throw new Error(`axis ${axis} out of range for rank ${r}`);
    return { a, outer: numel(x.shape.slice(0, a)), R: x.shape[a]!, inner: numel(x.shape.slice(a + 1)) };
  }

  private reduce(op: "sum" | "max" | "min" | "mean", x: WebGpuTensor, axis: number, keepDims = false): WebGpuTensor {
    this.live(x);
    const { a, outer, R, inner } = this.axis3(x, axis);
    const shape = keepDims ? x.shape.map((d, i) => (i === a ? 1 : d)) : x.shape.filter((_, i) => i !== a);
    const outDtype: DType =
      op === "mean" ? (isFloat(x.dtype) ? x.dtype : "f32") : op === "sum" && (x.dtype === "bool" || x.dtype === "i32") ? "i32" : x.dtype;
    const out = this.alloc(shape, outDtype);
    const n = outer * inner;
    if (!n || !R) return out;
    const ik = this.kind(x.dtype), ok = this.kind(outDtype);
    const src = () => reduceKernel(op, ik, ok);
    this.run(`reduce:${op}:${keyOf(ik)}:${keyOf(ok)}`, src, [x.storage.buffer, out.storage.buffer], { n, R, inner, off: x.offset }, this.flatGroups(n));
    return out;
  }
  sum(x: WebGpuTensor, axis: number, keepDims?: boolean) { return this.reduce("sum", x, axis, keepDims); }
  max(x: WebGpuTensor, axis: number, keepDims?: boolean) { return this.reduce("max", x, axis, keepDims); }

  softmax(x: WebGpuTensor, axis: number): WebGpuTensor {
    this.live(x);
    const { outer, R, inner } = this.axis3(x, axis);
    const dtype = isFloat(x.dtype) ? x.dtype : "f32";
    const out = this.alloc(x.shape, dtype);
    const ik = this.kind(x.dtype), ok = this.kind(dtype);
    if (!x.size) return out;
    if (inner === 1) {
      this.run(`softmaxrow:${keyOf(ik, ok)}`, () => softmaxRowKernel(ik, ok), [x.storage.buffer, out.storage.buffer], { rows: outer, D: R, off: x.offset }, grid(outer));
    } else {
      const n = outer * inner;
      this.run(`softmaxcol:${keyOf(ik, ok)}`, () => softmaxColKernel(ik, ok), [x.storage.buffer, out.storage.buffer], { n, R, inner, off: x.offset }, this.flatGroups(n));
    }
    return out;
  }

  sort(x: WebGpuTensor, axis: number): WebGpuTensor {
    this.live(x);
    const r = x.shape.length;
    const a = axis < 0 ? axis + r : axis;
    if (a !== r - 1) {
      return this.scope(() => {
        const perm = [...Array(r).keys()];
        perm[a] = r - 1;
        perm[r - 1] = a;
        return this.transpose(this.sort(this.transpose(x, perm), -1), perm);
      });
    }
    const n = x.shape[r - 1]!;
    const rows = x.size / Math.max(1, n);
    const out = this.alloc(x.shape, x.dtype);
    if (!x.size) return out;
    const k = this.kind(x.dtype);
    if (n <= 4096) {
      const NP = Math.max(2, 2 ** Math.ceil(Math.log2(n)));
      this.run(`sort:${keyOf(k)}:${NP}`, () => sortKernel(k, k, NP), [x.storage.buffer, out.storage.buffer], { rows, n, off: x.offset }, grid(rows));
    } else {
      const st = contiguousStrides(x.shape);
      this.stridedCopy(x, x.shape, st, x.offset, out, st, 0);
      this.run(`sortslow:${keyOf(k)}`, () => sortSlowKernel(k), [out.storage.buffer], { rows, n }, this.flatGroups(rows, 64));
    }
    return out;
  }

  // ---- linear algebra ------------------------------------------------------

  private gemm(
    a: WebGpuTensor,
    b: WebGpuTensor,
    bias: WebGpuTensor | null,
    M: number,
    N: number,
    K: number,
    transB: boolean,
    batchShape: number[],
    aBatchStrides: number[],
    bBatchStrides: number[],
    outShape: number[],
  ): WebGpuTensor {
    const outDtype = [a.dtype, b.dtype, ...(bias ? [bias.dtype] : [])].reduce(promote);
    const outFloat = isFloat(outDtype) ? outDtype : "f32";
    const out = this.alloc(outShape, outFloat);
    const batch = numel(batchShape);
    if (!M || !N || !batch) return out;
    const ak = this.kind(a.dtype), bk = this.kind(b.dtype), ok = this.kind(outFloat);
    const biask = bias ? this.kind(bias.dtype) : null;
    const vecA = K % 4 === 0 && a.offset % 4 === 0 && aBatchStrides.every((s) => s % 4 === 0);
    const vecB = (transB ? K % 4 === 0 : N % 4 === 0) && b.offset % 4 === 0 && bBatchStrides.every((s) => s % 4 === 0);
    const sk = transB && vecA && vecB && batch === 1 ? this.gemmConfig.skinny.find((c) => M <= c.maxM) : undefined;
    if (sk) {
      const TM = Math.ceil(M / sk.WY);
      this.run(`gemmskinny:${keyOf(ak, bk, biask, ok, sk, TM)}`, () => gemmSkinnyKernel(ak, bk, biask, ok, sk, TM),
        [a.storage.buffer, b.storage.buffer, ...(bias ? [bias.storage.buffer] : []), out.storage.buffer],
        { M, N, K, oa: a.offset, ob: b.offset, obias: bias?.offset ?? 0 },
        [Math.ceil(N / (sk.WX * sk.TN)), 1, 1]);
      return out;
    }
    const sg = this.hasSubgroupMatrix && transB && vecA && vecB && batch === 1
      ? this.gemmConfig.sg?.find((c) => M > c.minM && M <= (c.maxM ?? Infinity))
      : undefined;
    if (sg) {
      this.run(`gemmsg:${keyOf(ak, bk, biask, ok, sg)}`, () => gemmSgKernel(ak, bk, biask, ok, sg),
        [a.storage.buffer, b.storage.buffer, ...(bias ? [bias.storage.buffer] : []), out.storage.buffer],
        { M, N, K, oa: a.offset, ob: b.offset, obias: bias?.offset ?? 0 },
        [Math.ceil(N / sg.BN), Math.ceil(M / sg.BM), 1]);
      return out;
    }
    const dc = this.gemmConfig.direct;
    if (dc && transB && vecA && vecB && batch === 1) {
      this.run(`gemmdirect:${keyOf(ak, bk, biask, ok, dc)}`, () => gemmDirectKernel(ak, bk, biask, ok, dc),
        [a.storage.buffer, b.storage.buffer, ...(bias ? [bias.storage.buffer] : []), out.storage.buffer],
        { M, N, K, oa: a.offset, ob: b.offset, obias: bias?.offset ?? 0 },
        [Math.ceil(N / (dc.WX * dc.TN)), Math.ceil(M / (dc.WY * dc.TM)), 1]);
      return out;
    }
    const cfg = this.gemmConfig.tiled;
    const key = `gemm:${keyOf(ak, bk, biask, ok)}:${transB}:${vecA}:${vecB}:${cfg.BM}x${cfg.BN}x${cfg.BK}/${cfg.TM}x${cfg.TN}`;
    const bufs = [a.storage.buffer, b.storage.buffer, ...(bias ? [bias.storage.buffer] : []), out.storage.buffer];
    if (batch > 65535) throw new Error("gemm: batch > 65535");
    this.run(key, () => gemmKernel(ak, bk, biask, ok, transB, vecA, vecB, cfg), bufs, {
      M, N, K, oa: a.offset, ob: b.offset, obias: bias?.offset ?? 0,
      bsh: pad8(batchShape, 1), ast: pad8(aBatchStrides, 0), bst: pad8(bBatchStrides, 0),
    }, [Math.ceil(N / cfg.BN), Math.ceil(M / cfg.BM), batch]);
    return out;
  }

  matmul(a: WebGpuTensor, b: WebGpuTensor): WebGpuTensor {
    this.live(a);
    this.live(b);
    if (a.shape.length < 2 || b.shape.length < 2) throw new Error("matmul: operands must be at least 2-D");
    const M = a.shape[a.shape.length - 2]!, K = a.shape[a.shape.length - 1]!;
    const K2 = b.shape[b.shape.length - 2]!, N = b.shape[b.shape.length - 1]!;
    if (K !== K2) throw new Error(`matmul: [${a.shape}] @ [${b.shape}]`);
    const aB = a.shape.slice(0, -2), bB = b.shape.slice(0, -2);
    const batchShape = broadcastShapes(aB, bB);
    const outShape = [...batchShape, M, N];
    // B shared across the batch and A batch contiguous → one big GEMM.
    if (numel(bB) === 1 && numel(aB) === numel(batchShape)) {
      return this.gemm(a, b, null, numel(aB) * M, N, K, false, [], [], [], outShape);
    }
    const aS = broadcastStrides(aB, batchShape).map((s) => s * M * K);
    const bS = broadcastStrides(bB, batchShape).map((s) => s * K * N);
    return this.gemm(a, b, null, M, N, K, false, batchShape, aS, bS, outShape);
  }

  linear(x: WebGpuTensor, w: WebGpuTensor, b?: WebGpuTensor | null): WebGpuTensor {
    this.live(x);
    this.live(w);
    if (b) this.live(b);
    const K = x.shape[x.shape.length - 1]!;
    const [N, K2] = w.shape as [number, number];
    if (K !== K2) throw new Error(`linear: x [${x.shape}] w [${w.shape}]`);
    const M = x.size / K;
    return this.gemm(x, w, b ?? null, M, N, K, true, [], [], [], [...x.shape.slice(0, -1), N]);
  }

  layerNorm(x: WebGpuTensor, weight: WebGpuTensor | null, bias: WebGpuTensor | null, eps: number): WebGpuTensor {
    this.live(x);
    const D = x.shape[x.shape.length - 1]!;
    const rows = x.size / D;
    const out = this.alloc(x.shape, isFloat(x.dtype) ? x.dtype : "f32");
    if (!x.size) return out;
    const ik = this.kind(x.dtype), wk = weight ? this.kind(weight.dtype) : null, bk = bias ? this.kind(bias.dtype) : null, ok = this.kind(out.dtype);
    const bufs = [x.storage.buffer, ...(weight ? [weight.storage.buffer] : []), ...(bias ? [bias.storage.buffer] : []), out.storage.buffer];
    this.run(`ln:${keyOf(ik, wk, bk, ok)}`, () => layerNormKernel(ik, wk, bk, ok), bufs, {
      rows, D, off: x.offset, ow: weight?.offset ?? 0, ob: bias?.offset ?? 0, eps,
    }, grid(rows));
    return out;
  }

  embedding(table: WebGpuTensor, ids: WebGpuTensor): WebGpuTensor {
    this.live(table);
    this.live(ids);
    const [V, D] = table.shape as [number, number];
    const idsI = ids.dtype === "i32" ? ids : this.cast(ids, "i32");
    const out = this.alloc([...ids.shape, D], table.dtype);
    const n = ids.size * D;
    if (n) {
      const tk = this.kind(table.dtype);
      this.run(`gather:e:${keyOf(tk)}`, () => gatherKernel(tk, tk, "embedding"), [table.storage.buffer, idsI.storage.buffer, out.storage.buffer], {
        n, D, V, M: 1, ot: table.offset, oi: idsI.offset,
      }, this.flatGroups(n));
    }
    if (idsI !== ids) this.dispose(idsI);
    return out;
  }

  gatherRows(x: WebGpuTensor, idx: WebGpuTensor): WebGpuTensor {
    this.live(x);
    this.live(idx);
    const [B, L, D] = x.shape as [number, number, number];
    const M = idx.shape[1]!;
    const idxI = idx.dtype === "i32" ? idx : this.cast(idx, "i32");
    const out = this.alloc([B, M, D], x.dtype);
    const n = B * M * D;
    if (n) {
      const tk = this.kind(x.dtype);
      this.run(`gather:r:${keyOf(tk)}`, () => gatherKernel(tk, tk, "rows"), [x.storage.buffer, idxI.storage.buffer, out.storage.buffer], {
        n, D, V: L, M, ot: x.offset, oi: idxI.offset,
      }, this.flatGroups(n));
    }
    if (idxI !== idx) this.dispose(idxI);
    return out;
  }

  private ropeTable(L: number, D: number, base: number): Storage {
    const key = `${L}:${D}:${base}`;
    let s = this.ropeTables.get(key);
    if (s) return s;
    const half = D / 2;
    const data = new Float32Array(2 * L * half);
    // Angles emulate MLX's f32 computation (inv_freq and pos·inv_freq rounded to
    // f32, like @johnhenry/backend-cpu); cos/sin themselves are exact (f64).
    const lb = Math.fround(Math.log2(base));
    for (let i = 0; i < half; i++) {
      const inv = Math.fround(2 ** Math.fround(-Math.fround(i / half) * lb));
      for (let l = 0; l < L; l++) {
        const th = Math.fround(l * inv);
        data[l * half + i] = Math.cos(th);
        data[L * half + l * half + i] = Math.sin(th);
      }
    }
    const buffer = this.device.createBuffer({ size: Math.max(16, data.byteLength), usage: 0x80 | 0x08, mappedAtCreation: true });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    s = new Storage(buffer, data.byteLength);
    this.ropeTables.set(key, s);
    return s;
  }

  rope(x: WebGpuTensor, base: number): WebGpuTensor {
    this.live(x);
    const r = x.shape.length;
    const L = x.shape[r - 2]!, D = x.shape[r - 1]!;
    if (D % 2) throw new Error("rope: head dim must be even");
    const out = this.alloc(x.shape, isFloat(x.dtype) ? x.dtype : "f32");
    const n = x.size;
    if (!n) return out;
    const tab = this.ropeTable(L, D, base);
    const xk = this.kind(x.dtype), ok = this.kind(out.dtype);
    this.run(`rope:${keyOf(xk, ok)}`, () => ropeKernel(xk, ok), [x.storage.buffer, tab.buffer, out.storage.buffer], { n: n / 2, D, L, off: x.offset }, this.flatGroups(n / 2));
    return out;
  }

  sdpa(q: WebGpuTensor, k: WebGpuTensor, v: WebGpuTensor, mask: WebGpuTensor | null, scale: number): WebGpuTensor {
    for (const t of [q, k, v]) this.live(t);
    const [B, H, Lq, D] = q.shape as [number, number, number, number];
    const Lk = k.shape[2]!;
    if (k.shape[0] !== B || k.shape[1] !== H || v.shape[2] !== Lk || k.shape[3] !== D || v.shape[3] !== D)
      throw new Error(`sdpa: q [${q.shape}] k [${k.shape}] v [${v.shape}]`);
    const outDtype = [q.dtype, k.dtype, v.dtype].reduce(promote);
    const out = this.alloc([B, H, Lq, D], outDtype);
    if (!out.size) return out;
    let m = mask;
    if (m) {
      this.live(m);
      if (m.dtype !== "bool") m = this.cast(m, "bool");
    }
    const qk = this.kind(q.dtype), kk = this.kind(k.dtype), vk = this.kind(v.dtype), ok = this.kind(outDtype);
    const mk = m ? this.kind("bool") : null;
    const params: Record<string, number> = { H, Lq, Lk, oq: q.offset, ok: k.offset, ov: v.offset, scale };
    if (m) {
      const ms = m.shape;
      if (ms.length > 4) throw new Error("sdpa: mask rank > 4");
      const s4 = broadcastStrides(ms, [B, H, Lq, Lk].map((d, i) => Math.max(d, ms[ms.length - 4 + i] ?? 1)));
      Object.assign(params, { om: m.offset, msb: s4[0], msh: s4[1], msq: s4[2], msk: s4[3] });
    }
    const bufs = [q.storage.buffer, k.storage.buffer, v.storage.buffer, ...(m ? [m.storage.buffer] : []), out.storage.buffer];
    const fast = (D === 32 || D === 64) && q.offset % 4 === 0 && k.offset % 4 === 0 && v.offset % 4 === 0;
    if (fast) {
      this.run(`sdpafast:${keyOf(qk, kk, vk, mk, ok)}:${D}`, () => sdpaFastKernel(qk, kk, vk, mk, ok, D), bufs, params, [Math.ceil(Lq / 32), H, B]);
    } else {
      const { BQ } = sdpaConfig(D);
      this.run(`sdpa:${keyOf(qk, kk, vk, mk, ok)}:${D}`, () => sdpaKernel(qk, kk, vk, mk, ok, D), bufs, params, [Math.ceil(Lq / BQ), H, B]);
    }
    if (m && m !== mask) this.dispose(m);
    return out;
  }

  geglu(x: WebGpuTensor): WebGpuTensor {
    this.live(x);
    const r = x.shape.length;
    const F = x.shape[r - 1]! / 2;
    const out = this.alloc([...x.shape.slice(0, -1), F], isFloat(x.dtype) ? x.dtype : "f32");
    const n = out.size;
    if (!n) return out;
    const xk = this.kind(x.dtype), ok = this.kind(out.dtype);
    this.run(`geglu:${keyOf(xk, ok)}`, () => gegluKernel(xk, ok), [x.storage.buffer, out.storage.buffer], { n, F, off: x.offset }, this.flatGroups(n));
    return out;
  }

  meanPool(x: WebGpuTensor, mask: WebGpuTensor): WebGpuTensor {
    this.live(x);
    this.live(mask);
    const [B, L, D] = x.shape as [number, number, number];
    const out = this.alloc([B, D], "f32");
    const n = B * D;
    if (!n) return out;
    const xk = this.kind(x.dtype), mk = this.kind(mask.dtype);
    this.run(`meanpool:${keyOf(xk, mk)}`, () => meanPoolKernel(xk, mk), [x.storage.buffer, mask.storage.buffer, out.storage.buffer], {
      n, D, L, ox: x.offset, om: mask.offset,
    }, this.flatGroups(n));
    return out;
  }

  // ---- general numerics (optional contract ops) -------------------------------

  private compare(op: string, cmp: string, a: WebGpuTensor, b: WebGpuTensor): WebGpuTensor {
    const c: CType = isFloat(promote(a.dtype, b.dtype)) ? "f32" : "i32";
    const one = c === "f32" ? "1.0" : "1", zero = c === "f32" ? "0.0" : "0";
    return this.nary(op, `select(${zero}, ${one}, ${cmp})`, [a, b], "bool", c);
  }
  equal(a: WebGpuTensor, b: WebGpuTensor) { return this.compare("eq", "a == b", a, b); }
  notEqual(a: WebGpuTensor, b: WebGpuTensor) { return this.compare("ne", "a != b", a, b); }
  less(a: WebGpuTensor, b: WebGpuTensor) { return this.compare("lt", "a < b", a, b); }
  lessEqual(a: WebGpuTensor, b: WebGpuTensor) { return this.compare("le", "a <= b", a, b); }
  greater(a: WebGpuTensor, b: WebGpuTensor) { return this.compare("gt", "a > b", a, b); }
  greaterEqual(a: WebGpuTensor, b: WebGpuTensor) { return this.compare("ge", "a >= b", a, b); }
  logicalAnd(a: WebGpuTensor, b: WebGpuTensor) {
    const c: CType = isFloat(promote(a.dtype, b.dtype)) ? "f32" : "i32";
    return this.compare("and", c === "f32" ? "a != 0.0 && b != 0.0" : "a != 0 && b != 0", a, b);
  }
  logicalOr(a: WebGpuTensor, b: WebGpuTensor) {
    const c: CType = isFloat(promote(a.dtype, b.dtype)) ? "f32" : "i32";
    return this.compare("or", c === "f32" ? "a != 0.0 || b != 0.0" : "a != 0 || b != 0", a, b);
  }
  logicalNot(x: WebGpuTensor) {
    const c: CType = isFloat(x.dtype) ? "f32" : "i32";
    return this.nary("not", c === "f32" ? "select(0.0, 1.0, a == 0.0)" : "select(0, 1, a == 0)", [x], "bool", c);
  }
  sqrt(x: WebGpuTensor) { return this.unaryFloat("sqrt", "sqrt(a)", x); }
  rsqrt(x: WebGpuTensor) { return this.unaryFloat("rsqrt", "inverseSqrt(a)", x); }
  /** Clamped: tanh(±15) is ±1 in f32, and some drivers overflow exp inside tanh for large |x|. */
  tanh(x: WebGpuTensor) { return this.unaryFloat("tanh", "tanh(clamp(a, -15.0, 15.0))", x); }
  sigmoid(x: WebGpuTensor) { return this.unaryFloat("sigmoid", "1.0 / (1.0 + exp(-a))", x); }
  erf(x: WebGpuTensor) { return this.unaryFloat("erf", "erf_mp(a)", x, 0, ERF_HELPERS); }
  pow(a: WebGpuTensor, b: WebGpuTensor) {
    let out = promote(a.dtype, b.dtype);
    if (!isFloat(out)) out = "f32";
    return this.nary("pow", "pow_(a, b)", [a, b], out, "f32", 0, POW_HELPERS);
  }
  neg(x: WebGpuTensor) {
    if (isFloat(x.dtype)) return this.unaryFloat("neg", "-a", x);
    return this.nary("neg", "-a", [x], "i32", "i32");
  }
  abs(x: WebGpuTensor) {
    if (isFloat(x.dtype)) return this.unaryFloat("abs", "abs(a)", x);
    return this.nary("abs", "abs(a)", [x], "i32", "i32");
  }
  mean(x: WebGpuTensor, axis: number, keepDims?: boolean) { return this.reduce("mean", x, axis, keepDims); }
  min(x: WebGpuTensor, axis: number, keepDims?: boolean) { return this.reduce("min", x, axis, keepDims); }

  private argReduce(op: "argmax" | "argmin", x: WebGpuTensor, axis: number, keepDims = false): WebGpuTensor {
    this.live(x);
    const { a, outer, R, inner } = this.axis3(x, axis);
    const shape = keepDims ? x.shape.map((d, i) => (i === a ? 1 : d)) : x.shape.filter((_, i) => i !== a);
    const out = this.alloc(shape, "i32");
    const n = outer * inner;
    if (!n || !R) return out;
    const ik = this.kind(x.dtype);
    this.run(`${op}:${keyOf(ik)}`, () => argReduceKernel(op, ik), [x.storage.buffer, out.storage.buffer], { n, R, inner, off: x.offset }, this.flatGroups(n));
    return out;
  }
  argmax(x: WebGpuTensor, axis: number, keepDims?: boolean) { return this.argReduce("argmax", x, axis, keepDims); }
  argmin(x: WebGpuTensor, axis: number, keepDims?: boolean) { return this.argReduce("argmin", x, axis, keepDims); }

  cumsum(x: WebGpuTensor, axis: number): WebGpuTensor {
    this.live(x);
    const { outer, R, inner } = this.axis3(x, axis);
    const outDtype: DType = isFloat(x.dtype) ? x.dtype : "i32";
    const out = this.alloc(x.shape, outDtype);
    const n = outer * inner;
    if (!n || !R) return out;
    const ik = this.kind(x.dtype), ok = this.kind(outDtype);
    this.run(`cumsum:${keyOf(ik, ok)}`, () => cumsumKernel(ik, ok), [x.storage.buffer, out.storage.buffer], { n, R, inner, off: x.offset }, this.flatGroups(n));
    return out;
  }
}
