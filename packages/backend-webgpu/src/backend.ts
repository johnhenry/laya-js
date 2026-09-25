import type { Backend, DType, HostQuantized, HostTensor, QuantBits, QuantizedLinearOptions, QuantizedTensor, QuantMode, Shape, Tensor } from "@johnhenry/tensor-backend";
import { f32ToBf16Bits, packQuantized } from "@johnhenry/tensor-backend";
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
  gemmQmvKernel,
  gemmSgKernel,
  grid,
  layerNormKernel,
  meanPoolKernel,
  quantGatherKernel,
  quantKey,
  type QuantSpec,
  naryKernel,
  reduceKernel,
  ropeKernel,
  sdpaConfig,
  sdpaFastBytes,
  sdpaKernel,
  sdpaFastKernel,
  softmaxColKernel,
  softmaxRowKernel,
  sortKernel,
  sortSlowKernel,
  splitKReduceKernel,
  type CType,
  type GemmConfig,
  type Kind,
  type SgGemmConfig,
  type SkinnyGemmConfig,
  type QmvGemmConfig,
  type QuantGemmConfig,
  QUANT_GEMM_DEFAULT,
  QUANT_GEMM_NAVIGATOR,
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
  /** Previously measured `tuneGemm` choices to restore (merged into the device's table). */
  gemmTuning?: Record<string, GemmChoice>;
  /**
   * Before awaiting a readback, sleep for most of the GPU time the same
   * amount of work took last time, instead of letting the runtime poll
   * (Dawn-node polls in a busy loop). Default: true for Node/Bun (Dawn),
   * false for navigator.gpu.
   */
  sleepWhileWaiting?: boolean;
  /** With `sleepWhileWaiting`, only sleep when the expected wait exceeds this many milliseconds (default 3). */
  sleepThresholdMs?: number;
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

/** Subgroup-matrix tile for quantized Linears with 16 < M ≤ 64. */
const QUANT_SG_SMALL_M: SgGemmConfig = { BM: 32, BN: 64, BK: 8, WM: 1, WN: 2, pad: 0 };

type LinearPick = { skinny?: SkinnyGemmConfig; sg?: SgGemmConfig; qmv?: QmvGemmConfig };

/** A Linear kernel choice: skinny, direct, qmv (quantized only), or an index into `GemmConfig.sg`. */
export type GemmChoice = "skinny" | "direct" | "qmv" | number;
const tunings = new WeakMap<GPUDevice, Map<string, GemmChoice>>();
const inRange = (v: number, [lo, hi]: readonly [number, number]) => v >= lo && v <= hi;
const tuneKey = (k: Kind, M: number, N: number, K: number) => `${k.st}:${M}x${N}x${K}`;

export class WebGpuBackend implements Backend<WebGpuTensor> {
  readonly name = "webgpu";
  readonly rt: Runtime;
  readonly hasF16: boolean;
  /** f32 8×8×8 subgroup matrices usable (Dawn chromium-experimental-subgroup-matrix, subgroup size 32). */
  readonly hasSubgroupMatrix: boolean;
  /** WGSL `subgroups` usable with a fixed subgroup size (that size), else 0. */
  readonly subgroupSize: number;
  gemmConfig: GemmConfig;
  private scopes: Set<WebGpuTensor>[] = [];
  private ropeTables = new Map<string, Storage>();
  private destroyed = false;

  readonly device: GPUDevice;
  readonly adapterInfo: AdapterSummary;
  private readonly ownsDevice: boolean;

  constructor(device: GPUDevice, adapterInfo: AdapterSummary, opts: WebGpuBackendOptions & { f16: boolean; ownsDevice: boolean; subgroupMatrix?: boolean; subgroupSize?: number }) {
    this.device = device;
    this.adapterInfo = adapterInfo;
    this.hasF16 = opts.f16;
    this.hasSubgroupMatrix = opts.subgroupMatrix ?? false;
    this.subgroupSize = opts.subgroupSize ?? 0;
    this.ownsDevice = opts.ownsDevice;
    this.rt = new Runtime(device, opts.maxBatch ?? 128, opts.maxPooledBytes ?? 2 ** 30, opts.firstBatch ?? 24);
    // navigator.gpu (browsers, Deno): the quantized choice measured in Chromium
    this.gemmConfig = opts.gemm ?? (adapterInfo.source === "navigator.gpu" ? { ...GEMM_DEFAULT, quant: QUANT_GEMM_NAVIGATOR } : GEMM_DEFAULT);
    for (const [k, v] of Object.entries(opts.gemmTuning ?? {})) this.gemmTuning.set(k, v);
    this.rt.sleepWhileWaiting = opts.sleepWhileWaiting ?? adapterInfo.source !== "navigator.gpu";
    if (opts.sleepThresholdMs !== undefined) this.rt.sleepThresholdMs = opts.sleepThresholdMs;
  }

  /**
   * f32/bf16/i32/bool/u32 work on every WebGPU implementation; f16 needs
   * native shader-f16. Every other dtype added to the contract 2026-09-25
   * (i8/u8/i16/u16/i64/u64/f64) is permanently unsupported here -- not a
   * gap to close, a real WGSL spec limit (see README "Limitations"):
   * i8/u8/i16/u16 aren't in the WGSL spec at all (an open, unresolved
   * proposal: gpuweb/gpuweb#5152); i64/u64 exist only behind the
   * non-standard, browser-unreliable `SHADER_INT64` native wgpu feature;
   * f64 has no WGSL type at all. u32 is the one dtype added 2026-09-25 that
   * WebGPU genuinely gains, since it's already a real core WGSL type.
   */
  supports(dtype: DType): boolean {
    switch (dtype) {
      case "f16":
        return this.hasF16;
      case "f32":
      case "bf16":
      case "i32":
      case "bool":
      case "u32":
        return true;
      case "u8":
      case "i8":
      case "u16":
      case "i16":
      case "u64":
      case "i64":
      case "f64":
        return false;
    }
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
        return { st: "u32", bool: true };
      case "u32":
        return { st: "u32" };
      case "u8":
      case "i8":
      case "u16":
      case "i16":
      case "u64":
      case "i64":
      case "f64":
        throw new TypeError(`backend-webgpu: dtype ${d} is not supported (WGSL has no ${d} type; see README "Limitations")`);
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
    if (--t.storage.refs === 0 && !t.storage.external) this.rt.release(t.storage.buffer, t.storage.bytes);
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
      case "u32":
        data = d instanceof Uint32Array ? d : Uint32Array.from(d as ArrayLike<number>);
        break;
      case "u8":
      case "i8":
      case "u16":
      case "i16":
      case "u64":
      case "i64":
      case "f64":
        // Unreachable: this.kind(h.dtype) above already threw for these.
        throw new TypeError(`backend-webgpu: dtype ${h.dtype} is not supported`);
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
      case "u32":
        return { dtype: "u32", shape, data: new Uint32Array(raw) };
      case "u8":
      case "i8":
      case "u16":
      case "i16":
      case "u64":
      case "i64":
      case "f64":
        // Unreachable: this.kind(t.dtype) above already threw for these.
        throw new TypeError(`backend-webgpu: dtype ${t.dtype} is not supported`);
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

  private nary(op: string, expr: string, xs: WebGpuTensor[], outDtype: DType, c: CType, s = 0, helpers = "", names?: readonly string[]): WebGpuTensor {
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
    this.run(key, () => naryKernel(op, expr, ins, ok, c, helpers, names), [...xs.map((x) => x.storage.buffer), out.storage.buffer], params, this.flatGroups(n));
    return out;
  }

  // ---- extension hooks -----------------------------------------------------

  /**
   * An uninitialised tensor of `shape`/`dtype` from the runtime's buffer
   * pool, tracked by the enclosing `scope` like any op result. For custom
   * kernels (`rt.kernel` / `rt.dispatch`) that write their own output.
   */
  empty(shape: Shape, dtype: DType): WebGpuTensor {
    return this.alloc(shape, dtype);
  }

  /**
   * A tensor view of a `GPUBuffer` you own (no copy). It must hold
   * `offset + numel(shape)` elements in this backend's storage for `dtype`
   * (4 bytes each; f16 is 2 bytes when `hasF16`) and carry STORAGE (plus
   * COPY_SRC to be read). `dispose` never pools or destroys it — the buffer
   * stays yours; keep it alive while the view (or work reading it) is in use.
   */
  wrapBuffer(buffer: GPUBuffer, shape: Shape, dtype: DType, offset = 0): WebGpuTensor {
    const need = (offset + numel(shape)) * this.bytesPer(dtype);
    if (buffer.size < need) throw new RangeError(`wrapBuffer: [${shape}] ${dtype} at offset ${offset} needs ${need} bytes, the buffer has ${buffer.size}`);
    return this.track(new WebGpuTensor([...shape], dtype, new Storage(buffer, buffer.size, true), offset));
  }

  /**
   * A custom n-ary elementwise kernel with NumPy broadcasting: `expr` is an
   * f32-valued WGSL expression over the inputs `x0, x1, …` (loaded and
   * computed as f32; i32/bool inputs are converted) whose value is stored in
   * `outDtype` (default f32, rounded once on store; for `"bool"` nonzero is
   * true, so wrap a comparison as `select(0.0, 1.0, …)`). `helpers` is WGSL placed before
   * the entry point (functions, constants). Compiled once per
   * (`expr`, `helpers`, input kinds and layouts); one dispatch. Up to the
   * device's storage-buffer limit minus one inputs (the WebGPU default is 8
   * bindings; devices from `createWebGpuBackend` raise it to the adapter's).
   */
  elementwise(expr: string, xs: readonly WebGpuTensor[], opts: { outDtype?: DType; helpers?: string } = {}): WebGpuTensor {
    if (!xs.length) throw new Error("elementwise: needs at least one input");
    const helpers = opts.helpers ?? "";
    const names = xs.map((_, j) => `x${j}`);
    return this.nary(`custom:${expr}:${helpers}`, expr, [...xs], opts.outDtype ?? "f32", "f32", 0, helpers, names);
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
    quant: { spec: QuantSpec; bufs: GPUBuffer[] } | null = null,
  ): WebGpuTensor {
    // a quantized B (u32 words) takes a's float dtype; its values are read through qw4()
    const outDtype = [a.dtype, ...(quant ? [] : [b.dtype]), ...(bias ? [bias.dtype] : [])].reduce(promote);
    const outFloat = isFloat(outDtype) ? outDtype : "f32";
    const out = this.alloc(outShape, outFloat);
    const batch = numel(batchShape);
    if (!M || !N || !batch) return out;
    const ak = this.kind(a.dtype), bk = this.kind(b.dtype), ok = this.kind(outFloat);
    const biask = bias ? this.kind(bias.dtype) : null;
    const q = quant?.spec ?? null;
    const qk = q ? `:${quantKey(q)}` : "";
    const bBufs = quant ? quant.bufs : [b.storage.buffer];
    const vecA = K % 4 === 0 && a.offset % 4 === 0 && aBatchStrides.every((s) => s % 4 === 0);
    const vecB = (transB ? K % 4 === 0 : N % 4 === 0) && b.offset % 4 === 0 && bBatchStrides.every((s) => s % 4 === 0);
    const linearPath = transB && vecA && vecB && batch === 1;
    if (q && !linearPath) throw new Error("webgpu quantizedLinear: needs K % 4 == 0 and an aligned, unbatched x");
    const choice = linearPath ? this.pickLinear(M, N, K, ak, q) : undefined;
    const qmv = choice?.qmv;
    if (qmv && q) {
      // M split into balanced blocks of at most qmv.MT rows (grid.y)
      const blocks = Math.ceil(M / qmv.MT), MT = Math.ceil(M / blocks);
      const sub = !!qmv.sub && this.subgroupSize >= qmv.TK && this.subgroupSize % qmv.TK === 0;
      const cfg = sub === !!qmv.sub ? qmv : { ...qmv, sub };
      this.run(`gemmqmv:${keyOf(ak, biask, ok)}:${qmv.TK}x${qmv.NR}x${qmv.R}/${qmv.C}/${MT}${sub ? "s" : ""}${qk}`, () => gemmQmvKernel(ak, biask, ok, q, cfg, MT),
        [a.storage.buffer, ...bBufs, ...(bias ? [bias.storage.buffer] : []), out.storage.buffer],
        { M, N, K, oa: a.offset, obias: bias?.offset ?? 0 },
        [Math.ceil(N / (qmv.NR * qmv.R)), blocks, 1]);
      return out;
    }
    const sk = choice?.skinny;
    if (sk) {
      const TM = Math.ceil(M / sk.WY);
      const q8step = !!q && K % 8 === 0 && q.g % 8 === 0;
      this.run(`gemmskinny:${keyOf(ak, bk, biask, ok, sk, TM)}${qk}${q8step ? ":8" : ""}`, () => gemmSkinnyKernel(ak, bk, biask, ok, sk, TM, q, q8step),
        [a.storage.buffer, ...bBufs, ...(bias ? [bias.storage.buffer] : []), out.storage.buffer],
        { M, N, K, oa: a.offset, ob: b.offset, obias: bias?.offset ?? 0 },
        [Math.ceil(N / (sk.WX * sk.TN)), 1, 1]);
      return out;
    }
    const sg = choice?.sg;
    if (sg) {
      const wide = (sg.wide ?? true) && ak.st === "f16" && K % 8 === 0 && a.offset % 8 === 0 && (q ? q.g % 8 === 0 : bk.st === "f16" && b.offset % 8 === 0);
      const cfgKey = `${sg.BM}x${sg.BN}x${sg.BK}/${sg.WM}x${sg.WN}/${sg.db ?? false}/${sg.epi ?? "frag"}/${sg.pad ?? 4}/${wide}${qk}`;
      const grid: [number, number, number] = [Math.ceil(N / sg.BN), Math.ceil(M / sg.BM), 1];
      const groups = grid[0] * grid[1];
      const S = Math.min(sg.splitK?.find((c) => M <= (c.maxM ?? Infinity) && groups <= (c.maxGroups ?? Infinity))?.S ?? 1, Math.floor(K / sg.BK));
      if (S <= 1) {
        this.run(`gemmsg:${keyOf(ak, bk, biask, ok)}:${cfgKey}`, () => gemmSgKernel(ak, bk, biask, ok, sg, false, wide, q),
          [a.storage.buffer, ...bBufs, ...(bias ? [bias.storage.buffer] : []), out.storage.buffer],
          { M, N, K, kc: K, oa: a.offset, ob: b.offset, obias: bias?.offset ?? 0 }, grid);
        return out;
      }
      // Split-K: f32 partials [S, M, N], then one reduction (+ bias, rounded once).
      const kc = Math.ceil(K / S / sg.BK) * sg.BK;
      grid[2] = Math.ceil(K / kc);
      const f32: Kind = { st: "f32" };
      const part = this.rt.acquire(grid[2] * M * N * 4);
      this.run(`gemmsg:${keyOf(ak, bk, "-", f32)}:${cfgKey}:split`, () => gemmSgKernel(ak, bk, null, f32, sg, true, wide, q),
        [a.storage.buffer, ...bBufs, part.buffer],
        { M, N, K, kc, oa: a.offset, ob: b.offset, obias: 0 }, grid);
      const n = M * N;
      this.run(`splitk:${keyOf(biask, ok)}:${grid[2]}`, () => splitKReduceKernel(biask, ok, grid[2]),
        [part.buffer, ...(bias ? [bias.storage.buffer] : []), out.storage.buffer],
        { n, N, obias: bias?.offset ?? 0 }, this.flatGroups(n));
      this.rt.release(part.buffer, part.bytes);
      return out;
    }
    const dc = this.gemmConfig.direct;
    if (dc && linearPath) {
      this.run(`gemmdirect:${keyOf(ak, bk, biask, ok, dc)}${qk}`, () => gemmDirectKernel(ak, bk, biask, ok, dc, q),
        [a.storage.buffer, ...bBufs, ...(bias ? [bias.storage.buffer] : []), out.storage.buffer],
        { M, N, K, oa: a.offset, ob: b.offset, obias: bias?.offset ?? 0 },
        [Math.ceil(N / (dc.WX * dc.TN)), Math.ceil(M / (dc.WY * dc.TM)), 1]);
      return out;
    }
    const cfg = this.gemmConfig.tiled;
    const key = `gemm:${keyOf(ak, bk, biask, ok)}:${transB}:${vecA}:${vecB}:${cfg.BM}x${cfg.BN}x${cfg.BK}/${cfg.TM}x${cfg.TN}${qk}`;
    const bufs = [a.storage.buffer, ...bBufs, ...(bias ? [bias.storage.buffer] : []), out.storage.buffer];
    if (batch > 65535) throw new Error("gemm: batch > 65535");
    this.run(key, () => gemmKernel(ak, bk, biask, ok, transB, vecA, vecB, cfg, q), bufs, {
      M, N, K, oa: a.offset, ob: b.offset, obias: bias?.offset ?? 0,
      bsh: pad8(batchShape, 1), ast: pad8(aBatchStrides, 0), bst: pad8(bBatchStrides, 0),
    }, [Math.ceil(N / cfg.BN), Math.ceil(M / cfg.BM), batch]);
    return out;
  }

  /**
   * Kernel for a Linear (transB, vec4-aligned, unbatched) of shape (M, N, K):
   * a `tuneGemm` measurement for this exact shape if there is one; for
   * quantized weights with `gemmConfig.quant`, its qmv / subgroup-matrix tile
   * rules (`quantPick`); else the first skinny config with M ≤ maxM, else
   * the first subgroup-matrix config whose M range, minimum workgroup count
   * and row-padding limit all hold, else the direct/tiled kernels (empty
   * choice).
   */
  private pickLinear(M: number, N: number, K: number, ak: Kind, q: QuantSpec | null = null): LinearPick {
    const cfg = this.gemmConfig;
    const quant = !!q;
    const sgs = this.hasSubgroupMatrix ? cfg.sg ?? [] : [];
    const tuned = this.gemmTuning.get((quant ? "q" : "") + tuneKey(ak, M, N, K));
    if (tuned !== undefined) {
      if (tuned === "qmv") {
        const qs = (cfg.quant ?? QUANT_GEMM_DEFAULT).qmv.filter((c) => q && K % c.C === 0 && q.g % c.C === 0);
        const c = qs.find((c) => M <= c.maxM) ?? qs.at(-1);
        if (c) return { qmv: c };
      } else if (tuned === "skinny") {
        const sk = cfg.skinny.find((c) => M <= c.maxM) ?? cfg.skinny[cfg.skinny.length - 1];
        if (sk) return { skinny: sk };
      } else if (tuned === "direct") return {};
      else if (sgs[tuned]) return { sg: sgs[tuned] };
    }
    if (q && cfg.quant) {
      const p = this.quantPick(M, N, K, q, cfg.quant);
      if (p) return p;
    } else if (q) {
      // 0.4.0 rules (measured on M2, bench/quantized-gemm.ts): skinny for the
      // smallest M; up to 64 rows the subgroup-matrix kernel with split-K 2,
      // else the wide-row skinny config.
      if (M <= 16 && cfg.skinny[0]) return { skinny: cfg.skinny[0] };
      if (M <= 64) {
        if (sgs[0]) return { sg: { ...QUANT_SG_SMALL_M, splitK: [{ S: 2 }] } };
        const sk = cfg.skinny.find((c) => c.WY >= 8 && M <= c.maxM) ?? cfg.skinny.find((c) => M <= c.maxM);
        if (sk) return { skinny: sk };
      }
    }
    const skinny = cfg.skinny.find((c) => M <= c.maxM);
    if (skinny) return { skinny };
    const sg = sgs.find((c) =>
      M > c.minM && M <= (c.maxM ?? Infinity) &&
      (c.minGroups === undefined || Math.ceil(M / c.BM) * Math.ceil(N / c.BN) >= c.minGroups) &&
      (c.maxPad === undefined || (Math.ceil(M / c.BM) * c.BM) / M <= c.maxPad) &&
      (c.skipGroups === undefined || !inRange(Math.ceil(M / c.BM) * Math.ceil(N / c.BN), c.skipGroups)));
    return sg ? { sg } : {};
  }

  /** Quantized-weight rules of `QuantGemmConfig` (undefined: fall through to the float rules). */
  private quantPick(M: number, N: number, K: number, q: QuantSpec, qc: QuantGemmConfig): LinearPick | undefined {
    const sgOk = this.hasSubgroupMatrix && !!qc.sg;
    const qmv = qc.qmv.find((c) => M <= c.maxM && (c.bits === undefined || c.bits === q.bits) && K % c.C === 0 && q.g % c.C === 0 && !(sgOk && c.maxN !== undefined && N > c.maxN));
    if (qmv) return { qmv };
    if (!sgOk) return undefined;
    const r = qc.sg!;
    const groups = (BM: number) => Math.ceil(M / BM) * Math.ceil(N / 64);
    const tile = (BM: number, S = 1): SgGemmConfig => ({ BM, BN: 64, BK: 8, WM: 1, WN: 2, pad: 0, ...(S > 1 ? { splitK: [{ S }] } : {}) });
    if (Math.ceil(M / 64) * 64 <= r.maxPad64 * M && groups(64) >= r.minGroups64) return { sg: tile(64) };
    const pad = (BM: number) => Math.ceil(M / BM) * BM;
    const least = Math.min(...r.rows.map(pad));
    const rows = [...r.rows].sort((a, b) => b - a).filter((BM) => pad(BM) <= 1.1 * least);
    const tall = rows.find((BM) => groups(BM) >= r.minGroups);
    if (tall !== undefined) return { sg: tile(tall) };
    const BM = rows[rows.length - 1]!;
    const S = Math.min(r.maxSplit, Math.ceil(r.minGroups / groups(BM)));
    return { sg: tile(BM, S) };
  }

  /**
   * Measured (not modelled) Linear kernel choices, by `${storage}:${M}x${N}x${K}`:
   * "skinny", "direct" or an index into `gemmConfig.sg`. Filled by
   * `tuneGemm`, shared by every backend on the same GPUDevice; export it
   * with `Object.fromEntries` and restore it with `gemmTuning` (option) to
   * skip re-tuning.
   */
  get gemmTuning(): Map<string, GemmChoice> {
    let m = tunings.get(this.device);
    if (!m) tunings.set(this.device, (m = new Map()));
    return m;
  }

  /**
   * Autotunes the Linear kernel per shape on this device: for each
   * {M, N, K} (x [M, K] · w [N, K]ᵀ, in `dtype`), times every applicable
   * candidate (skinny for M ≤ 128, each `gemmConfig.sg` entry, the direct
   * kernel) in interleaved rounds, and records the fastest in `gemmTuning`.
   * With `quantized`, tunes `quantizedLinear` for weights of that format
   * instead (keys prefixed "q"): the candidates are the built-in quantized
   * choice, qmv, skinny, each `gemmConfig.sg` entry and direct, and a
   * choice is recorded only when it beats the built-in one. Takes ~10–50 ms
   * per shape. Returns the recorded choices.
   */
  async tuneGemm(
    shapes: readonly { M: number; N: number; K: number }[],
    opts: { dtype?: "f16" | "f32"; rounds?: number; quantized?: { bits: QuantBits; groupSize?: number; mode?: QuantMode } } = {},
  ): Promise<Record<string, GemmChoice>> {
    const dtype = opts.dtype ?? (this.hasF16 ? "f16" : "f32");
    const ak = this.kind(dtype);
    const rounds = opts.rounds ?? 3;
    const qo = opts.quantized;
    const out: Record<string, GemmChoice> = {};
    const rnd = (n: number, sc: number) => Float32Array.from({ length: n }, (_, i) => (((i * 2654435761) >>> 0) / 2 ** 32 - 0.5) * sc);
    for (const { M, N, K } of shapes) {
      if (K % 4) continue;
      const key = (qo ? "q" : "") + tuneKey(ak, M, N, K);
      // undefined: the built-in choice (quantized only)
      const cands: (GemmChoice | undefined)[] = qo ? [undefined] : [];
      if (qo && (this.gemmConfig.quant ?? QUANT_GEMM_DEFAULT).qmv.length) cands.push("qmv");
      if (M <= 128 && this.gemmConfig.skinny.length) cands.push("skinny");
      (this.hasSubgroupMatrix ? this.gemmConfig.sg ?? [] : []).forEach((_, i) => cands.push(i));
      if (this.gemmConfig.direct) cands.push("direct");
      if (cands.length < 2) continue;
      const x = await this.fromHost({ dtype: "f32", shape: [M, K], data: rnd(M * K, 2) });
      const xs = this.cast(x, dtype);
      let run: () => WebGpuTensor, free: () => void;
      if (qo) {
        const groupSize = qo.groupSize ?? 64, mode = qo.mode ?? (qo.bits === 8 ? "symmetric" : "affine");
        const G = Math.ceil(K / groupSize);
        const lo = mode === "symmetric" ? 1 - (1 << (qo.bits - 1)) : 0, span = mode === "symmetric" ? (1 << qo.bits) - 1 : 1 << qo.bits;
        const qv = Int32Array.from({ length: N * K }, (_, i) => lo + (((i * 2654435761) >>> 8) % span));
        const sc = Float16Array.from({ length: N * G }, (_, i) => 1e-3 * (1 + (i % 7) / 7));
        const h: HostQuantized = {
          shape: [N, K], bits: qo.bits, groupSize, mode, data: packQuantized(qv, qo.bits),
          scales: { dtype: "f16", shape: [N, G], data: sc },
          biases: mode === "affine" ? { dtype: "f16", shape: [N, G], data: sc.map((v) => -8 * v) } : null,
        };
        const q = await this.fromHostQuantized(h, dtype);
        if (!q) {
          for (const t of [x, xs]) this.dispose(t);
          continue;
        }
        const qopts = { bits: qo.bits, groupSize, mode };
        run = () => this.quantizedLinear(xs, q.w, q.scales, q.biases, qopts);
        free = () => [q.w, q.scales, ...(q.biases ? [q.biases] : [])].forEach((t) => this.dispose(t));
      } else {
        const w = await this.fromHost({ dtype: "f32", shape: [N, K], data: rnd(N * K, 0.06) });
        const ws = this.cast(w, dtype);
        run = () => this.linear(xs, ws);
        free = () => [w, ws].forEach((t) => this.dispose(t));
      }
      const iters = Math.max(2, Math.min(20, Math.round(4e9 / (2 * M * N * K))));
      const best = cands.map(() => Infinity);
      const prev = this.gemmTuning.get(key);
      const set = (c: GemmChoice | undefined) => (c === undefined ? this.gemmTuning.delete(key) : this.gemmTuning.set(key, c));
      try {
        for (let r = 0; r < rounds; r++) {
          for (let c = 0; c < cands.length; c++) {
            set(cands[c]);
            this.dispose(run());
            await this.sync();
            const t0 = performance.now();
            for (let i = 0; i < iters; i++) this.dispose(run());
            await this.sync();
            best[c] = Math.min(best[c]!, performance.now() - t0);
          }
        }
      } finally {
        set(prev);
        for (const t of [x, xs]) this.dispose(t);
        free();
      }
      const pick = cands[best.indexOf(Math.min(...best))];
      set(pick);
      if (pick !== undefined) out[key] = pick;
    }
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

  // ---- quantized weights ---------------------------------------------------

  /**
   * Keeps a quantized matrix packed on the device: the laya-js bytes upload
   * as-is (u32 words [out, in·bits/32]; read as little-endian words they are
   * the MLX packing), scales / biases as `dtype` (f16 storage when the device
   * has shader-f16). Every GEMM path (skinny, subgroup-matrix, direct, tiled)
   * dequantizes inside its B tile load and accumulates in f32. Resolves to
   * null when the group size is not a multiple of 4.
   */
  async fromHostQuantized(h: HostQuantized, dtype: DType): Promise<QuantizedTensor<WebGpuTensor> | null> {
    const [N, K] = h.shape;
    if (h.groupSize % 4 || (K * h.bits) % 32) return null;
    if (!isFloat(dtype)) throw new TypeError(`webgpu fromHostQuantized: dtype must be float, got ${dtype}`);
    const words = [N, (K * h.bits) / 32];
    const { buffer, bytes, writeHazard } = this.rt.acquire(Math.max(4, h.data.byteLength));
    this.rt.write(buffer, writeHazard, h.data);
    const w = this.track(new WebGpuTensor(words, "i32", new Storage(buffer, bytes), 0));
    const up = (t: HostTensor) => {
      const u = this.upload(t);
      if (u.dtype === dtype) return u;
      const c = this.cast(u, dtype);
      this.dispose(u);
      return c;
    };
    const { bits, groupSize, mode } = h;
    return { shape: [N, K], bits, groupSize, mode, dtype, native: true, w, scales: up(h.scales), biases: h.biases ? up(h.biases) : null };
  }

  private quantSpec(scales: WebGpuTensor, biases: WebGpuTensor | null, opts: QuantizedLinearOptions, r16 = false): { spec: QuantSpec; bufs: GPUBuffer[] } {
    const sym = opts.mode === "symmetric";
    if (!sym && !biases) throw new Error("webgpu: affine quantized weights need biases");
    if (opts.groupSize % 4) throw new Error("webgpu: quantized group size must be a multiple of 4");
    this.live(scales);
    if (biases) this.live(biases);
    return {
      spec: { bits: opts.bits, g: opts.groupSize, sym, scale: this.kind(scales.dtype), ...(r16 ? { r16 } : {}) },
      bufs: [scales.storage.buffer, ...(sym ? [] : [biases!.storage.buffer])],
    };
  }

  quantizedLinear(x: WebGpuTensor, w: WebGpuTensor, scales: WebGpuTensor, biases: WebGpuTensor | null, opts: QuantizedLinearOptions, bias?: WebGpuTensor | null): WebGpuTensor {
    this.live(x);
    this.live(w);
    if (bias) this.live(bias);
    if (!isFloat(x.dtype)) throw new Error(`quantizedLinear: x must be float, got ${x.dtype}`);
    const K = x.shape[x.shape.length - 1]!;
    const N = scales.shape[0]!;
    if (w.shape[0] !== N || (w.shape[1]! * 32) / opts.bits !== K) throw new Error(`quantizedLinear: x [${x.shape}] vs packed w [${w.shape}] (q${opts.bits})`);
    // f16 activations: round each weight to f16 like host dequantization does
    const q = this.quantSpec(scales, biases, opts, this.kind(x.dtype).st === "f16");
    q.bufs.unshift(w.storage.buffer);
    const M = x.size / K;
    const xc = x.offset % 4 ? this.copyContig(x) : x;
    try {
      return this.gemm(xc, w, bias ?? null, M, N, K, true, [], [], [], [...x.shape.slice(0, -1), N], q);
    } finally {
      if (xc !== x) this.dispose(xc);
    }
  }

  quantizedEmbedding(w: WebGpuTensor, scales: WebGpuTensor, biases: WebGpuTensor | null, opts: QuantizedLinearOptions, ids: WebGpuTensor): WebGpuTensor {
    this.live(w);
    this.live(ids);
    const V = scales.shape[0]!, K = (w.shape[1]! * 32) / opts.bits;
    const q = this.quantSpec(scales, biases, opts);
    const idsI = ids.dtype === "i32" ? ids : this.cast(ids, "i32");
    const out = this.alloc([...ids.shape, K], scales.dtype);
    const n = (ids.size * K) / 4;
    if (n) {
      const ok = this.kind(scales.dtype);
      this.run(`qgather:${quantKey(q.spec)}:${keyOf(ok)}`, () => quantGatherKernel(q.spec, ok), [w.storage.buffer, ...q.bufs, idsI.storage.buffer, out.storage.buffer], {
        n, K, V, oi: idsI.offset,
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
    // Both kernels respect the device's workgroup-memory limit: the fast one
    // only runs where it fits (D = 64 needs ~20 KiB), and the generic one
    // shrinks its tiles until it fits (the WebGPU default is 16 KiB).
    const limit = this.device.limits.maxComputeWorkgroupStorageSize ?? 16384;
    const fast = (D === 32 || D === 64) && q.offset % 4 === 0 && k.offset % 4 === 0 && v.offset % 4 === 0 && sdpaFastBytes(D, m !== null) <= limit;
    if (fast) {
      this.run(`sdpafast:${keyOf(qk, kk, vk, mk, ok)}:${D}`, () => sdpaFastKernel(qk, kk, vk, mk, ok, D), bufs, params, [Math.ceil(Lq / 32), H, B]);
    } else {
      const { BQ, BKV } = sdpaConfig(D, limit);
      this.run(`sdpa:${keyOf(qk, kk, vk, mk, ok)}:${D}:${BQ}x${BKV}`, () => sdpaKernel(qk, kk, vk, mk, ok, D, limit), bufs, params, [Math.ceil(Lq / BQ), H, B]);
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
    // bool -> i32 (unchanged); every other dtype (float, i32, and u32 added
    // 2026-09-25) keeps its own dtype -- was previously force-cast to i32
    // unconditionally for any non-float input, silently discarding u32
    // (a real conformance failure this was added to fix: `cumsum/u32`
    // returning i32). Matches the same rule `sum` above already applies.
    const outDtype: DType = x.dtype === "bool" ? "i32" : x.dtype;
    const out = this.alloc(x.shape, outDtype);
    const n = outer * inner;
    if (!n || !R) return out;
    const ik = this.kind(x.dtype), ok = this.kind(outDtype);
    this.run(`cumsum:${keyOf(ik, ok)}`, () => cumsumKernel(ik, ok), [x.storage.buffer, out.storage.buffer], { n, R, inner, off: x.offset }, this.flatGroups(n));
    return out;
  }
}
