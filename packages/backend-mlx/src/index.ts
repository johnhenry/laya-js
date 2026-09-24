/**
 * @johnhenry/backend-mlx — native Apple Silicon backend for
 * @johnhenry/tensor-backend, over Apple's mlx-c via FFI (bun:ffi on Bun,
 * Deno.dlopen on Deno, koffi on Node).
 *
 * Semantics: every op appends a node to MLX's lazy graph and returns
 * immediately (one FFI call per op); `fromHost` copies the host buffer into
 * an MLX allocation at call time and resolves; `read` evaluates and copies
 * to the host; `flush` evaluates without copying. Fused ops map to `mlx.fast`
 * kernels; `compile` maps to `mlx_compile`.
 */
import type { Backend, DType, HostData, HostQuantized, HostTensor, QuantizedLinearOptions, QuantizedTensor, Shape, Tensor } from "@johnhenry/tensor-backend";
import { f32ToBf16Bits, toF32 } from "@johnhenry/tensor-backend";
import { openNative, type Native } from "./ffi.ts";
import { mlxPlatformSupported, resolveLib } from "./lib.ts";

export { libCandidates, mlxPlatformSupported, resolveLib } from "./lib.ts";

export interface MlxBackendOptions {
  /** Default "gpu" (Metal). "cpu" uses MLX's CPU backend; `compile` is then the identity. */
  device?: "gpu" | "cpu";
  /** Explicit path to libmlxc.dylib (else see `libCandidates()`). */
  libPath?: string;
  /**
   * Register every tensor with a FinalizationRegistry so leaked handles are
   * freed after GC. Default true. `dispose`/`scope` remain the primary,
   * deterministic mechanism.
   */
  finalizers?: boolean;
  /** Use an `mlx_compile`d (fused, shapeless) GELU on the GPU. Default true. */
  compiledGelu?: boolean;
}

export interface MlxTensor extends Tensor {
  readonly shape: Shape;
  readonly dtype: DType;
}

export interface MlxBackend extends Backend<MlxTensor> {
  readonly name: "mlx";
  readonly device: "gpu" | "cpu";
  /** Which libmlxc was loaded, via which FFI, and which mlx-c ABI it has. */
  readonly info: { libPath: string; runtime: "bun" | "deno" | "node"; mlxcAbi: "mlx-c<0.6 (MLX 0.32.1)" | "mlx-c>=0.6 (MLX 0.32.2)" };
  flush(...ts: MlxTensor[]): void;
  destroy(): void;
  geglu(x: MlxTensor): MlxTensor;
  meanPool(x: MlxTensor, mask: MlxTensor): MlxTensor;
  // general numerics: all native (mlx-c)
  equal(a: MlxTensor, b: MlxTensor): MlxTensor;
  notEqual(a: MlxTensor, b: MlxTensor): MlxTensor;
  less(a: MlxTensor, b: MlxTensor): MlxTensor;
  lessEqual(a: MlxTensor, b: MlxTensor): MlxTensor;
  greater(a: MlxTensor, b: MlxTensor): MlxTensor;
  greaterEqual(a: MlxTensor, b: MlxTensor): MlxTensor;
  logicalAnd(a: MlxTensor, b: MlxTensor): MlxTensor;
  logicalOr(a: MlxTensor, b: MlxTensor): MlxTensor;
  logicalNot(x: MlxTensor): MlxTensor;
  sqrt(x: MlxTensor): MlxTensor;
  rsqrt(x: MlxTensor): MlxTensor;
  pow(a: MlxTensor, b: MlxTensor): MlxTensor;
  neg(x: MlxTensor): MlxTensor;
  abs(x: MlxTensor): MlxTensor;
  tanh(x: MlxTensor): MlxTensor;
  sigmoid(x: MlxTensor): MlxTensor;
  erf(x: MlxTensor): MlxTensor;
  argmax(x: MlxTensor, axis: number, keepDims?: boolean): MlxTensor;
  argmin(x: MlxTensor, axis: number, keepDims?: boolean): MlxTensor;
  mean(x: MlxTensor, axis: number, keepDims?: boolean): MlxTensor;
  min(x: MlxTensor, axis: number, keepDims?: boolean): MlxTensor;
  cumsum(x: MlxTensor, axis: number): MlxTensor;
  // quantized weights: native (mlx quantized_matmul / dequantize, affine groups of 32/64/128)
  fromHostQuantized(h: HostQuantized, dtype: DType): Promise<QuantizedTensor<MlxTensor> | null>;
  quantizedLinear(x: MlxTensor, w: MlxTensor, scales: MlxTensor, biases: MlxTensor | null, opts: QuantizedLinearOptions, bias?: MlxTensor | null): MlxTensor;
  quantizedEmbedding(w: MlxTensor, scales: MlxTensor, biases: MlxTensor | null, opts: QuantizedLinearOptions, ids: MlxTensor): MlxTensor;
  compile<A extends MlxTensor[], R>(fn: (...args: A) => R): (...args: A) => R;
  /** Synchronous `read`. */
  readSync(t: MlxTensor): HostTensor;
  /** MLX allocator statistics in bytes. */
  memory(): { active: number; peak: number };
  /** Number of live (not yet disposed) tensor handles created by this backend. */
  liveTensors(): number;
}

// mlx_dtype enum (mlx/c/array.h)
const MLX_UINT32 = 3;
/** Group sizes MLX's affine quantization supports. */
const MLX_GROUP_SIZES = new Set([32, 64, 128]);
const MLX_DTYPE: Record<DType, number> = { bool: 0, i32: 7, f16: 9, f32: 10, bf16: 12 };
const FROM_MLX: Record<number, DType> = { 0: "bool", 7: "i32", 9: "f16", 10: "f32", 12: "bf16" };
const BYTES: Record<DType, number> = { bool: 1, i32: 4, f16: 2, f32: 4, bf16: 2 };
const TWO32 = 4294967296;

const HOST_CTOR = {
  f32: Float32Array,
  f16: Float16Array,
  bf16: Uint16Array,
  i32: Int32Array,
  bool: Uint8Array,
} as const;

class MlxArray implements MlxTensor {
  h: number;
  _shape: number[] | undefined;
  _dtype: DType | undefined;
  readonly _b: MlxBackendImpl;
  constructor(b: MlxBackendImpl, h: number, shape?: number[], dtype?: DType) {
    this._b = b;
    this.h = h;
    this._shape = shape;
    this._dtype = dtype;
  }
  get shape(): Shape {
    return (this._shape ??= this._b.queryShape(this));
  }
  get dtype(): DType {
    return (this._dtype ??= this._b.queryDtype(this));
  }
}

// Scratch layout (bytes): [0,8) op out slot · [8,16) closure-apply out slot ·
// [64,320) ints A · [320,576) ints B · [576,832) ints C ·
// [1024,4096) handle vector (up to 384 handles)
const SCRATCH_BYTES = 4096;
const APPLY_SLOT = 8, INTS_A = 64, INTS_B = 320, INTS_C = 576, HANDLES = 1024;
const MAX_HANDLES = (SCRATCH_BYTES - HANDLES) / 8;

let pendingTraceError: unknown = null;

class MlxBackendImpl implements MlxBackend {
  readonly name = "mlx" as const;
  readonly device: "gpu" | "cpu";
  readonly info: MlxBackend["info"];
  private readonly n: Native;
  private stream: number;
  private readonly scratch = new ArrayBuffer(SCRATCH_BYTES);
  private readonly u32 = new Uint32Array(this.scratch);
  private readonly i32 = new Int32Array(this.scratch);
  private readonly base: number;
  private readonly scopes: Set<MlxArray>[] = [];
  private readonly registry: FinalizationRegistry<number> | null;
  private live = 0;
  private destroyed = false;
  // one-element scalar staging arrays (mlx_array_new_data copies)
  private readonly sF32 = new Float32Array(1);
  private readonly sF16 = new Float16Array(1);
  private readonly sU16 = new Uint16Array(1);
  private readonly sI32 = new Int32Array(1);
  private readonly sU8 = new Uint8Array(1);
  private readonly sdpaMaskArray = cstr("array");
  private readonly affine = cstr("affine");
  private readonly sdpaMaskNone = cstr("");
  // compile support
  private closureTramp = 0;
  private readonly closures = new Map<number, (res: number, input: number) => number>();
  private nextClosureId = 1;
  private compiledGelu: ((x: MlxTensor) => MlxTensor) | null | undefined;
  private tracing = 0;

  constructor(opts: MlxBackendOptions = {}) {
    if (!mlxPlatformSupported()) throw new Error("backend-mlx: MLX requires macOS on Apple Silicon (darwin/arm64)");
    this.device = opts.device ?? "gpu";
    const lib = resolveLib(opts.libPath);
    this.n = openNative(lib.path);
    this.info = {
      libPath: lib.path,
      runtime: this.n.runtime,
      mlxcAbi: this.n.sdpaForceFused ? "mlx-c>=0.6 (MLX 0.32.2)" : "mlx-c<0.6 (MLX 0.32.1)",
    };
    this.base = this.n.addressOf(this.u32);
    this.stream = this.device === "gpu" ? this.n.mlx_default_gpu_stream_new() : this.n.mlx_default_cpu_stream_new();
    if (!this.stream) throw new Error(`backend-mlx: could not create ${this.device} stream: ${this.n.takeError()}`);
    if (opts.compiledGelu === false) this.compiledGelu = null;
    const n = this.n;
    this.registry = opts.finalizers === false ? null : new FinalizationRegistry<number>((h) => {
      if (h) n.mlx_array_free(h);
    });
  }

  // ---- plumbing -------------------------------------------------------------

  private wrap(h: number, shape?: number[], dtype?: DType): MlxArray {
    const t = new MlxArray(this, h, shape, dtype);
    this.live++;
    this.registry?.register(t, h, t);
    const s = this.scopes[this.scopes.length - 1];
    if (s) s.add(t);
    return t;
  }

  /** Reads and clears an out slot after a call; throws on mlx-c failure. */
  private take(rc: number, op: string, slot = 0): number {
    const u = this.u32;
    const i = slot >> 2;
    const h = u[i]! + u[i + 1]! * TWO32;
    u[i] = 0;
    u[i + 1] = 0;
    if (rc !== 0) {
      if (h) this.n.mlx_array_free(h);
      throw new Error(`backend-mlx ${op}: ${this.n.takeError() ?? "mlx-c error"}`);
    }
    return h;
  }

  private out(rc: number, op: string, shape?: number[], dtype?: DType): MlxArray {
    return this.wrap(this.take(rc, op), shape, dtype);
  }

  private h(t: MlxTensor): number {
    const a = t as MlxArray;
    if (!a.h) throw new Error("backend-mlx: tensor used after dispose");
    return a.h;
  }

  /** Writes ints at a scratch region, returns its address. */
  private ints(region: number, xs: readonly number[]): number {
    if (xs.length > 64) throw new RangeError("backend-mlx: more than 64 dims");
    const off = region >> 2;
    for (let i = 0; i < xs.length; i++) this.i32[off + i] = xs[i]!;
    return this.base + region;
  }

  /** Builds an mlx_vector_array of handles (caller frees it). */
  private vec(ts: readonly MlxTensor[]): number {
    let addr: number;
    if (ts.length <= MAX_HANDLES) {
      const off = HANDLES >> 2;
      for (let i = 0; i < ts.length; i++) {
        const h = this.h(ts[i]!);
        this.u32[off + 2 * i] = h % TWO32;
        this.u32[off + 2 * i + 1] = Math.floor(h / TWO32);
      }
      addr = this.base + HANDLES;
      const v = this.n.mlx_vector_array_new_data(addr, ts.length);
      if (!v) throw new Error(`backend-mlx: vector_array: ${this.n.takeError()}`);
      return v;
    }
    const tmp = handleBuffer(ts.map((t) => this.h(t)));
    const v = this.n.mlx_vector_array_new_data(this.n.addressOf(tmp), ts.length);
    if (!v) throw new Error(`backend-mlx: vector_array: ${this.n.takeError()}`);
    return v;
  }

  /** Unpacks an mlx_vector_array into new tensors; frees the vector if `free`. */
  private unvec(v: number, op: string, free: boolean): MlxArray[] {
    const count = this.n.mlx_vector_array_size(v);
    const out: MlxArray[] = [];
    try {
      for (let i = 0; i < count; i++) out.push(this.out(this.n.mlx_vector_array_get(this.base, v, i), op));
    } finally {
      if (free) this.n.mlx_vector_array_free(v);
    }
    return out;
  }

  queryShape(t: MlxArray): number[] {
    const nd = this.n.mlx_array_ndim(this.h(t));
    if (nd === 0) return [];
    const p = this.n.mlx_array_shape(t.h);
    const out = new Int32Array(nd);
    this.n.memcpy(out, p, nd * 4);
    return Array.from(out);
  }

  queryDtype(t: MlxArray): DType {
    const d = this.n.mlx_array_dtype(this.h(t));
    const dt = FROM_MLX[d];
    if (!dt) throw new Error(`backend-mlx: unsupported mlx dtype ${d}`);
    return dt;
  }

  private scalar(v: number, dtype: DType): MlxArray {
    let buf: HostData;
    switch (dtype) {
      case "f32": this.sF32[0] = v; buf = this.sF32; break;
      case "f16": this.sF16[0] = v; buf = this.sF16; break;
      case "bf16": this.sU16[0] = f32ToBf16Bits(v); buf = this.sU16; break;
      case "i32": this.sI32[0] = v; buf = this.sI32; break;
      case "bool": this.sU8[0] = v ? 1 : 0; buf = this.sU8; break;
    }
    const h = this.n.mlx_array_new_data(buf, this.base + INTS_A, 0, MLX_DTYPE[dtype]);
    if (!h) throw new Error(`backend-mlx scalar: ${this.n.takeError()}`);
    return this.wrap(h, [], dtype);
  }

  /** Runs `f` with a temporary scalar handle, freed right after (the graph keeps its own reference). */
  private withScalar(v: number, dtype: DType, f: (s: number) => MlxArray): MlxArray {
    const s = this.scalar(v, dtype);
    try {
      return f(s.h);
    } finally {
      this.dispose(s);
    }
  }

  // ---- Backend: transfer / lifetime ------------------------------------------

  supports(dtype: DType): boolean {
    return dtype in MLX_DTYPE;
  }

  async fromHost(t: HostTensor): Promise<MlxTensor> {
    return this.upload(t);
  }

  /** The copy behind `fromHost` (synchronous: mlx_array_new_data copies immediately). */
  private upload(t: HostTensor): MlxTensor {
    const shape = [...t.shape];
    let size = 1;
    for (const d of shape) size *= d;
    if (t.data.length !== size) throw new RangeError(`backend-mlx fromHost: ${t.data.length} values for shape [${shape}]`);
    const dt = MLX_DTYPE[t.dtype];
    if (dt === undefined) throw new TypeError(`backend-mlx: unsupported dtype ${t.dtype}`);
    // mlx_array_new_data copies once, straight from the JS buffer into an
    // MLX (unified-memory, GPU-visible) allocation.
    const data = t.data.length ? t.data : new HOST_CTOR[t.dtype](1);
    const h = this.n.mlx_array_new_data(data, this.ints(INTS_A, shape), shape.length, dt);
    if (!h) throw new Error(`backend-mlx fromHost: ${this.n.takeError()}`);
    return this.wrap(h, shape, t.dtype);
  }

  async read(t: MlxTensor): Promise<HostTensor> {
    return this.readSync(t);
  }

  readSync(t: MlxTensor): HostTensor {
    const a = t as MlxArray;
    const src = this.h(a);
    const dtype = a.dtype;
    const shape = [...a.shape];
    // Row-contiguous (a no-op when already dense) so the raw data pointer is dense.
    const c = this.take(this.n.mlx_contiguous(this.base, src, false, this.stream), "read");
    try {
      if (this.n.mlx_array_eval(c) !== 0) throw new Error(`backend-mlx eval: ${this.n.takeError()}`);
      let size = 1;
      for (const d of shape) size *= d;
      const out = new HOST_CTOR[dtype](size);
      const bytes = size * BYTES[dtype];
      if (bytes) {
        const nb = this.n.mlx_array_nbytes(c);
        if (nb !== bytes) throw new Error(`backend-mlx read: ${nb} bytes, expected ${bytes}`);
        this.n.memcpy(out, this.n.mlx_array_data_uint8(c), bytes);
      }
      return { dtype, shape, data: out };
    } finally {
      this.n.mlx_array_free(c);
    }
  }

  dispose(t: MlxTensor): void {
    const a = t as MlxArray;
    if (!a.h) return;
    this.registry?.unregister(a);
    this.n.mlx_array_free(a.h);
    a.h = 0;
    this.live--;
  }

  scope<R>(fn: () => R): R {
    const cur = new Set<MlxArray>();
    this.scopes.push(cur);
    let result: R;
    try {
      result = fn();
    } catch (e) {
      this.scopes.pop();
      for (const t of cur) this.dispose(t);
      throw e;
    }
    this.scopes.pop();
    const keep = new Set<MlxTensor>();
    collect(result, keep);
    for (const t of cur) if (!keep.has(t)) this.dispose(t);
    const parent = this.scopes[this.scopes.length - 1];
    if (parent) for (const t of keep) if (cur.has(t as MlxArray)) parent.add(t as MlxArray);
    return result;
  }

  flush(...ts: MlxTensor[]): void {
    if (ts.length === 0) {
      if (this.n.mlx_synchronize(this.stream) !== 0) throw new Error(`backend-mlx synchronize: ${this.n.takeError()}`);
      return;
    }
    const v = this.vec(ts);
    const rc = this.n.mlx_eval(v);
    this.n.mlx_vector_array_free(v);
    if (rc !== 0) throw new Error(`backend-mlx eval: ${this.n.takeError()}`);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.n.mlx_stream_free(this.stream);
    this.stream = 0;
  }

  memory(): { active: number; peak: number } {
    const u = new Uint32Array(new ArrayBuffer(16));
    const p = this.n.addressOf(u);
    this.n.mlx_get_active_memory(p);
    this.n.mlx_get_peak_memory(p + 8);
    return { active: u[0]! + u[1]! * TWO32, peak: u[2]! + u[3]! * TWO32 };
  }

  liveTensors(): number {
    return this.live;
  }

  // ---- shape --------------------------------------------------------------

  reshape(x: MlxTensor, shape: Shape): MlxTensor {
    return this.out(this.n.mlx_reshape(this.base, this.h(x), this.ints(INTS_A, shape), shape.length, this.stream), "reshape");
  }

  transpose(x: MlxTensor, perm: readonly number[]): MlxTensor {
    return this.out(this.n.mlx_transpose_axes(this.base, this.h(x), this.ints(INTS_A, perm), perm.length, this.stream), "transpose");
  }

  slice(x: MlxTensor, begin: readonly number[], end: readonly number[]): MlxTensor {
    const shape = x.shape;
    const nd = shape.length;
    const b: number[] = [], e: number[] = [], st: number[] = [];
    for (let i = 0; i < nd; i++) {
      b.push(i < begin.length ? begin[i]! : 0);
      e.push(i < end.length ? end[i]! : shape[i]!);
      st.push(1);
    }
    return this.out(
      this.n.mlx_slice(this.base, this.h(x), this.ints(INTS_A, b), nd, this.ints(INTS_B, e), nd, this.ints(INTS_C, st), nd, this.stream),
      "slice",
    );
  }

  split(x: MlxTensor, parts: number, axis: number): MlxTensor[] {
    const v = this.take(this.n.mlx_split(this.base, this.h(x), parts, axis, this.stream), "split");
    return this.unvec(v, "split", true);
  }

  concat(xs: readonly MlxTensor[], axis: number): MlxTensor {
    const v = this.vec(xs);
    try {
      return this.out(this.n.mlx_concatenate_axis(this.base, v, axis, this.stream), "concat");
    } finally {
      this.n.mlx_vector_array_free(v);
    }
  }

  cast(x: MlxTensor, dtype: DType): MlxTensor {
    const dt = MLX_DTYPE[dtype];
    if (dt === undefined) throw new TypeError(`backend-mlx: unsupported dtype ${dtype}`);
    return this.out(this.n.mlx_astype(this.base, this.h(x), dt, this.stream), "cast", undefined, dtype);
  }

  // ---- elementwise ----------------------------------------------------------

  add(a: MlxTensor, b: MlxTensor): MlxTensor {
    return this.out(this.n.mlx_add(this.base, this.h(a), this.h(b), this.stream), "add");
  }
  sub(a: MlxTensor, b: MlxTensor): MlxTensor {
    return this.out(this.n.mlx_subtract(this.base, this.h(a), this.h(b), this.stream), "sub");
  }
  mul(a: MlxTensor, b: MlxTensor): MlxTensor {
    return this.out(this.n.mlx_multiply(this.base, this.h(a), this.h(b), this.stream), "mul");
  }
  div(a: MlxTensor, b: MlxTensor): MlxTensor {
    return this.out(this.n.mlx_divide(this.base, this.h(a), this.h(b), this.stream), "div");
  }
  maximum(a: MlxTensor, b: MlxTensor): MlxTensor {
    return this.out(this.n.mlx_maximum(this.base, this.h(a), this.h(b), this.stream), "maximum");
  }
  where(cond: MlxTensor, a: MlxTensor, b: MlxTensor): MlxTensor {
    return this.out(this.n.mlx_where(this.base, this.h(cond), this.h(a), this.h(b), this.stream), "where");
  }
  scale(x: MlxTensor, s: number): MlxTensor {
    const hx = this.h(x);
    return this.withScalar(s, floatOr(x.dtype), (hs) => this.out(this.n.mlx_multiply(this.base, hx, hs, this.stream), "scale"));
  }
  exp(x: MlxTensor): MlxTensor {
    return this.out(this.n.mlx_exp(this.base, this.h(x), this.stream), "exp");
  }
  log(x: MlxTensor): MlxTensor {
    return this.out(this.n.mlx_log(this.base, this.h(x), this.stream), "log");
  }
  relu(x: MlxTensor): MlxTensor {
    const hx = this.h(x);
    return this.withScalar(0, x.dtype, (hs) => this.out(this.n.mlx_maximum(this.base, hx, hs, this.stream), "relu"));
  }
  gelu(x: MlxTensor): MlxTensor {
    if (this.compiledGelu === undefined) this.compiledGelu = this.makeCompiledGelu();
    if (this.compiledGelu && this.tracing === 0) return this.compiledGelu(x);
    return this.geluEager(x);
  }

  /** x · (1 + erf(x / √2)) / 2 — the same graph as `mlx.nn.gelu`. */
  private geluEager(x: MlxTensor): MlxTensor {
    return this.scope(() => {
      const dt = floatOr(x.dtype);
      const hx = this.h(x);
      const sq = this.scalar(Math.SQRT2, dt), one = this.scalar(1, dt), two = this.scalar(2, dt);
      const t = this.out(this.n.mlx_divide(this.base, hx, sq.h, this.stream), "gelu");
      const e = this.out(this.n.mlx_erf(this.base, t.h, this.stream), "gelu");
      const p = this.out(this.n.mlx_add(this.base, one.h, e.h, this.stream), "gelu");
      const m = this.out(this.n.mlx_multiply(this.base, hx, p.h, this.stream), "gelu");
      return this.out(this.n.mlx_divide(this.base, m.h, two.h, this.stream), "gelu");
    });
  }

  private makeCompiledGelu(): ((x: MlxTensor) => MlxTensor) | null {
    // Like mlx.nn.gelu: @mx.compile(shapeless=True). GPU only — MLX's CPU
    // compile path JIT-builds C++ with the host toolchain.
    if (this.device !== "gpu") return null;
    return this.compileImpl((x: MlxTensor) => this.geluEager(x), true);
  }

  // ---- reductions ------------------------------------------------------------

  sum(x: MlxTensor, axis: number, keepDims = false): MlxTensor {
    return this.out(this.n.mlx_sum_axis(this.base, this.h(x), axis, keepDims, this.stream), "sum");
  }
  max(x: MlxTensor, axis: number, keepDims = false): MlxTensor {
    return this.out(this.n.mlx_max_axis(this.base, this.h(x), axis, keepDims, this.stream), "max");
  }
  softmax(x: MlxTensor, axis: number): MlxTensor {
    // precise=true: f32 accumulation for f16/bf16 inputs; result keeps x's dtype.
    return this.out(this.n.mlx_softmax_axis(this.base, this.h(x), axis, true, this.stream), "softmax");
  }
  sort(x: MlxTensor, axis: number): MlxTensor {
    return this.out(this.n.mlx_sort_axis(this.base, this.h(x), axis, this.stream), "sort");
  }

  // ---- linear algebra & NN ------------------------------------------------------

  matmul(a: MlxTensor, b: MlxTensor): MlxTensor {
    return this.out(this.n.mlx_matmul(this.base, this.h(a), this.h(b), this.stream), "matmul");
  }

  linear(x: MlxTensor, w: MlxTensor, b?: MlxTensor | null): MlxTensor {
    const hx = this.h(x);
    // wᵀ is a strided view (no copy); MLX's GEMM consumes it directly. Same
    // graph as mlx.nn.Linear: addmm(b, x, w.T) / x @ w.T.
    const wt = this.take(this.n.mlx_transpose_axes(this.base, this.h(w), this.ints(INTS_A, [1, 0]), 2, this.stream), "linear");
    try {
      const rc = b ? this.n.mlx_addmm(this.base, this.h(b), hx, wt, 1, 1, this.stream) : this.n.mlx_matmul(this.base, hx, wt, this.stream);
      return this.out(rc, "linear");
    } finally {
      this.n.mlx_array_free(wt);
    }
  }

  layerNorm(x: MlxTensor, weight: MlxTensor | null, bias: MlxTensor | null, eps: number): MlxTensor {
    return this.out(
      this.n.mlx_fast_layer_norm(this.base, this.h(x), weight ? this.h(weight) : 0, bias ? this.h(bias) : 0, eps, this.stream),
      "layerNorm",
    );
  }

  embedding(table: MlxTensor, ids: MlxTensor): MlxTensor {
    return this.out(this.n.mlx_take_axis(this.base, this.h(table), this.h(ids), 0, this.stream), "embedding");
  }

  gatherRows(x: MlxTensor, idx: MlxTensor): MlxTensor {
    const e = this.take(this.n.mlx_expand_dims(this.base, this.h(idx), 2, this.stream), "gatherRows");
    try {
      return this.out(this.n.mlx_take_along_axis(this.base, this.h(x), e, 1, this.stream), "gatherRows");
    } finally {
      this.n.mlx_array_free(e);
    }
  }

  rope(x: MlxTensor, base: number): MlxTensor {
    const dims = x.shape[x.shape.length - 1]!;
    return this.out(this.n.mlx_fast_rope(this.base, this.h(x), dims, false, optFloat(base), 1, 0, 0, this.stream), "rope");
  }

  sdpa(q: MlxTensor, k: MlxTensor, v: MlxTensor, mask: MlxTensor | null, scale: number): MlxTensor {
    const mode = mask ? this.sdpaMaskArray : this.sdpaMaskNone;
    const hm = mask ? this.h(mask) : 0;
    const rc = this.n.sdpaForceFused
      ? this.n.mlx_fast_scaled_dot_product_attention(this.base, this.h(q), this.h(k), this.h(v), scale, mode, hm, 0, false, this.stream)
      : this.n.mlx_fast_scaled_dot_product_attention(this.base, this.h(q), this.h(k), this.h(v), scale, mode, hm, 0, this.stream);
    return this.out(rc, "sdpa");
  }

  geglu(x: MlxTensor): MlxTensor {
    return this.scope(() => {
      const [value, gate] = this.split(x, 2, x.shape.length - 1);
      return this.mul(this.gelu(value!), gate!);
    });
  }

  meanPool(x: MlxTensor, mask: MlxTensor): MlxTensor {
    return this.scope(() => {
      const m = this.cast(this.reshape(mask, [mask.shape[0]!, mask.shape[1]!, 1]), "f32");
      const total = this.sum(this.mul(this.cast(x, "f32"), m), 1);
      const count = this.withScalar(1, "f32", (one) => this.out(this.n.mlx_maximum(this.base, this.h(this.sum(m, 1)), one, this.stream), "meanPool"));
      return this.div(total, count);
    });
  }

  // ---- general numerics ------------------------------------------------------

  private bin(fn: (res: number, a: number, b: number, s: number) => number, a: MlxTensor, b: MlxTensor, op: string, dtype?: DType): MlxTensor {
    return this.out(fn(this.base, this.h(a), this.h(b), this.stream), op, undefined, dtype);
  }
  private un(fn: (res: number, a: number, s: number) => number, x: MlxTensor, op: string, dtype?: DType): MlxTensor {
    return this.out(fn(this.base, this.h(x), this.stream), op, undefined, dtype);
  }
  /** Float-valued unary op; integer/bool input is computed in f32 (MLX would keep some ints). */
  private unF(fn: (res: number, a: number, s: number) => number, x: MlxTensor, op: string): MlxTensor {
    if (isFloat(x.dtype)) return this.un(fn, x, op, x.dtype);
    const f = this.take(this.n.mlx_astype(this.base, this.h(x), MLX_DTYPE.f32, this.stream), op);
    try {
      return this.out(fn(this.base, f, this.stream), op, undefined, "f32");
    } finally {
      this.n.mlx_array_free(f);
    }
  }
  private axisOp(fn: (res: number, a: number, axis: number, keep: boolean, s: number) => number, x: MlxTensor, axis: number, keepDims: boolean, op: string): number {
    return this.take(fn(this.base, this.h(x), axis, keepDims, this.stream), op);
  }

  equal(a: MlxTensor, b: MlxTensor): MlxTensor { return this.bin(this.n.mlx_equal, a, b, "equal", "bool"); }
  notEqual(a: MlxTensor, b: MlxTensor): MlxTensor { return this.bin(this.n.mlx_not_equal, a, b, "notEqual", "bool"); }
  less(a: MlxTensor, b: MlxTensor): MlxTensor { return this.bin(this.n.mlx_less, a, b, "less", "bool"); }
  lessEqual(a: MlxTensor, b: MlxTensor): MlxTensor { return this.bin(this.n.mlx_less_equal, a, b, "lessEqual", "bool"); }
  greater(a: MlxTensor, b: MlxTensor): MlxTensor { return this.bin(this.n.mlx_greater, a, b, "greater", "bool"); }
  greaterEqual(a: MlxTensor, b: MlxTensor): MlxTensor { return this.bin(this.n.mlx_greater_equal, a, b, "greaterEqual", "bool"); }
  logicalAnd(a: MlxTensor, b: MlxTensor): MlxTensor { return this.bin(this.n.mlx_logical_and, a, b, "logicalAnd", "bool"); }
  logicalOr(a: MlxTensor, b: MlxTensor): MlxTensor { return this.bin(this.n.mlx_logical_or, a, b, "logicalOr", "bool"); }
  logicalNot(x: MlxTensor): MlxTensor { return this.un(this.n.mlx_logical_not, x, "logicalNot", "bool"); }
  sqrt(x: MlxTensor): MlxTensor { return this.unF(this.n.mlx_sqrt, x, "sqrt"); }
  rsqrt(x: MlxTensor): MlxTensor { return this.unF(this.n.mlx_rsqrt, x, "rsqrt"); }
  tanh(x: MlxTensor): MlxTensor { return this.unF(this.n.mlx_tanh, x, "tanh"); }
  sigmoid(x: MlxTensor): MlxTensor { return this.unF(this.n.mlx_sigmoid, x, "sigmoid"); }
  erf(x: MlxTensor): MlxTensor { return this.unF(this.n.mlx_erf, x, "erf"); }
  neg(x: MlxTensor): MlxTensor { return this.un(this.n.mlx_negative, x, "neg"); }
  abs(x: MlxTensor): MlxTensor { return this.un(this.n.mlx_abs, x, "abs"); }
  pow(a: MlxTensor, b: MlxTensor): MlxTensor {
    if (isFloat(a.dtype) || isFloat(b.dtype)) return this.bin(this.n.mlx_power, a, b, "pow");
    return this.scope(() => this.bin(this.n.mlx_power, this.cast(a, "f32"), b, "pow", "f32"));
  }
  argmax(x: MlxTensor, axis: number, keepDims = false): MlxTensor { return this.argIdx(this.n.mlx_argmax_axis, x, axis, keepDims, "argmax"); }
  argmin(x: MlxTensor, axis: number, keepDims = false): MlxTensor { return this.argIdx(this.n.mlx_argmin_axis, x, axis, keepDims, "argmin"); }
  /** MLX arg-reductions return uint32; the contract says i32. */
  private argIdx(fn: (res: number, a: number, axis: number, keep: boolean, s: number) => number, x: MlxTensor, axis: number, keepDims: boolean, op: string): MlxTensor {
    const u = this.axisOp(fn, x, axis, keepDims, op);
    try {
      return this.out(this.n.mlx_astype(this.base, u, MLX_DTYPE.i32, this.stream), op, undefined, "i32");
    } finally {
      this.n.mlx_array_free(u);
    }
  }
  mean(x: MlxTensor, axis: number, keepDims = false): MlxTensor { return this.wrap(this.axisOp(this.n.mlx_mean_axis, x, axis, keepDims, "mean")); }
  min(x: MlxTensor, axis: number, keepDims = false): MlxTensor { return this.wrap(this.axisOp(this.n.mlx_min_axis, x, axis, keepDims, "min")); }
  cumsum(x: MlxTensor, axis: number): MlxTensor {
    if (x.dtype === "bool") return this.scope(() => this.cumsum(this.cast(x, "i32"), axis));
    return this.out(this.n.cumsum(this.base, this.h(x), axis, false, true, this.stream), "cumsum");
  }

  // ---- quantized weights ---------------------------------------------------

  /**
   * Repacks (never dequantizes) a laya-js quantized matrix into MLX's affine
   * layout: the packed bytes read as little-endian u32 words already are
   * MLX's packing, so affine data uploads as-is (uint32 [out, in·bits/32]).
   * Symmetric q → unsigned q + 2^(bits−1) (a per-byte XOR of 0x80 / 0x88)
   * with bias = −2^(bits−1)·scale, exact in f16/f32. Scales and biases are
   * stored in `dtype`. Resolves to null for what MLX lacks: group sizes other
   * than 32/64/128, or a partial last group.
   */
  async fromHostQuantized(h: HostQuantized, dtype: DType): Promise<QuantizedTensor<MlxTensor> | null> {
    const [N, K] = h.shape;
    if (!MLX_GROUP_SIZES.has(h.groupSize) || K % h.groupSize || (K * h.bits) % 32) return null;
    if (!isFloat(dtype)) throw new TypeError(`backend-mlx fromHostQuantized: dtype must be float, got ${dtype}`);
    const G = K / h.groupSize;
    let data = h.data;
    let biases: HostTensor | null = h.biases;
    if (h.mode === "symmetric") {
      const flip = h.bits === 8 ? 0x80 : 0x88;
      const d = new Uint8Array(data.length);
      for (let i = 0; i < d.length; i++) d[i] = data[i]! ^ flip;
      data = d;
      const s = toF32(h.scales), off = -(1 << (h.bits - 1));
      const b = new Float32Array(s.length);
      for (let i = 0; i < s.length; i++) b[i] = off * s[i]!;
      biases = { dtype: "f32", shape: [N, G], data: b };
    }
    const shape = [N, (K * h.bits) / 32];
    const wh = this.n.mlx_array_new_data(data, this.ints(INTS_A, shape), 2, MLX_UINT32);
    if (!wh) throw new Error(`backend-mlx fromHostQuantized: ${this.n.takeError()}`);
    const w = this.wrap(wh, shape, "i32"); // u32 words; "i32" is only the label (not read by other ops)
    const up = (t: HostTensor) => this.scope(() => {
      const u = this.upload(t);
      return u.dtype === dtype ? u : this.cast(u, dtype);
    });
    const { bits, groupSize, mode } = h;
    return { shape: [N, K], bits, groupSize, mode, dtype, native: true, w, scales: up(h.scales), biases: up(biases!) };
  }

  quantizedLinear(x: MlxTensor, w: MlxTensor, scales: MlxTensor, biases: MlxTensor | null, opts: QuantizedLinearOptions, bias?: MlxTensor | null): MlxTensor {
    if (!biases) throw new Error("backend-mlx quantizedLinear: weights must come from this backend's fromHostQuantized");
    return this.scope(() => {
      let y: MlxTensor = this.out(
        this.n.mlx_quantized_matmul(this.base, this.h(x), this.h(w), this.h(scales), this.h(biases), true, optInt(opts.groupSize), optInt(opts.bits), this.affine, this.stream),
        "quantizedLinear",
      );
      if (bias) y = this.add(y, bias);
      return y.dtype === x.dtype || !isFloat(x.dtype) ? y : this.cast(y, x.dtype);
    });
  }

  quantizedEmbedding(w: MlxTensor, scales: MlxTensor, biases: MlxTensor | null, opts: QuantizedLinearOptions, ids: MlxTensor): MlxTensor {
    if (!biases) throw new Error("backend-mlx quantizedEmbedding: weights must come from this backend's fromHostQuantized");
    return this.scope(() => {
      const n = ids.shape.reduce((a, d) => a * d, 1);
      const flat = this.reshape(ids, [n]);
      const rows = (t: MlxTensor) => this.embedding(t, flat);
      const dt = scales.dtype;
      const d = this.out(
        this.n.mlx_dequantize(this.base, this.h(rows(w)), this.h(rows(scales)), this.h(rows(biases)), optInt(opts.groupSize), optInt(opts.bits), this.affine, 0, optDtype(MLX_DTYPE[dt]), this.stream),
        "quantizedEmbedding", undefined, dt,
      );
      return this.reshape(d, [...ids.shape, d.shape[1]!]);
    });
  }

  // ---- compile ---------------------------------------------------------------

  compile<A extends MlxTensor[], R>(fn: (...args: A) => R): (...args: A) => R {
    if (this.device !== "gpu") return fn; // MLX's CPU compile path needs a working host C++ JIT
    return this.compileImpl(fn, false);
  }

  /**
   * Wraps `fn` in an mlx_closure whose C function is a trampoline back into
   * JS, compiles it with mlx_compile, and returns a function applying the
   * compiled closure. `fn` is traced once per distinct input signature (or
   * once, when `shapeless`), synchronously inside mlx_closure_apply. Outputs
   * may be a tensor, an array of tensors, or an object of tensors.
   * Constraint: `fn` must be pure in its tensor arguments (tensors captured
   * from outside become compile-time constants, as in Python MLX).
   */
  private compileImpl<A extends MlxTensor[], R>(fn: (...args: A) => R, shapeless: boolean): (...args: A) => R {
    const n = this.n;
    if (!this.closureTramp) {
      this.closureTramp = n.closureTrampoline((res, input, payload) => {
        const f = this.closures.get(payload);
        return f ? f(res, input) : 1;
      });
    }
    const id = this.nextClosureId++;
    let rebuild: ((outs: MlxTensor[]) => unknown) | null = null;
    this.closures.set(id, (res, input) => {
      // Runs inside a C frame: nothing may throw out of here.
      this.tracing++;
      const ins: MlxArray[] = [];
      try {
        const count = n.mlx_vector_array_size(input);
        for (let i = 0; i < count; i++) ins.push(this.out(n.mlx_vector_array_get(this.base, input, i), "trace input"));
        this.scope(() => {
          const flat = flatten(fn(...(ins as unknown as A)));
          rebuild = flat.rebuild;
          const buf = handleBuffer(flat.flat.map((t) => this.h(t)));
          if (n.mlx_vector_array_set_data(res, n.addressOf(buf), flat.flat.length) !== 0) throw new Error(n.takeError() ?? "vector set failed");
          return null;
        });
        return 0;
      } catch (e) {
        pendingTraceError = e;
        return 1;
      } finally {
        for (const t of ins) this.dispose(t);
        this.tracing--;
      }
    });
    const cls = n.mlx_closure_new_func_payload(this.closureTramp, id, 0);
    if (!cls) throw new Error(`backend-mlx compile: ${n.takeError()}`);
    let compiled: number;
    try {
      compiled = this.take(n.mlx_compile(this.base, cls, shapeless), "compile");
    } finally {
      n.mlx_closure_free(cls); // the compiled closure holds its own reference
    }
    const closures = this.closures;
    const wrapper = (...args: A): R => {
      const vin = this.vec(args);
      let rc: number;
      try {
        rc = n.mlx_closure_apply(this.base + APPLY_SLOT, compiled, vin);
      } finally {
        n.mlx_vector_array_free(vin);
      }
      const u = this.u32, i = APPLY_SLOT >> 2;
      const vout = u[i]! + u[i + 1]! * TWO32;
      u[i] = 0;
      u[i + 1] = 0;
      if (rc !== 0) {
        if (vout) n.mlx_vector_array_free(vout);
        const e = pendingTraceError;
        pendingTraceError = null;
        const msg = n.takeError();
        throw e instanceof Error ? e : new Error(`backend-mlx compiled call: ${msg ?? String(e)}`);
      }
      return rebuild!(this.unvec(vout, "compiled call", true)) as R;
    };
    compiledRegistry.register(wrapper, { n, compiled, closures, id });
    return wrapper;
  }
}

const compiledRegistry = new FinalizationRegistry<{ n: Native; compiled: number; closures: Map<number, unknown>; id: number }>((c) => {
  c.n.mlx_closure_free(c.compiled);
  c.closures.delete(c.id);
});

function handleBuffer(hs: number[]): Uint32Array {
  const buf = new Uint32Array(new ArrayBuffer(Math.max(8, hs.length * 8)));
  for (let i = 0; i < hs.length; i++) {
    buf[2 * i] = hs[i]! % TWO32;
    buf[2 * i + 1] = Math.floor(hs[i]! / TWO32);
  }
  return buf;
}

function isFloat(d: DType): boolean {
  return d === "f32" || d === "f16" || d === "bf16";
}

function floatOr(d: DType): DType {
  return d === "i32" || d === "bool" ? "f32" : d;
}

const f32b = new Float32Array(1);
const u32b = new Uint32Array(f32b.buffer);
/** mlx_optional_float {float value; bool has_value} as one 8-byte register. */
function optFloat(v: number): bigint {
  f32b[0] = v;
  return BigInt(u32b[0]!) | (1n << 32n);
}

/** mlx_optional_int {int value; bool has_value} as one 8-byte register. */
function optInt(v: number): bigint {
  return BigInt(v >>> 0) | (1n << 32n);
}
/** mlx_optional_dtype {mlx_dtype value; bool has_value}. */
function optDtype(d: number): bigint {
  return BigInt(d) | (1n << 32n);
}

function cstr(s: string): Uint8Array {
  const b = new TextEncoder().encode(s);
  const out = new Uint8Array(new ArrayBuffer(b.length + 8));
  out.set(b);
  return out;
}

function collect(v: unknown, into: Set<MlxTensor>): void {
  if (v instanceof MlxArray) into.add(v);
  else if (Array.isArray(v)) {
    for (const x of v) if (x instanceof MlxArray) into.add(x);
  } else if (v && typeof v === "object") {
    for (const x of Object.values(v)) if (x instanceof MlxArray) into.add(x);
  }
}

function flatten(r: unknown): { flat: MlxTensor[]; rebuild: (outs: MlxTensor[]) => unknown } {
  if (r instanceof MlxArray) return { flat: [r], rebuild: (o) => o[0] };
  if (Array.isArray(r) && r.every((x) => x instanceof MlxArray)) return { flat: r, rebuild: (o) => o };
  if (r && typeof r === "object" && !Array.isArray(r)) {
    const keys = Object.keys(r);
    const vals = Object.values(r);
    if (vals.every((x) => x instanceof MlxArray)) {
      return { flat: vals as MlxTensor[], rebuild: (o) => Object.fromEntries(keys.map((k, i) => [k, o[i]])) };
    }
  }
  throw new TypeError("backend-mlx compile: fn must return a tensor, an array of tensors, or an object of tensors");
}

/** Creates an MLX backend. Throws on non-macOS/arm64 or when libmlxc cannot be found. */
export function createMlxBackend(opts: MlxBackendOptions = {}): MlxBackend {
  return new MlxBackendImpl(opts);
}
