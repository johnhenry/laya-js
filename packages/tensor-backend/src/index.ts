/**
 * @johnhenry/tensor-backend — the op contract every inference backend
 * (CPU reference, native MLX, WebGPU) implements.
 *
 * Design rules:
 * - Backends are passed explicitly; there is no global default.
 * - Ops are synchronous and return opaque handles. Backends may evaluate
 *   lazily (MLX graphs, queued WebGPU passes). Device transfers are async
 *   in both directions: `fromHost` and `read` return Promises, so a
 *   transfer is never silently synchronous (math-plus RFC 0001 §12 Q2).
 * - Masks and index tensors are usually built on the host and uploaded;
 *   the optional "general numerics" ops (comparisons, `argmax`, `cumsum`,
 *   …) exist for device-side code, not for the transformer hot path.
 * - Fused ops (`linear`, `layerNorm`, `rope`, `sdpa`, `gelu`) are
 *   required so each backend can use its fastest kernel. Every optional op
 *   is called through its `./compose.ts` helper (`geglu(b, x)`,
 *   `sqrt(b, x)`, …), which uses the native kernel when the backend has
 *   one and a default composition otherwise (`cumsum` has none; see
 *   `NUMERICS_OPS`).
 * - dtype names match @johnhenry/math-plus-tensor-core ("f32", "f16", ...),
 *   so a HostTensor feeds `Tensor.fromTypedArray` without a copy.
 */

export type DType = "f32" | "f16" | "bf16" | "i32" | "bool";

export type Shape = readonly number[];

/**
 * Host-side storage per dtype:
 * - f32 → Float32Array, f16 → Float16Array, bf16 → Uint16Array (raw bits),
 *   i32 → Int32Array, bool → Uint8Array (0/1).
 */
export type HostData = Float32Array | Float16Array | Uint16Array | Int32Array | Uint8Array;

export interface HostTensor {
  readonly dtype: DType;
  readonly shape: Shape;
  readonly data: HostData;
}

/** Opaque backend tensor handle. */
export interface Tensor {
  readonly shape: Shape;
  readonly dtype: DType;
}

export interface Backend<T extends Tensor = Tensor> {
  /** e.g. "cpu", "mlx", "webgpu". */
  readonly name: string;
  /** Whether this backend can store and compute in `dtype`. */
  supports(dtype: DType): boolean;

  // ---- transfer / lifetime ------------------------------------------------
  /**
   * Uploads a host tensor. Accepts every DType. A backend may widen to a
   * supported storage dtype (e.g. f16 → f32 on the CPU reference); the
   * resulting `.dtype` reflects storage. Async like `read`: do not mutate
   * `t.data` until the Promise settles. Batch independent uploads and
   * await them together (`Promise.all`) rather than one at a time.
   */
  fromHost(t: HostTensor): Promise<T>;
  /** Materializes and copies to host. */
  read(t: T): Promise<HostTensor>;
  /** Frees a tensor now. Idempotent. */
  dispose(t: T): void;
  /**
   * Runs `fn`; every tensor created inside it and not returned (directly,
   * or inside a returned array/object one level deep) is disposed afterward.
   */
  scope<R>(fn: () => R): R;
  /** Forces pending lazy work to execute (MLX eval / GPU submit). Optional. */
  flush?(...ts: T[]): void;
  /** Releases all backend resources (device, pools). Optional. */
  destroy?(): void;

  // ---- shape --------------------------------------------------------------
  reshape(x: T, shape: Shape): T;
  /** Permute axes, like numpy.transpose(x, perm). */
  transpose(x: T, perm: readonly number[]): T;
  /** Half-open slice per axis; omitted trailing axes are taken whole. */
  slice(x: T, begin: readonly number[], end: readonly number[]): T;
  /** Split into `parts` equal chunks along `axis`. */
  split(x: T, parts: number, axis: number): T[];
  concat(xs: readonly T[], axis: number): T;
  cast(x: T, dtype: DType): T;

  // ---- elementwise (numpy broadcasting) ------------------------------------
  add(a: T, b: T): T;
  sub(a: T, b: T): T;
  mul(a: T, b: T): T;
  div(a: T, b: T): T;
  maximum(a: T, b: T): T;
  /** cond is bool; a and b broadcast against cond. */
  where(cond: T, a: T, b: T): T;
  /** Multiply by a scalar constant. */
  scale(x: T, s: number): T;
  exp(x: T): T;
  log(x: T): T;
  relu(x: T): T;
  /** Exact erf GELU: 0.5·x·(1 + erf(x/√2)). Not the tanh approximation. */
  gelu(x: T): T;

  // ---- reductions ---------------------------------------------------------
  sum(x: T, axis: number, keepDims?: boolean): T;
  max(x: T, axis: number, keepDims?: boolean): T;
  /** Numerically stable softmax. Computed in f32 internally. */
  softmax(x: T, axis: number): T;
  /** Ascending sort along `axis`. */
  sort(x: T, axis: number): T;

  // ---- linear algebra & NN ------------------------------------------------
  /** Batched matmul with broadcasting of leading dims: [..., m, k] @ [..., k, n]. */
  matmul(a: T, b: T): T;
  /**
   * y = x · wᵀ (+ b). `w` is PyTorch layout [out, in]. Accumulates in f32
   * regardless of storage dtype.
   */
  linear(x: T, w: T, b?: T | null): T;
  /** LayerNorm over the last axis; weight/bias may be null. Stats in f32. */
  layerNorm(x: T, weight: T | null, bias: T | null, eps: number): T;
  /** Row gather: table [V, D], ids i32 [...]. Returns [..., D]. */
  embedding(table: T, ids: T): T;
  /**
   * Per-batch row gather: x [B, L, D], idx i32 [B, M] → [B, M, D].
   * (numpy: x[arange(B)[:, None], idx])
   */
  gatherRows(x: T, idx: T): T;
  /**
   * Rotary embedding on x [B, H, L, Dh], positions 0..L-1, split-half
   * (NeoX / HF, MLX `traditional=False`) rotation, full head dim, scale 1.
   */
  rope(x: T, base: number): T;
  /**
   * Scaled dot-product attention. q/k/v [B, H, L, Dh]; mask must be *bool*
   * (additive float masks are not part of the contract), broadcastable to
   * [B, H, Lq, Lk] (true = attend). Softmax in f32.
   * Rows that are entirely masked are undefined behaviour; callers
   * must avoid them.
   */
  sdpa(q: T, k: T, v: T, mask: T | null, scale: number): T;

  // ---- optional fused fast paths (see compose.ts for defaults) ------------
  /** Splits the last axis into [value, gate] halves, returns gelu(value)·gate. */
  geglu?(x: T): T;
  /** Masked mean over axis 1: x [B, L, D], mask bool [B, L] → [B, D] f32. */
  meanPool?(x: T, mask: T): T;
  /** Wrap a pure function of tensors for graph compilation (MLX). */
  compile?<A extends T[], R>(fn: (...args: A) => R): (...args: A) => R;

  // ---- optional general numerics (call through compose.ts) ----------------
  // Elementwise ops broadcast like numpy. Comparisons and logical ops return
  // bool; integer inputs of float-valued ops compute in f32. Results for
  // NaN inputs are backend-defined (WebGPU may assume no NaNs).
  /** a == b → bool. */
  equal?(a: T, b: T): T;
  /** a != b → bool. */
  notEqual?(a: T, b: T): T;
  /** a < b → bool. */
  less?(a: T, b: T): T;
  /** a <= b → bool. */
  lessEqual?(a: T, b: T): T;
  /** a > b → bool. */
  greater?(a: T, b: T): T;
  /** a >= b → bool. */
  greaterEqual?(a: T, b: T): T;
  /** Nonzero-is-true AND → bool. */
  logicalAnd?(a: T, b: T): T;
  /** Nonzero-is-true OR → bool. */
  logicalOr?(a: T, b: T): T;
  /** Nonzero-is-true NOT → bool. */
  logicalNot?(x: T): T;
  sqrt?(x: T): T;
  /** 1 / √x. */
  rsqrt?(x: T): T;
  /** aᵇ (float result). A negative base is defined for integral exponents (MLX/C `pow`). */
  pow?(a: T, b: T): T;
  /** −x; keeps f32/f16/bf16/i32. */
  neg?(x: T): T;
  /** |x|; keeps f32/f16/bf16/i32. */
  abs?(x: T): T;
  tanh?(x: T): T;
  /** 1 / (1 + e⁻ˣ). */
  sigmoid?(x: T): T;
  /** Error function, accurate to f32 (≈1e-7 absolute). */
  erf?(x: T): T;
  /** Index (i32) of the first maximum along `axis`. */
  argmax?(x: T, axis: number, keepDims?: boolean): T;
  /** Index (i32) of the first minimum along `axis`. */
  argmin?(x: T, axis: number, keepDims?: boolean): T;
  /** Mean along `axis` (f32 accumulation; integer input → f32). */
  mean?(x: T, axis: number, keepDims?: boolean): T;
  /** Minimum along `axis`; keeps dtype. */
  min?(x: T, axis: number, keepDims?: boolean): T;
  /** Inclusive prefix sum along `axis`; floats keep dtype, i32/bool → i32. */
  cumsum?(x: T, axis: number): T;
}

export * from "./host.ts";
export * from "./compose.ts";
