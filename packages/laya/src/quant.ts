/**
 * laya-js quantized checkpoints (format version 1): large Linear weights and
 * the token embedding stored as 8-bit or 4-bit integers inside an ordinary
 * safetensors file, dequantized on the host when the checkpoint is loaded.
 * Browser-safe (no node: imports).
 *
 * A quantized matrix `W` [rows, cols] (row-major, cols = the input dim) is
 * split along each row into groups of `g` consecutive values (`g` = cols
 * for "per-row"); every group has its own float16 parameters:
 *
 * - q8 (symmetric): `W` → I8 [rows, cols], `W.scales` → F16 [rows, cols/g];
 *   w ≈ q · scale, q ∈ [-127, 127], scale = max|w| / 127.
 * - q4 (affine): `W` → U8 [rows, cols/2] (two values per byte: column 2k in
 *   the low nibble, 2k+1 in the high nibble), `W.scales` and `W.biases` →
 *   F16 [rows, cols/g]; w ≈ q · scale + bias, q ∈ [0, 15],
 *   scale = (max − min) / 15, bias = min (MLX-style affine quantization).
 *
 * Each quantized tensor's width follows its dtype (I8 = q8, U8 = q4), so a
 * "q4" checkpoint may keep sensitive matrices at q8 (mixed precision).
 * Everything else (norms, biases, small tensors) keeps its original dtype.
 * The file's `__metadata__` carries `laya_quant` ("q8" | "q4"),
 * `group_size` (a number or "row"), `version` ("1") and
 * `laya_quant_source_dtype`. A loader that finds `laya_quant` rebuilds each
 * quantized tensor from `W` + `W.scales` (+ `W.biases`); any other
 * safetensors reader can still parse the file.
 */
import type { LazySafetensors, SafeDType, TensorInput } from "@johnhenry/math-plus-safetensors";
import { toFloat32, writeSafetensors } from "@johnhenry/math-plus-safetensors";

export const QUANT_FORMAT_VERSION = "1";
export type QuantBits = 4 | 8;
export type QuantScheme = "q8" | "q4";
/** A value dtype the dequantizer can produce. */
export type DequantDtype = "f16" | "f32";

/** Parsed `__metadata__` of a quantized checkpoint. */
export interface QuantMetadata {
  scheme: QuantScheme;
  bits: QuantBits;
  /** Values per group along the input dim; `null` = one group per row. */
  groupSize: number | null;
  version: string;
  sourceDtype: string | undefined;
}

/** One quantized matrix, as stored. */
export interface QuantizedMatrix {
  bits: QuantBits;
  rows: number;
  cols: number;
  /** q8: Int8Array [rows*cols]; q4: Uint8Array [rows*cols/2] (low nibble = even column). */
  data: Int8Array | Uint8Array;
  /** [rows, cols/groupSize] */
  scales: Float16Array;
  /** q4 only: [rows, cols/groupSize] */
  biases?: Float16Array;
  groupSize: number;
}

/**
 * Returns the checkpoint's quantization scheme from its safetensors
 * `__metadata__`, or null for an ordinary (float) checkpoint. Throws on a
 * newer format version or an unknown scheme rather than misreading it.
 */
export function quantMetadata(metadata: Readonly<Record<string, string>> | undefined): QuantMetadata | null {
  const scheme = metadata?.laya_quant;
  if (scheme === undefined) return null;
  if (scheme !== "q8" && scheme !== "q4") throw new Error(`laya: unknown quantization scheme laya_quant=${JSON.stringify(scheme)} (expected "q8" or "q4")`);
  const version = metadata!.version ?? "";
  if (version !== QUANT_FORMAT_VERSION) {
    throw new Error(`laya: quantized checkpoint format version ${JSON.stringify(version)} is not supported (this laya reads version ${QUANT_FORMAT_VERSION}); upgrade @johnhenry/laya`);
  }
  const g = metadata!.group_size;
  const groupSize = g === undefined || g === "row" ? null : Number(g);
  if (groupSize !== null && !(Number.isInteger(groupSize) && groupSize > 0)) throw new Error(`laya: invalid group_size ${JSON.stringify(g)} in quantized checkpoint`);
  return { scheme, bits: scheme === "q8" ? 8 : 4, groupSize, version, sourceDtype: metadata!.laya_quant_source_dtype };
}

const f16 = new Float16Array(1);
/** Rounds to the nearest float16 value (what storing the parameter will do). */
const toF16 = (x: number): number => ((f16[0] = x), f16[0]!);

/**
 * Quantizes a row-major [rows, cols] matrix. `groupSize` must divide `cols`
 * (omit it, or pass `cols`, for one group per row); q4 also needs an even
 * `cols` and group size.
 *
 * Each group starts from the range fit (q8: scale = max|w|/127; q4:
 * bias = min, scale = (max − min)/15) and, unless `refine` is false, then
 * alternates "assign q by rounding" with a least-squares refit of
 * scale (and bias) for `REFINE_STEPS` rounds, keeping whichever parameters
 * give the smallest squared error. Scale and bias are always rounded to
 * float16 *before* the q are chosen, so the stored parameters are exactly
 * the ones the stored q were fitted to.
 */
export function quantizeMatrix(
  values: ArrayLike<number>,
  rows: number,
  cols: number,
  bits: QuantBits,
  groupSize: number = cols,
  { refine = true }: { refine?: boolean } = {},
): QuantizedMatrix {
  if (values.length !== rows * cols) throw new RangeError(`quantizeMatrix: ${values.length} values for [${rows}, ${cols}]`);
  if (bits !== 8 && bits !== 4) throw new RangeError(`quantizeMatrix: bits must be 8 or 4, got ${String(bits)}`);
  if (!(Number.isInteger(groupSize) && groupSize > 0 && cols % groupSize === 0)) throw new RangeError(`quantizeMatrix: group size ${groupSize} must divide ${cols}`);
  if (bits === 4 && (cols % 2 || groupSize % 2)) throw new RangeError("quantizeMatrix: q4 needs an even column count and group size");
  const G = cols / groupSize;
  const scales = new Float16Array(rows * G);
  const biases = bits === 4 ? new Float16Array(rows * G) : undefined;
  const q = new Int32Array(rows * cols);
  const fit = bits === 8 ? fitSymmetric : fitAffine;
  const steps = refine ? REFINE_STEPS : 0;
  const w = new Float64Array(groupSize), cur = new Int32Array(groupSize), best = new Int32Array(groupSize);
  for (let r = 0; r < rows; r++) {
    for (let gi = 0; gi < G; gi++) {
      const start = r * cols + gi * groupSize;
      for (let k = 0; k < groupSize; k++) w[k] = values[start + k]!;
      const [s, b] = fit(w, cur, best, steps);
      scales[r * G + gi] = s;
      if (biases) biases[r * G + gi] = b;
      q.set(best, start);
    }
  }
  if (bits === 8) return { bits, rows, cols, data: Int8Array.from(q), scales, groupSize };
  const data = new Uint8Array((rows * cols) / 2);
  for (let i = 0; i < rows * cols; i += 2) data[i >> 1] = q[i]! | (q[i + 1]! << 4);
  return { bits, rows, cols, data, scales, biases: biases!, groupSize };
}

/** Least-squares refinement rounds per group (each: assign q, refit scale/bias). */
export const REFINE_STEPS = 8;

/** Assigns q = clamp(round((w − b)/s)) into `q`; returns the squared error with (s, b). */
function assign(w: Float64Array, q: Int32Array, s: number, b: number, qmin: number, qmax: number): number {
  let err = 0;
  const ok = s > 0 && Number.isFinite(s);
  for (let k = 0; k < w.length; k++) {
    let v = ok ? Math.round((w[k]! - b) / s) : 0;
    v = v < qmin ? qmin : v > qmax ? qmax : v;
    q[k] = v;
    const d = w[k]! - (v * (ok ? s : 0) + b);
    err += d * d;
  }
  return err;
}

/** q8: w ≈ q·s, q ∈ [-127, 127]. Returns [s, 0]; the chosen q end up in `best`. */
function fitSymmetric(w: Float64Array, cur: Int32Array, best: Int32Array, steps: number): [number, number] {
  let amax = 0;
  for (let k = 0; k < w.length; k++) amax = Math.max(amax, Math.abs(w[k]!));
  let s = toF16(amax / 127);
  let bestErr = assign(w, best, s, 0, -127, 127), bestS = s;
  for (let it = 0; it < steps; it++) {
    // least squares for w ≈ s·q given q
    let num = 0, den = 0;
    for (let k = 0; k < w.length; k++) (num += best[k]! * w[k]!), (den += best[k]! * best[k]!);
    if (den === 0) break;
    s = toF16(num / den);
    const err = assign(w, cur, s, 0, -127, 127);
    if (!(err < bestErr)) break;
    (bestErr = err), (bestS = s), best.set(cur);
  }
  return [bestS, 0];
}

/** q4: w ≈ q·s + b, q ∈ [0, 15]. Returns [s, b]; the chosen q end up in `best`. */
function fitAffine(w: Float64Array, cur: Int32Array, best: Int32Array, steps: number): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (let k = 0; k < w.length; k++) (lo = Math.min(lo, w[k]!)), (hi = Math.max(hi, w[k]!));
  let b = toF16(lo), s = toF16((hi - b) / 15);
  let bestErr = assign(w, best, s, b, 0, 15), bestS = s, bestB = b;
  for (let it = 0; it < steps; it++) {
    // least squares for w ≈ s·q + b given q
    const n = w.length;
    let sq = 0, sw = 0, sqq = 0, sqw = 0;
    for (let k = 0; k < n; k++) (sq += best[k]!), (sw += w[k]!), (sqq += best[k]! * best[k]!), (sqw += best[k]! * w[k]!);
    const den = n * sqq - sq * sq;
    if (den === 0) break;
    s = toF16((n * sqw - sq * sw) / den);
    b = toF16((sw - s * sq) / n);
    const err = assign(w, cur, s, b, 0, 15);
    if (!(err < bestErr)) break;
    (bestErr = err), (bestS = s), (bestB = b), best.set(cur);
  }
  return [bestS, bestB];
}

/**
 * Rebuilds a quantized matrix as float16 (`Float16Array`) or float32
 * values, each one `fl(q · scale + bias)` rounded once from double. Uses a
 * per-group lookup table of the 2^bits possible outputs when the group is
 * large enough to amortize it (every q4 group; per-row q8), so the inner
 * loop only copies 16-bit patterns.
 */
export function dequantizeMatrix(m: QuantizedMatrix, dtype: "f16"): Float16Array;
export function dequantizeMatrix(m: QuantizedMatrix, dtype: "f32"): Float32Array;
export function dequantizeMatrix(m: QuantizedMatrix, dtype: DequantDtype): Float16Array | Float32Array;
export function dequantizeMatrix(m: QuantizedMatrix, dtype: DequantDtype): Float16Array | Float32Array {
  const { bits, rows, cols, data, scales, biases, groupSize } = m;
  const G = cols / groupSize;
  if (!Number.isInteger(G) || scales.length !== rows * G) throw new RangeError(`dequantizeMatrix: scales has ${scales.length} values, want ${rows} × ${G}`);
  const n = rows * cols;
  const out = dtype === "f16" ? new Float16Array(n) : new Float32Array(n);
  // bit patterns: Uint16 for f16, Uint32 for f32 (copies never touch float conversion)
  const bitsOut = dtype === "f16" ? new Uint16Array(out.buffer) : new Uint32Array(out.buffer);
  const levels = 1 << bits;
  const lut = dtype === "f16" ? new Float16Array(levels) : new Float32Array(levels);
  const lutBits = dtype === "f16" ? new Uint16Array(lut.buffer) : new Uint32Array(lut.buffer);
  if (bits === 8) {
    if (data.length !== n) throw new RangeError(`dequantizeMatrix: q8 data has ${data.length} values, want ${n}`);
    const q = data as Int8Array;
    const useLut = groupSize >= 256;
    for (let r = 0; r < rows; r++) {
      for (let gi = 0; gi < G; gi++) {
        const s = scales[r * G + gi]!;
        const start = r * cols + gi * groupSize, end = start + groupSize;
        if (useLut) {
          for (let k = 0; k < 256; k++) lut[k] = (k - 128) * s;
          for (let i = start; i < end; i++) bitsOut[i] = lutBits[q[i]! + 128]!;
        } else {
          for (let i = start; i < end; i++) out[i] = q[i]! * s;
        }
      }
    }
    return out;
  }
  if (!biases || biases.length !== rows * G) throw new RangeError("dequantizeMatrix: q4 needs biases [rows, cols/groupSize]");
  if (data.length * 2 !== n) throw new RangeError(`dequantizeMatrix: q4 data has ${data.length} bytes, want ${n / 2}`);
  for (let r = 0; r < rows; r++) {
    for (let gi = 0; gi < G; gi++) {
      const s = scales[r * G + gi]!, b = biases[r * G + gi]!;
      for (let k = 0; k < 16; k++) lut[k] = k * s + b;
      const start = r * cols + gi * groupSize, end = start + groupSize;
      for (let i = start; i < end; i += 2) {
        const byte = data[i >> 1]!;
        bitsOut[i] = lutBits[byte & 15]!;
        bitsOut[i + 1] = lutBits[byte >> 4]!;
      }
    }
  }
  return out;
}

// ------------------------------------------------------------------ converter
export interface QuantizeOptions {
  bits: QuantBits;
  /** Values per group along the input dim (default 64 for q4, one group per row for q8); "row" = per row. */
  groupSize?: number | "row";
  /** Quantize the token embedding too (default true; it is the largest tensor). */
  embeddings?: boolean;
  /** Tensor names matching any of these stay unquantized. */
  exclude?: readonly RegExp[];
  /** In a q4 checkpoint, tensors matching any of these are stored as q8 instead (mixed precision). */
  q8?: readonly RegExp[];
  /**
   * Least-squares refinement of each q4 group's scale/bias (default true; see
   * quantizeMatrix). q8 tensors always keep the range fit: at 8 bits the refit
   * clips outliers, which measured slightly worse end to end.
   */
  refine?: boolean;
  /** Extra `__metadata__` entries (e.g. the source repo). */
  metadata?: Readonly<Record<string, string>>;
}

export const DEFAULT_GROUP_SIZE = 64;

/**
 * Whether the converter quantizes this tensor: a 2-D float `*.weight` with
 * at least 64 rows whose input dim the group size divides — the encoder's
 * Wqkv / Wo / Wi / mlp.Wo, the decision head's in_proj / out_proj /
 * linear1 / linear2, scorer.layers.1 and act_head.layers.0 when their shapes
 * allow, and the token embedding (unless `embeddings: false`). Norms,
 * biases, `type_emb`, the 1-row scorer output and `temperature` stay as they are.
 */
export function shouldQuantize(name: string, dtype: string, shape: readonly number[], opts: QuantizeOptions): boolean {
  if (!["F16", "BF16", "F32"].includes(dtype) || shape.length !== 2 || !name.endsWith(".weight")) return false;
  if (/(^|\.)(norm\d*|[a-z_]*norm)\./.test(name) || name === "type_emb.weight") return false;
  if (name.endsWith("tok_embeddings.weight") && opts.embeddings === false) return false;
  if (opts.exclude?.some((re) => re.test(name))) return false;
  const [rows, cols] = shape as [number, number];
  if (rows < 64) return false;
  return groupOf(opts, bitsOf(name, opts), cols) !== null;
}

/** The width a tensor is stored at: `opts.bits`, or 8 when a q8 pattern matches. */
export function bitsOf(name: string, opts: QuantizeOptions): QuantBits {
  return opts.bits === 4 && opts.q8?.some((re) => re.test(name)) ? 8 : opts.bits;
}

/** The group size used for a matrix with `cols` inputs, or null when it does not divide. */
function groupOf(opts: QuantizeOptions, bits: QuantBits, cols: number): number | null {
  const g = opts.groupSize ?? DEFAULT_GROUP_SIZE;
  const size = g === "row" ? cols : g;
  if (cols % size) return null;
  if (bits === 4 && (size % 2 || cols % 2)) return null;
  return size;
}

export interface QuantizeReport {
  scheme: QuantScheme;
  groupSize: number | "row";
  /** Tensors stored quantized. */
  quantized: string[];
  /** Of those, the ones stored as q8 in a q4 checkpoint (`q8` patterns). */
  promoted: string[];
  /** Tensors kept in their original dtype. */
  kept: string[];
  bytesIn: number;
  bytesOut: number;
}

/**
 * Converts an opened float checkpoint to a quantized safetensors file (bytes).
 * Reads one tensor at a time, so peak memory is the quantized output plus
 * one float32 copy of the largest tensor.
 */
export async function quantizeSafetensors(file: LazySafetensors, opts: QuantizeOptions): Promise<{ bytes: Uint8Array; report: QuantizeReport }> {
  if (opts.bits !== 8 && opts.bits !== 4) throw new RangeError(`bits must be 8 or 4, got ${String(opts.bits)}`);
  const g = opts.groupSize ?? DEFAULT_GROUP_SIZE;
  if (g !== "row" && !(Number.isInteger(g) && g > 0)) throw new RangeError(`group size must be a positive integer or "row", got ${String(g)}`);
  if (quantMetadata(file.metadata)) throw new Error("laya quantize: the source checkpoint is already quantized");
  const scheme: QuantScheme = opts.bits === 8 ? "q8" : "q4";
  const out = new Map<string, TensorInput>();
  const report: QuantizeReport = { scheme, groupSize: g, quantized: [], promoted: [], kept: [], bytesIn: 0, bytesOut: 0 };
  const dtypes = new Set<string>();
  for (const name of file.names()) {
    const info = file.info(name);
    const bytes = await file.readBytes(name);
    report.bytesIn += bytes.byteLength;
    if (/\.(scales|biases)$/.test(name) && file.has(name.replace(/\.(scales|biases)$/, ""))) {
      throw new Error(`laya quantize: tensor name ${name} collides with the quantization companions`);
    }
    if (!shouldQuantize(name, info.dtype, info.shape, opts)) {
      out.set(name, { dtype: info.dtype, shape: info.shape, data: bytes });
      report.kept.push(name);
      continue;
    }
    dtypes.add(info.dtype);
    const [rows, cols] = info.shape as [number, number];
    const bits = bitsOf(name, opts);
    const q = quantizeMatrix(toFloat32(info.dtype, bytes), rows, cols, bits, groupOf(opts, bits, cols)!, { refine: bits === 4 && (opts.refine ?? true) });
    const G = cols / q.groupSize;
    if (bits !== opts.bits) report.promoted.push(name);
    out.set(name, bits === 8 ? { dtype: "I8", shape: [rows, cols], data: q.data } : { dtype: "U8", shape: [rows, cols / 2], data: q.data });
    out.set(name + ".scales", { dtype: "F16", shape: [rows, G], data: q.scales });
    if (q.biases) out.set(name + ".biases", { dtype: "F16", shape: [rows, G], data: q.biases });
    report.quantized.push(name);
  }
  const metadata: Record<string, string> = {
    ...opts.metadata,
    laya_quant: scheme,
    group_size: String(g),
    version: QUANT_FORMAT_VERSION,
    laya_quant_source_dtype: [...dtypes].join(",") || "none",
  };
  const bytes = writeSafetensors(out, metadata);
  report.bytesOut = bytes.byteLength;
  return { bytes, report };
}

// ------------------------------------------------------------------ loader
/** A tensor as read from the file: safetensors dtype + typed view. */
export interface RawTensor {
  dtype: SafeDType;
  shape: readonly number[];
  data: ArrayBufferView;
}

/**
 * Splits a quantized file's tensors into quantized matrices (base name →
 * matrix) and plain tensors, validating every shape. Companion `.scales` /
 * `.biases` tensors are folded into their matrix.
 */
export function groupQuantized(tensors: ReadonlyMap<string, RawTensor>): { quantized: Map<string, QuantizedMatrix>; plain: Map<string, RawTensor> } {
  const quantized = new Map<string, QuantizedMatrix>();
  const plain = new Map<string, RawTensor>();
  const companion = new Set<string>();
  for (const [name, t] of tensors) {
    const scales = tensors.get(name + ".scales");
    if (!scales) continue;
    const where = `laya: quantized tensor ${name}`;
    if (scales.dtype !== "F16" || scales.shape.length !== 2) throw new Error(`${where}: .scales must be F16 [rows, groups]`);
    const [rows, G] = scales.shape as [number, number];
    let bits: QuantBits, cols: number, biases: Float16Array | undefined;
    if (t.dtype === "I8") {
      bits = 8;
      cols = t.shape[1]!;
    } else if (t.dtype === "U8") {
      bits = 4;
      cols = 2 * t.shape[1]!;
      const b = tensors.get(name + ".biases");
      if (!b || b.dtype !== "F16" || b.shape[0] !== rows || b.shape[1] !== G) throw new Error(`${where}: q4 needs .biases F16 [${rows}, ${G}]`);
      biases = b.data as Float16Array;
      companion.add(name + ".biases");
    } else throw new Error(`${where}: dtype ${t.dtype} is not a quantized dtype (I8 or U8)`);
    if (t.shape.length !== 2 || t.shape[0] !== rows || cols % G) throw new Error(`${where}: shape [${t.shape}] does not match .scales [${scales.shape}]`);
    companion.add(name + ".scales");
    quantized.set(name, { bits, rows, cols, data: t.data as Int8Array | Uint8Array, scales: scales.data as Float16Array, ...(biases ? { biases } : {}), groupSize: cols / G });
  }
  for (const [name, t] of tensors) if (!quantized.has(name) && !companion.has(name)) plain.set(name, t);
  return { quantized, plain };
}
