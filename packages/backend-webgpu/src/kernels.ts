/**
 * WGSL kernel generators. Every generator is a pure function of its
 * specialization; the returned `key` is the pipeline-cache key.
 *
 * Conventions:
 * - Storage element types: f32 (also bf16, rounded on store), f16, i32,
 *   u32 (bool, 0/1). Math happens in f32 (or i32 for all-integer ops);
 *   accumulations (matmul, reductions, norms, softmax, attention) are f32.
 * - 1-D launches use a 2-D grid (x ≤ 65535) and `num_workgroups` to
 *   linearize, so sizes beyond 16M elements work.
 */
import type { BindingSpec, KernelSource, ParamSpec } from "./runtime.ts";

export type SType = "f32" | "f16" | "i32" | "u32";
export type CType = "f32" | "i32";
/** Storage kind + whether stores must round to bf16 precision. */
export interface Kind {
  st: SType;
  bf16?: boolean;
}

export const kindKey = (k: Kind) => k.st + (k.bf16 ? "b" : "");

/** Load expression converting a storage element to the compute type. */
export function ld(k: Kind, expr: string, c: CType): string {
  if (k.st === c) return expr;
  return `${c}(${expr})`;
}

/** Store expression converting a compute value to the storage type. */
export function st(k: Kind, expr: string, c: CType): string {
  if (k.st === "u32") return `select(0u, 1u, (${expr}) != ${c === "f32" ? "0.0" : "0"})`;
  if (k.bf16) return `bf16r(f32(${expr}))`;
  if (k.st === c) return expr;
  return `${k.st}(${expr})`;
}

const needsF16 = (...ks: Kind[]) => ks.some((k) => k.st === "f16");

export const HELPERS = /* wgsl */ `
fn bf16r(v: f32) -> f32 {
  let u = bitcast<u32>(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) { return v; }
  return bitcast<f32>((u + 0x7fffu + ((u >> 16u) & 1u)) & 0xffff0000u);
}
// erf with max abs error ~1.6e-7 in f32: odd polynomial on |x|<1 (least-squares
// fit), Numerical Recipes erfc Chebyshev form (rel. err 1.2e-7) beyond.
fn erf_(x: f32) -> f32 {
  let z = abs(x);
  if (z < 1.0) {
    let u = x * x;
    var p = 7.933111375e-5;
    p = fma(u, p, -8.034733209e-4);
    p = fma(u, p, 5.191205827e-3);
    p = fma(u, p, -2.685539311e-2);
    p = fma(u, p, 1.128362558e-1);
    p = fma(u, p, -3.761262987e-1);
    p = fma(u, p, 1.128379167);
    return x * p;
  }
  let t = 1.0 / (1.0 + 0.5 * z);
  var p = 0.17087277;
  p = fma(t, p, -0.82215223);
  p = fma(t, p, 1.48851587);
  p = fma(t, p, -1.13520398);
  p = fma(t, p, 0.27886807);
  p = fma(t, p, -0.18628806);
  p = fma(t, p, 0.09678418);
  p = fma(t, p, 0.37409196);
  p = fma(t, p, 1.00002368);
  p = fma(t, p, -1.26551223);
  let r = 1.0 - t * exp(-z * z + p);
  return select(r, -r, x < 0.0);
}
fn gelu_(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_(x * 0.70710678118654752)); }
`;

const FLAT_IDX = /* wgsl */ `
  let wg = wid.x + wid.y * nwg.x;
  let i = wg * WG + lid.x;`;
const ENTRY = (wg: number) =>
  `@compute @workgroup_size(${wg}) fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>)`;

/** Grid for `n` workgroups with the 65535-per-dim limit. */
export function grid(n: number): [number, number, number] {
  if (n <= 65535) return [Math.max(1, n), 1, 1];
  const y = Math.ceil(n / 65535);
  return [Math.ceil(n / y), y, 1];
}

// ---------------------------------------------------------------------------
// Elementwise (n-ary, broadcasting) — covers unary, binary, where and casts.

export interface NaryInput {
  kind: Kind;
  /** true: same numel as output and contiguous → flat index. */
  flat: boolean;
}

/**
 * `expr` uses a, b, c for the inputs (compute type) and P.s for a scalar.
 * Broadcast inputs use params sh (output shape, rank 8 padded) and st{j}
 * (input strides, 0 on broadcast axes).
 */
export function naryKernel(op: string, expr: string, ins: NaryInput[], out: Kind, c: CType, helpers = "", names: readonly string[] = ["a", "b", "c"]): KernelSource {
  const WG = 256;
  if (ins.length > names.length) throw new Error(`nary ${op}: ${ins.length} inputs, only ${names.length} names`);
  const bindings: BindingSpec[] = ins.map((k, j) => ({ name: `in${j}`, elem: k.kind.st, access: "read" }));
  bindings.push({ name: "outp", elem: out.st, access: "read_write" });
  const params: [string, ParamSpec[number][1]][] = [["n", "u32"], ["s", "f32"]];
  const anyBroadcast = ins.some((k) => !k.flat);
  if (anyBroadcast) params.push(["sh", "vec8"]);
  ins.forEach((k, j) => {
    params.push([`o${j}`, "u32"]);
    if (!k.flat) params.push([`st${j}`, "vec8"]);
  });
  let body = `const WG = ${WG}u;\n${HELPERS}\n${helpers}\n${ENTRY(WG)} {\n  let lid = lid3;${FLAT_IDX}\n  if (i >= P.n) { return; }\n`;
  if (anyBroadcast) {
    ins.forEach((k, j) => {
      if (!k.flat) body += `  var off${j} = P.o${j};\n`;
    });
    body += `  var r = i;\n  for (var d = 7i; d >= 0i; d--) {\n    let du = u32(d);\n    let dim = P.sh[du / 4u][du % 4u];\n    let cd = r % dim; r = r / dim;\n`;
    ins.forEach((k, j) => {
      if (!k.flat) body += `    off${j} += cd * P.st${j}[du / 4u][du % 4u];\n`;
    });
    body += `  }\n`;
  }
  ins.forEach((k, j) => {
    const idx = k.flat ? `P.o${j} + i` : `off${j}`;
    body += `  let ${names[j]} = ${ld(k.kind, `in${j}[${idx}]`, c)};\n`;
  });
  body += `  outp[i] = ${st(out, expr, c)};\n}\n`;
  const key = `nary:${op}:${ins.map((k) => kindKey(k.kind) + (k.flat ? "f" : "s")).join(",")}:${kindKey(out)}:${c}`;
  return { key, bindings, params, body, f16: needsF16(out, ...ins.map((k) => k.kind)) };
}

// ---------------------------------------------------------------------------
// Strided copy (transpose / slice / concat / split / non-contiguous cast).

/**
 * Strided copy specialized on the (collapsed) rank R ≤ 8 and on V elements
 * per thread (V = 4 when the innermost axis is unit-stride on both sides and
 * everything is 4-aligned; the shape's innermost extent is then given in V-groups).
 */
export function copyKernel(inp: Kind, out: Kind, R = 8, V = 1): KernelSource {
  const WG = 256;
  const c: CType = inp.st === "i32" && out.st === "i32" ? "i32" : "f32";
  let dec = "";
  for (let d = 7; d >= 8 - R; d--) {
    const q = `${Math.floor(d / 4)}u][${d % 4}u`;
    dec += d > 8 - R
      ? `  { let dim = P.sh[${q}]; let cd = r % dim; r = r / dim; io += cd * P.ist[${q}]; oo += cd * P.ost[${q}]; }\n`
      : `  io += r * P.ist[${q}]; oo += r * P.ost[${q}];\n`;
  }
  let mv = "";
  for (let v = 0; v < V; v++) mv += `  outp[oo + ${v}u] = ${st(out, ld(inp, `inp[io + ${v}u]`, c), c)};\n`;
  const body = `const WG = ${WG}u;\n${HELPERS}\n${ENTRY(WG)} {\n  let lid = lid3;${FLAT_IDX}
  if (i >= P.n) { return; }
  var r = i; var io = P.io; var oo = P.oo;
${dec}${mv}}`;
  return {
    key: `copy:${kindKey(inp)}:${kindKey(out)}:${R}:${V}`,
    bindings: [
      { name: "inp", elem: inp.st, access: "read" },
      { name: "outp", elem: out.st, access: "read_write" },
    ],
    params: [["n", "u32"], ["io", "u32"], ["oo", "u32"], ["sh", "vec8"], ["ist", "vec8"], ["ost", "vec8"]],
    body,
    f16: needsF16(inp, out),
  };
}

// ---------------------------------------------------------------------------
// Reductions over [outer, R, inner] (one thread per output).

export function reduceKernel(op: "sum" | "max" | "min" | "mean", inp: Kind, out: Kind): KernelSource {
  const WG = 256;
  const c: CType = op !== "mean" && (inp.st === "i32" || inp.st === "u32") ? "i32" : "f32";
  const acc = op === "sum" || op === "mean" ? "acc = acc + v;" : `acc = ${op}(acc, v);`;
  const res = op === "mean" ? "acc / f32(P.R)" : "acc";
  const body = `const WG = ${WG}u;\n${ENTRY(WG)} {\n  let lid = lid3;${FLAT_IDX}
  if (i >= P.n) { return; }
  let o = i / P.inner; let inn = i % P.inner;
  let base = P.off + o * P.R * P.inner + inn;
  var acc = ${ld(inp, "inp[base]", c)};
  for (var r = 1u; r < P.R; r++) {
    let v = ${ld(inp, "inp[base + r * P.inner]", c)};
    ${acc}
  }
  outp[i] = ${out.st === "u32" ? "select(0u, 1u, acc != 0)" : st(out, res, c)};
}`;
  return {
    key: `reduce:${op}:${kindKey(inp)}:${kindKey(out)}`,
    bindings: [
      { name: "inp", elem: inp.st, access: "read" },
      { name: "outp", elem: out.st, access: "read_write" },
    ],
    params: [["n", "u32"], ["R", "u32"], ["inner", "u32"], ["off", "u32"]],
    body: (out.bf16 ? HELPERS : "") + body,
    f16: needsF16(inp, out),
  };
}

const ROW_REDUCE = (WG: number) => /* wgsl */ `
var<workgroup> red: array<f32, ${WG}>;
fn rsum(v: f32, lid: u32) -> f32 {
  red[lid] = v;
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s = s >> 1u) {
    if (lid < s) { red[lid] = red[lid] + red[lid + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
fn rmax(v: f32, lid: u32) -> f32 {
  red[lid] = v;
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s = s >> 1u) {
    if (lid < s) { red[lid] = max(red[lid], red[lid + s]); }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}`;

/** Softmax along the last axis: one workgroup per row. */
export function softmaxRowKernel(inp: Kind, out: Kind): KernelSource {
  const WG = 128;
  const body = `const WG = ${WG}u;\nconst NEG = -3.4028234e38;\n${HELPERS}\n${ROW_REDUCE(WG)}
${ENTRY(WG)} {
  let lid = lid3.x;
  let row = wid.x + wid.y * nwg.x;
  if (row >= P.rows) { return; }
  let base = P.off + row * P.D;
  let ob = row * P.D;
  var m = NEG;
  for (var j = lid; j < P.D; j += WG) { m = max(m, ${ld(inp, "inp[base + j]", "f32")}); }
  m = rmax(m, lid);
  var s = 0.0;
  for (var j = lid; j < P.D; j += WG) { s += exp(${ld(inp, "inp[base + j]", "f32")} - m); }
  s = rsum(s, lid);
  let inv = 1.0 / s;
  for (var j = lid; j < P.D; j += WG) { outp[ob + j] = ${st(out, `exp(${ld(inp, "inp[base + j]", "f32")} - m) * inv`, "f32")}; }
}`;
  return {
    key: `softmaxrow:${kindKey(inp)}:${kindKey(out)}`,
    bindings: [
      { name: "inp", elem: inp.st, access: "read" },
      { name: "outp", elem: out.st, access: "read_write" },
    ],
    params: [["rows", "u32"], ["D", "u32"], ["off", "u32"]],
    body,
    f16: needsF16(inp, out),
  };
}

/** Softmax along a non-last axis of [outer, R, inner]: one thread per column. */
export function softmaxColKernel(inp: Kind, out: Kind): KernelSource {
  const WG = 256;
  const L = (e: string) => ld(inp, `inp[${e}]`, "f32");
  const body = `const WG = ${WG}u;\n${HELPERS}\n${ENTRY(WG)} {\n  let lid = lid3;${FLAT_IDX}
  if (i >= P.n) { return; }
  let o = i / P.inner; let inn = i % P.inner;
  let base = o * P.R * P.inner + inn;
  var m = ${L("P.off + base")};
  for (var r = 1u; r < P.R; r++) { m = max(m, ${L("P.off + base + r * P.inner")}); }
  var s = 0.0;
  for (var r = 0u; r < P.R; r++) { s += exp(${L("P.off + base + r * P.inner")} - m); }
  for (var r = 0u; r < P.R; r++) { outp[base + r * P.inner] = ${st(out, `exp(${L("P.off + base + r * P.inner")} - m) / s`, "f32")}; }
}`;
  return {
    key: `softmaxcol:${kindKey(inp)}:${kindKey(out)}`,
    bindings: [
      { name: "inp", elem: inp.st, access: "read" },
      { name: "outp", elem: out.st, access: "read_write" },
    ],
    params: [["n", "u32"], ["R", "u32"], ["inner", "u32"], ["off", "u32"]],
    body,
    f16: needsF16(inp, out),
  };
}

/** LayerNorm over the last axis; f32 two-pass stats. */
export function layerNormKernel(inp: Kind, w: Kind | null, b: Kind | null, out: Kind): KernelSource {
  const WG = 128;
  const bindings: BindingSpec[] = [{ name: "inp", elem: inp.st, access: "read" }];
  if (w) bindings.push({ name: "wt", elem: w.st, access: "read" });
  if (b) bindings.push({ name: "bs", elem: b.st, access: "read" });
  bindings.push({ name: "outp", elem: out.st, access: "read_write" });
  let y = "(v - mean) * rstd";
  if (w) y = `${y} * ${ld(w, "wt[P.ow + j]", "f32")}`;
  if (b) y = `${y} + ${ld(b, "bs[P.ob + j]", "f32")}`;
  const body = `const WG = ${WG}u;\n${HELPERS}\n${ROW_REDUCE(WG)}
${ENTRY(WG)} {
  let lid = lid3.x;
  let row = wid.x + wid.y * nwg.x;
  if (row >= P.rows) { return; }
  let base = P.off + row * P.D;
  let orow = row * P.D;
  var s = 0.0;
  for (var j = lid; j < P.D; j += WG) { s += ${ld(inp, "inp[base + j]", "f32")}; }
  let mean = rsum(s, lid) / f32(P.D);
  var q = 0.0;
  for (var j = lid; j < P.D; j += WG) { let d = ${ld(inp, "inp[base + j]", "f32")} - mean; q += d * d; }
  let rstd = inverseSqrt(rsum(q, lid) / f32(P.D) + P.eps);
  for (var j = lid; j < P.D; j += WG) {
    let v = ${ld(inp, "inp[base + j]", "f32")};
    outp[orow + j] = ${st(out, y, "f32")};
  }
}`;
  return {
    key: `layernorm:${kindKey(inp)}:${w ? kindKey(w) : "-"}:${b ? kindKey(b) : "-"}:${kindKey(out)}`,
    bindings,
    params: [["rows", "u32"], ["D", "u32"], ["off", "u32"], ["ow", "u32"], ["ob", "u32"], ["eps", "f32"]],
    body,
    f16: needsF16(inp, out, ...(w ? [w] : []), ...(b ? [b] : [])),
  };
}

// ---------------------------------------------------------------------------
// GEMM: C[batch, M, N] = A[batch, M, K] · B (+ bias[N]).
// transB: B is [N, K] (PyTorch Linear weight); else B is [batch, K, N].
// Tiles live in workgroup memory as vec4 along M / N ([k][m/4], [k][n/4]) so
// the inner loop does TM/4 + TN/4 vec4 loads per TM·TN FMAs; accumulation f32.

// ---------------------------------------------------------------------------
// Quantized weights (Linear B operand): packed u32 words QW [N, K·bits/32]
// (little-endian fields, the MLX / laya-js layout), per-group scales QS
// [N, ⌈K/g⌉] and, for affine, biases QB. The GEMM kernels read B through
// `qw4(row, k)`, which dequantizes 4 consecutive values (k % 4 == 0, all in
// one group since g % 4 == 0) to f32 inside the tile load: w = q·s (+ b).

export interface QuantSpec {
  bits: 4 | 8;
  /** Group size (a multiple of 4). */
  g: number;
  /** Symmetric (signed q, no biases) or affine (unsigned q, biases). */
  sym: boolean;
  /** Storage kind of the scales / biases. */
  scale: Kind;
  /**
   * Round each dequantized weight to f16 before the multiply (f16
   * activations): the same weights host dequantization uploads, so f16
   * results track the dequantize-on-load model.
   */
  r16?: boolean;
}

export const quantKey = (q: QuantSpec | null | undefined): string => (q ? `q${q.bits}${q.sym ? "s" : "a"}g${q.g}${kindKey(q.scale)}${q.r16 ? "r" : ""}` : "");

/** Bindings that replace `B` for a quantized weight (in this order). */
export function quantBindings(q: QuantSpec): BindingSpec[] {
  return [
    { name: "QW", elem: "u32", access: "read" },
    { name: "QS", elem: q.scale.st, access: "read" },
    ...(q.sym ? [] : [{ name: "QB", elem: q.scale.st, access: "read" as const }]),
  ];
}

/** WGSL `fn qw4(row, k) -> vec4<f32>`: dequantized W[row, k..k+3] (needs P.K). */
export function quantHelper(q: QuantSpec): string {
  const G = `((P.K + ${q.g - 1}u) / ${q.g}u)`;
  let v: string;
  if (q.bits === 8) {
    const w = "let w = QW[row * (P.K / 4u) + k / 4u];";
    v = q.sym
      ? `${w}
  let v = vec4<f32>(vec4<i32>(bitcast<i32>(w << 24u) >> 24u, bitcast<i32>(w << 16u) >> 24u, bitcast<i32>(w << 8u) >> 24u, bitcast<i32>(w) >> 24u));`
      : `${w}
  let v = vec4<f32>(vec4<u32>(w & 255u, (w >> 8u) & 255u, (w >> 16u) & 255u, w >> 24u));`;
  } else {
    const w = "let w = QW[row * (P.K / 8u) + k / 8u] >> ((k & 4u) * 4u);";
    v = q.sym
      ? `${w}
  let v = vec4<f32>(vec4<i32>(bitcast<i32>(w << 28u) >> 28u, bitcast<i32>(w << 24u) >> 28u, bitcast<i32>(w << 20u) >> 28u, bitcast<i32>(w << 16u) >> 28u));`
      : `${w}
  let v = vec4<f32>(vec4<u32>(w & 15u, (w >> 4u) & 15u, (w >> 8u) & 15u, (w >> 12u) & 15u));`;
  }
  const gi = `row * ${G} + k / ${q.g}u`;
  const b = q.sym ? "0.0" : `f32(QB[${gi}])`;
  return `
fn qw4(row: u32, k: u32) -> vec4<f32> {
  ${v}
  let gi = ${gi};
  let d = fma(v, vec4<f32>(f32(QS[gi])), vec4<f32>(${b}));
  return ${q.r16 ? "vec4<f32>(vec4<f16>(d))" : "d"};
}
`;
}

/**
 * Dequantizing row gather (quantizedEmbedding): out[r, d] = W[ids[r], d],
 * one thread per 4 consecutive values (D % 4 == 0).
 */
export function quantGatherKernel(q: QuantSpec, out: Kind): KernelSource {
  const WG = 256;
  const body = `const WG = ${WG}u;
${quantHelper(q)}
${ENTRY(WG)} {
  let lid = lid3;${FLAT_IDX}
  if (i >= P.n) { return; }
  let D4 = P.K / 4u;
  let r = i / D4; let d = (i % D4) * 4u;
  let src = u32(clamp(ids[P.oi + r], 0, i32(P.V) - 1));
  let v = qw4(src, d);
  let o = r * P.K + d;
  outp[o] = ${st(out, "v.x", "f32")}; outp[o + 1u] = ${st(out, "v.y", "f32")}; outp[o + 2u] = ${st(out, "v.z", "f32")}; outp[o + 3u] = ${st(out, "v.w", "f32")};
}`;
  return {
    key: `qgather:${quantKey(q)}:${kindKey(out)}`,
    bindings: [...quantBindings(q), { name: "ids", elem: "i32", access: "read" }, { name: "outp", elem: out.st, access: "read_write" }],
    params: [["n", "u32"], ["K", "u32"], ["V", "u32"], ["oi", "u32"]],
    body: (out.bf16 ? HELPERS : "") + body,
    f16: needsF16(q.scale, out, ...(q.r16 ? [{ st: "f16" as const }] : [])),
  };
}

/** Workgroup-memory tiled GEMM (batched matmul, and Linear when unaligned). */
export interface TiledGemmConfig {
  BM: number;
  BN: number;
  BK: number;
  TM: number;
  TN: number;
}
/** Register-blocked GEMM reading global memory directly (Linear, K % 4 == 0). */
export interface DirectGemmConfig {
  TM: number;
  TN: number;
  WX: number;
  WY: number;
}
export interface GemmConfig {
  tiled: TiledGemmConfig;
  /** null disables the direct kernel. */
  direct: DirectGemmConfig | null;
  /** Skinny-Linear configs, each used for M ≤ maxM (first match wins; empty disables). */
  skinny: (SkinnyGemmConfig & { maxM: number })[];
  /**
   * Subgroup-matrix Linear configs (only on devices with f32 8×8×8 subgroup
   * matrices and subgroup size 32): the first with minM < M ≤ maxM wins;
   * null/absent/empty disables.
   */
  sg?: (SgGemmConfig & {
    minM: number;
    maxM?: number;
    /** Only when the grid has at least this many workgroups (ceil(M/BM)·ceil(N/BN)). */
    minGroups?: number;
    /** Only when the padded rows ceil(M/BM)·BM are at most maxPad·M. */
    maxPad?: number;
    /** Not when the workgroup count is within [lo, hi] (a poor last wave). */
    skipGroups?: readonly [number, number];
  })[] | null;
}
/** Tuned on Apple M2 (10-core GPU) via Dawn/Metal: see bench/gemm.ts. */
export const GEMM_DEFAULT: GemmConfig = {
  tiled: { BM: 64, BN: 64, BK: 16, TM: 4, TN: 4 },
  direct: { TM: 4, TN: 8, WX: 4, WY: 16 },
  skinny: [
    { maxM: 40, WX: 16, TN: 4, WY: 4, KS: 4, KP4: 0 },
    { maxM: 64, WX: 8, TN: 4, WY: 8, KS: 4, KP4: 0 },
  ],
  // Tuned on M2 with bench/linear-shapes.ts over the Laya Linear shapes:
  // - 64×64 tiles, 2 subgroups side by side, each 64×32 (8×4 fragments: fewer
  //   fragment loads per MMA): ≈1.95 TFLOP/s for large M (32×64: ≈1.75), but
  //   only with enough workgroups, little row padding, and not ~1 wave + a
  //   small tail (57–79 workgroups on the 10-core M2).
  // - else 32×64 tiles (2 subgroups × 4×4 fragments), split-K 2 for grids
  //   under 48 workgroups (N = 768, M ≈ 65–96).
  sg: [
    { minM: 64, BM: 64, BN: 64, BK: 8, WM: 1, WN: 2, pad: 0, minGroups: 48, skipGroups: [57, 79], maxPad: 1.1 },
    { minM: 64, BM: 32, BN: 64, BK: 8, WM: 1, WN: 2, pad: 0, splitK: [{ maxGroups: 47, S: 2 }] },
  ],
};

/** The 0.2.0 defaults (before per-shape tuning), kept for benchmarks. */
export const GEMM_V020: GemmConfig = {
  tiled: { BM: 64, BN: 64, BK: 16, TM: 4, TN: 4 },
  direct: { TM: 4, TN: 8, WX: 4, WY: 16 },
  skinny: [
    { maxM: 40, WX: 16, TN: 4, WY: 4, KS: 4, KP4: 0 },
    { maxM: 64, WX: 8, TN: 4, WY: 8, KS: 4, KP4: 0 },
  ],
  sg: [{ minM: 64, BM: 32, BN: 64, BK: 8, WM: 1, WN: 2 }],
};

export function gemmKernel(
  a: Kind,
  b: Kind,
  bias: Kind | null,
  out: Kind,
  transB: boolean,
  vecA: boolean,
  vecB: boolean,
  cfg: TiledGemmConfig,
  quant: QuantSpec | null = null,
): KernelSource {
  const { BM, BN, BK, TM, TN } = cfg;
  const TX = BN / TN, TY = BM / TM, WG = TX * TY;
  if (TN % 4 || TM % 4 || BK % 4) throw new Error("gemm: TM, TN, BK must be multiples of 4");
  if (quant && !(transB && vecB)) throw new Error("gemm: quantized B needs transB and K % 4 == 0");
  const BM4 = BM / 4, BN4 = BN / 4;
  const bindings: BindingSpec[] = [
    { name: "A", elem: vecA ? `vec4<${a.st}>` : a.st, access: "read" },
    ...(quant ? quantBindings(quant) : [{ name: "B", elem: vecB ? `vec4<${b.st}>` : b.st, access: "read" as const }]),
  ];
  if (bias) bindings.push({ name: "bias", elem: bias.st, access: "read" });
  bindings.push({ name: "C", elem: out.st, access: "read_write" });
  const f4 = (k: Kind, e: string) => (k.st === "f32" ? e : `vec4<f32>(${e})`);
  const f1 = (k: Kind, e: string) => ld(k, e, "f32");

  // K-contiguous operand (A, or B when transB) → S[k][row/4] (vec4 over 4 rows).
  // Each thread owns whole vec4s in workgroup memory (component writes from
  // different threads into one vec4 would race).
  const loadRowsK = (lim: string, name: string, kind: Kind, vec: boolean, S: string, rows: number, rowBase: string, base: string) => {
    const R4 = rows / 4;
    const rowsOf = (m4: string) => [0, 1, 2, 3].map((i) => `${rowBase} + ${m4} * 4u + ${i}u`);
    if (vec) {
      const n = R4 * (BK / 4);
      const vload = (row: string) => (quant && name === "B" ? `qw4(${row}, kk)` : f4(kind, `${name}[(${base} + (${row}) * P.K + kk) / 4u]`));
      const loads = rowsOf("m4")
        .map((row, i) => `      var q${i} = vec4<f32>(0.0);
      if (${row} < ${lim} && kk < P.K) { q${i} = ${vload(row)}; }`)
        .join("\n");
      return `    for (var t = lid; t < ${n}u; t += ${WG}u) {
      let m4 = t / ${BK / 4}u; let kq = t % ${BK / 4}u;
      let kk = k0 + kq * 4u;
${loads}
      let s = kq * 4u * ${R4}u + m4;
      ${S}[s] = vec4<f32>(q0.x, q1.x, q2.x, q3.x);
      ${S}[s + ${R4}u] = vec4<f32>(q0.y, q1.y, q2.y, q3.y);
      ${S}[s + ${2 * R4}u] = vec4<f32>(q0.z, q1.z, q2.z, q3.z);
      ${S}[s + ${3 * R4}u] = vec4<f32>(q0.w, q1.w, q2.w, q3.w);
    }\n`;
    }
    const n = R4 * BK;
    const loads = rowsOf("m4")
      .map((row, i) => `      var v${i} = 0.0;
      if (${row} < ${lim} && k < P.K) { v${i} = ${f1(kind, `${name}[${base} + (${row}) * P.K + k]`)}; }`)
      .join("\n");
    return `    for (var t = lid; t < ${n}u; t += ${WG}u) {
      let m4 = t / ${BK}u; let kk = t % ${BK}u; let k = k0 + kk;
${loads}
      ${S}[kk * ${R4}u + m4] = vec4<f32>(v0, v1, v2, v3);
    }\n`;
  };
  // N-contiguous B ([K, N]) → Bs[k][n/4].
  const loadBkn = () => {
    const n = (BK * BN) / 4;
    if (vecB) {
      return `    for (var t = lid; t < ${n}u; t += ${WG}u) {
      let kk = t / ${BN4}u; let nq = t % ${BN4}u;
      let k = k0 + kk; let col = c0 + nq * 4u;
      var v = vec4<f32>(0.0);
      if (k < P.K && col < P.N) { v = ${f4(b, `B[(bBase + k * P.N + col) / 4u]`)}; }
      Bs[kk * ${BN4}u + nq] = v;
    }\n`;
    }
    const loads = [0, 1, 2, 3]
      .map((q) => `      var v${q} = 0.0;
      if (k < P.K && col + ${q}u < P.N) { v${q} = ${f1(b, `B[bBase + k * P.N + col + ${q}u]`)}; }`)
      .join("\n");
    return `    for (var t = lid; t < ${n}u; t += ${WG}u) {
      let kk = t / ${BN4}u; let nq = t % ${BN4}u;
      let k = k0 + kk; let col = c0 + nq * 4u;
${loads}
      Bs[kk * ${BN4}u + nq] = vec4<f32>(v0, v1, v2, v3);
    }\n`;
  };

  const MV = TM / 4, NV = TN / 4;
  let decl = "";
  for (let i = 0; i < TM; i++) for (let j = 0; j < NV; j++) decl += `  var c${i}_${j} = vec4<f32>(0.0);\n`;
  let inner = "";
  for (let i = 0; i < MV; i++) inner += `      let a${i} = As[kk * ${BM4}u + ty * ${MV}u + ${i}u];\n`;
  for (let j = 0; j < NV; j++) inner += `      let b${j} = Bs[kk * ${BN4}u + tx * ${NV}u + ${j}u];\n`;
  for (let i = 0; i < TM; i++)
    for (let j = 0; j < NV; j++) inner += `      c${i}_${j} += a${i >> 2}.${"xyzw"[i & 3]} * b${j};\n`;
  let store = "";
  const vecStore = out.st === "f32" && !out.bf16;
  for (let i = 0; i < TM; i++) {
    store += `  { let row = r0 + ty * ${TM}u + ${i}u;\n    if (row < P.M) {\n`;
    for (let j = 0; j < NV; j++) {
      const col0 = `c0 + tx * ${TN}u + ${4 * j}u`;
      const bv = bias ? ` + vec4<f32>(${[0, 1, 2, 3].map((q) => f1(bias, `bias[P.obias + min(${col0} + ${q}u, P.N - 1u)]`)).join(", ")})` : "";
      store += `      { let col = ${col0}; let v = c${i}_${j}${bv};\n`;
      for (let q = 0; q < 4; q++) store += `        if (col + ${q}u < P.N) { C[cBase + row * P.N + col + ${q}u] = ${st(out, `v.${"xyzw"[q]}`, "f32")}; }\n`;
      store += `      }\n`;
    }
    store += `    }\n  }\n`;
  }
  void vecStore;

  const body = `${out.bf16 ? HELPERS : ""}${quant ? quantHelper(quant) : ""}
var<workgroup> As: array<vec4<f32>, ${BK * BM4}>;
var<workgroup> Bs: array<vec4<f32>, ${BK * BN4}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let tx = lid % ${TX}u; let ty = lid / ${TX}u;
  let r0 = wid.y * ${BM}u; let c0 = wid.x * ${BN}u;
  // batch offsets (up to 8 broadcast batch dims)
  var z = wid.z; var aBase = P.oa; var bBase = P.ob;
  for (var d = 7i; d >= 0i; d--) {
    let du = u32(d);
    let dim = P.bsh[du / 4u][du % 4u];
    let cd = z % dim; z = z / dim;
    aBase += cd * P.ast[du / 4u][du % 4u];
    bBase += cd * P.bst[du / 4u][du % 4u];
  }
  let cBase = wid.z * P.M * P.N;
${decl}
  for (var k0 = 0u; k0 < P.K; k0 += ${BK}u) {
${loadRowsK("P.M", "A", a, vecA, "As", BM, "r0", "aBase")}${transB ? loadRowsK("P.N", "B", b, vecB, "Bs", BN, "c0", "bBase") : loadBkn()}
    workgroupBarrier();
    for (var kk = 0u; kk < ${BK}u; kk++) {
${inner}    }
    workgroupBarrier();
  }
${store}}`;
  const key = `gemm:${kindKey(a)}:${quant ? quantKey(quant) : kindKey(b)}:${bias ? kindKey(bias) : "-"}:${kindKey(out)}:${transB ? "T" : "N"}:${vecA ? "v" : "s"}${vecB ? "v" : "s"}:${BM}x${BN}x${BK}/${TM}x${TN}`;
  const params: ParamSpec = [
    ["M", "u32"], ["N", "u32"], ["K", "u32"], ["oa", "u32"], ["ob", "u32"], ["obias", "u32"],
    ["bsh", "vec8"], ["ast", "vec8"], ["bst", "vec8"],
  ];
  return { key, bindings, params, body, f16: needsF16(a, out, ...(quant ? [quant.scale, ...(quant.r16 ? [{ st: "f16" as const }] : [])] : [b]), ...(bias ? [bias] : [])) };
}

/**
 * Register-blocked GEMM without workgroup memory, for transB with K % 4 == 0
 * (the Linear hot path): each thread owns a TM×TN block and streams vec4s of
 * A rows and W rows straight from global memory (served by L1/L2 on Apple and
 * most discrete GPUs). Accumulates dot4 products in f32.
 */
export function gemmDirectKernel(a: Kind, b: Kind, bias: Kind | null, out: Kind, cfg: DirectGemmConfig, quant: QuantSpec | null = null): KernelSource {
  const { TM, TN, WX, WY } = cfg;
  const bindings: BindingSpec[] = [
    { name: "A", elem: `vec4<${a.st}>`, access: "read" },
    ...(quant ? quantBindings(quant) : [{ name: "B", elem: `vec4<${b.st}>`, access: "read" as const }]),
  ];
  if (bias) bindings.push({ name: "bias", elem: bias.st, access: "read" });
  bindings.push({ name: "C", elem: out.st, access: "read_write" });
  const f4 = (k: Kind, e: string) => (k.st === "f32" ? e : `vec4<f32>(${e})`);
  let s = "";
  for (let i = 0; i < TM; i++) s += `  let ar${i} = aBase + min(row0 + ${i}u, P.M - 1u) * K4;\n`;
  for (let j = 0; j < TN; j++) s += quant ? `  let br${j} = min(col0 + ${j}u, P.N - 1u);\n` : `  let br${j} = bBase + min(col0 + ${j}u, P.N - 1u) * K4;\n`;
  for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `  var c${i}_${j} = 0.0;\n`;
  s += `  for (var k = 0u; k < K4; k++) {\n`;
  for (let i = 0; i < TM; i++) s += `    let a${i} = ${f4(a, `A[ar${i} + k]`)};\n`;
  for (let j = 0; j < TN; j++) s += `    let b${j} = ${quant ? `qw4(br${j}, k * 4u)` : f4(b, `B[br${j} + k]`)};\n`;
  for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `    c${i}_${j} += dot(a${i}, b${j});\n`;
  s += `  }\n`;
  for (let i = 0; i < TM; i++) {
    s += `  if (row0 + ${i}u < P.M) {\n`;
    for (let j = 0; j < TN; j++) {
      const bv = bias ? ` + ${ld(bias, `bias[P.obias + col0 + ${j}u]`, "f32")}` : "";
      s += `    if (col0 + ${j}u < P.N) { C[cBase + (row0 + ${i}u) * P.N + col0 + ${j}u] = ${st(out, `c${i}_${j}${bv}`, "f32")}; }\n`;
    }
    s += `  }\n`;
  }
  const body = `${out.bf16 ? HELPERS : ""}${quant ? quantHelper(quant) : ""}
@compute @workgroup_size(${WX}, ${WY}) fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let K4 = P.K / 4u;
  let col0 = (wid.x * ${WX}u + lid.x) * ${TN}u;
  let row0 = (wid.y * ${WY}u + lid.y) * ${TM}u;
  let aBase = P.oa / 4u; let bBase = P.ob / 4u;
  let cBase = 0u;
${s}}`;
  return {
    key: `gemmdirect:${kindKey(a)}:${quant ? quantKey(quant) : kindKey(b)}:${bias ? kindKey(bias) : "-"}:${kindKey(out)}:${TM}x${TN}:${WX}x${WY}`,
    bindings,
    params: [["M", "u32"], ["N", "u32"], ["K", "u32"], ["oa", "u32"], ["ob", "u32"], ["obias", "u32"]],
    body,
    f16: needsF16(a, out, ...(quant ? [quant.scale, ...(quant.r16 ? [{ st: "f16" as const }] : [])] : [b]), ...(bias ? [bias] : [])),
  };
}

/** Skinny Linear (small M, transB, K % 4 == 0): see gemmSkinnyKernel. */
export interface SkinnyGemmConfig {
  /** Column threads per workgroup and columns per thread. */
  WX: number;
  TN: number;
  /** Row threads per workgroup; rows per thread = ceil(M / WY) (specialized). */
  WY: number;
  /** K lanes per output (split-K inside the workgroup, reduced at the end). */
  KS: number;
  /** K panel (vec4s) staged through workgroup memory; 0 = read A from global (faster on Apple M2). */
  KP4: number;
}

/**
 * All M rows (M ≤ WY·TM) live in one workgroup, so each weight row is read
 * from DRAM once; A (small) is read from cache, or staged through
 * workgroup memory in K panels when KP4 > 0. KS lanes split K to raise the
 * thread count and reduce partials in workgroup memory at the end.
 * Designed for the latency path (one short question: M = tokens ≈ 16–64).
 * f32 accumulation.
 */
export function gemmSkinnyKernel(a: Kind, b: Kind, bias: Kind | null, out: Kind, cfg: SkinnyGemmConfig, TM: number, quant: QuantSpec | null = null, q8step = false): KernelSource {
  const { WX, TN, WY, KS, KP4 } = cfg;
  // q8step (quantized, g % 8 == 0, K % 8 == 0, KP4 == 0): 8 values per step
  // with one scale/bias, y += s·Σx·q + b·Σx (Σx shared by every column).
  const fast = !!quant && q8step && quant.g % 8 === 0 && KP4 === 0;
  const WG = WX * WY * KS, RM = WY * TM;
  const bindings: BindingSpec[] = [
    { name: "A", elem: `vec4<${a.st}>`, access: "read" },
    ...(quant ? quantBindings(quant) : [{ name: "B", elem: `vec4<${b.st}>`, access: "read" as const }]),
  ];
  if (bias) bindings.push({ name: "bias", elem: bias.st, access: "read" });
  bindings.push({ name: "C", elem: out.st, access: "read_write" });
  const f4 = (k: Kind, e: string) => (k.st === "f32" ? e : `vec4<f32>(${e})`);
  let s = "";
  for (let j = 0; j < TN; j++) s += quant ? `  let br${j} = min(c0 + ${j * WX}u, P.N - 1u);\n` : `  let br${j} = bBase + min(c0 + ${j * WX}u, P.N - 1u) * K4;\n`;
  const bload = (j: number, k: string) => (quant ? `qw4(br${j}, (${k}) * 4u)` : f4(b, `B[br${j} + ${k}]`));
  for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `  var c${i}_${j} = 0.0;\n`;
  if (fast) {
    const q = quant!;
    const aff = !q.sym;
    const wpr = q.bits / 4; // u32 words per 8 values
    s += `  let K8 = P.K / 8u; let WPR = P.K / ${32 / q.bits}u; let G = (P.K + ${q.g - 1}u) / ${q.g}u;\n`;
    for (let j = 0; j < TN; j++) s += `  let wb${j} = br${j} * WPR; let sb${j} = br${j} * G;\n`;
    for (let i = 0; i < TM; i++) s += `  let ar${i} = aBase + min(ty * ${TM}u + ${i}u, P.M - 1u) * K4;\n`;
    const unpack = (w: string, shift: number, bits: number, sym: boolean) => {
      const f = (o: number) => (sym ? `bitcast<i32>(${w} << ${32 - bits - o}u) >> ${32 - bits}u` : `(${w} >> ${o}u) & ${(1 << bits) - 1}u`);
      return `vec4<f32>(vec4<${sym ? "i32" : "u32"}>(${[0, 1, 2, 3].map((t) => f(shift + t * bits)).join(", ")}))`;
    };
    s += `  for (var k8 = ks; k8 < K8; k8 += ${KS}u) {\n    let gi = (k8 * 8u) / ${q.g}u;\n`;
    for (let i = 0; i < TM; i++) {
      s += `    let a${i}l = ${f4(a, `A[ar${i} + 2u * k8]`)}; let a${i}h = ${f4(a, `A[ar${i} + 2u * k8 + 1u]`)};\n`;
      if (aff && !q.r16) s += `    let sa${i} = dot(a${i}l + a${i}h, vec4<f32>(1.0));\n`;
    }
    for (let j = 0; j < TN; j++) {
      if (wpr === 1) s += `    { let w = QW[wb${j} + k8]; let ql = ${unpack("w", 0, 4, q.sym)}; let qh = ${unpack("w", 16, 4, q.sym)};\n`;
      else s += `    { let w0 = QW[wb${j} + 2u * k8]; let w1 = QW[wb${j} + 2u * k8 + 1u]; let ql = ${unpack("w0", 0, 8, q.sym)}; let qh = ${unpack("w1", 0, 8, q.sym)};\n`;
      s += `      let sc = f32(QS[sb${j} + gi]);${aff ? ` let bi = f32(QB[sb${j} + gi]);` : ""}\n`;
      if (q.r16) {
        // weights rounded to f16 (as host dequantization does), then plain dots
        const d = (v: string) => `vec4<f32>(vec4<f16>(fma(${v}, vec4<f32>(sc), vec4<f32>(${aff ? "bi" : "0.0"}))))`;
        s += `      let wl = ${d("ql")}; let wh = ${d("qh")};\n`;
        for (let i = 0; i < TM; i++) s += `      c${i}_${j} += dot(a${i}l, wl) + dot(a${i}h, wh);\n`;
      } else for (let i = 0; i < TM; i++) s += `      c${i}_${j} += sc * (dot(a${i}l, ql) + dot(a${i}h, qh))${aff ? ` + bi * sa${i}` : ""};\n`;
      s += `    }\n`;
    }
    s += `  }\n`;
  } else if (KP4 === 0) {
    // A straight from global memory (small; stays in cache), no barriers in the K loop.
    for (let i = 0; i < TM; i++) s += `  let ar${i} = aBase + min(ty * ${TM}u + ${i}u, P.M - 1u) * K4;\n`;
    s += `  for (var k = ks; k < K4; k += ${KS}u) {\n`;
    for (let j = 0; j < TN; j++) s += `    let b${j} = ${bload(j, "k")};\n`;
    for (let i = 0; i < TM; i++) {
      s += `    { let a = ${f4(a, `A[ar${i} + k]`)};\n`;
      for (let j = 0; j < TN; j++) s += `      c${i}_${j} += dot(a, b${j});\n`;
      s += `    }\n`;
    }
    s += `  }\n`;
  } else {
    s += `  for (var kp = 0u; kp < K4; kp += ${KP4}u) {
      for (var t = lid; t < ${RM * KP4}u; t += ${WG}u) {
        let r = t / ${KP4}u; let kk = kp + t % ${KP4}u;
        var v = vec4<f32>(0.0);
        if (r < P.M && kk < K4) { v = ${f4(a, "A[aBase + r * K4 + kk]")}; }
        As[t] = v;
      }
      workgroupBarrier();
      let kn = min(${KP4}u, K4 - kp);
      for (var kk = ks; kk < kn; kk += ${KS}u) {\n`;
    for (let j = 0; j < TN; j++) s += `      let b${j} = ${bload(j, "kp + kk")};\n`;
    for (let i = 0; i < TM; i++) {
      s += `      { let a = As[(ty * ${TM}u + ${i}u) * ${KP4}u + kk];\n`;
      for (let j = 0; j < TN; j++) s += `        c${i}_${j} += dot(a, b${j});\n`;
      s += `      }\n`;
    }
    s += `    }\n    workgroupBarrier();\n  }\n`;
  }
  if (KS > 1) {
    // Reduce one accumulator at a time through red[WG] (keeps workgroup memory small).
    for (let i = 0; i < TM; i++)
      for (let j = 0; j < TN; j++) {
        const bv = bias ? ` + ${ld(bias, `bias[P.obias + col]`, "f32")}` : "";
        s += `  red[lid] = c${i}_${j};
  workgroupBarrier();
  if (ks == 0u) {
    var acc = 0.0;
    for (var q = 0u; q < ${KS}u; q++) { acc += red[lid + q]; }
    let row = ty * ${TM}u + ${i}u; let col = c0 + ${j * WX}u;
    if (row < P.M && col < P.N) { C[row * P.N + col] = ${st(out, `acc${bv}`, "f32")}; }
  }
  workgroupBarrier();\n`;
      }
  } else {
    for (let i = 0; i < TM; i++) {
      s += `  { let row = ty * ${TM}u + ${i}u;\n    if (row < P.M) {\n`;
      for (let j = 0; j < TN; j++) {
        const col = `c0 + ${j * WX}u`;
        const bv = bias ? ` + ${ld(bias, `bias[P.obias + ${col}]`, "f32")}` : "";
        s += `      if (${col} < P.N) { C[row * P.N + ${col}] = ${st(out, `c${i}_${j}${bv}`, "f32")}; }\n`;
      }
      s += `    }\n  }\n`;
    }
  }
  const body = `${out.bf16 ? HELPERS : ""}${quant ? quantHelper(quant) : ""}
${KP4 ? `var<workgroup> As: array<vec4<f32>, ${RM * KP4}>;` : ""}
${KS > 1 ? `var<workgroup> red: array<f32, ${WG}>;` : ""}
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let K4 = P.K / 4u;
  let ks = lid % ${KS}u; let tx = (lid / ${KS}u) % ${WX}u; let ty = lid / ${KS * WX}u;
  // columns interleaved across threads: c0 + j*WX
  let c0 = wid.x * ${WX * TN}u + tx;
  let aBase = P.oa / 4u; let bBase = P.ob / 4u;
${s}}`;
  return {
    key: `gemmskinny:${kindKey(a)}:${quant ? quantKey(quant) + (fast ? "8" : "") : kindKey(b)}:${bias ? kindKey(bias) : "-"}:${kindKey(out)}:${WX}x${WY}x${KS}/${TM}x${TN}/${KP4}`,
    bindings,
    params: [["M", "u32"], ["N", "u32"], ["K", "u32"], ["oa", "u32"], ["ob", "u32"], ["obias", "u32"]],
    body,
    f16: needsF16(a, out, ...(quant ? [quant.scale, ...(quant.r16 ? [{ st: "f16" as const }] : [])] : [b]), ...(bias ? [bias] : [])),
  };
}

// ---------------------------------------------------------------------------
// Flash-style SDPA: one workgroup per (query block, head, batch); online
// softmax in f32 over key tiles held in workgroup memory.

/** Workgroup memory (bytes) of the generic sdpa kernel for a tile config. */
export function sdpaBytes(D: number, BQ: number, BKV: number): number {
  return 4 * (BQ * D + BKV * (D + 1) + BKV * D + BQ * BKV + 3 * BQ);
}

/**
 * Generic sdpa tile config for head dim D. `limit` is the device's
 * maxComputeWorkgroupStorageSize: tiles halve (keys first, then queries)
 * until the kernel fits (the WebGPU default of 16 KiB, for example).
 */
export function sdpaConfig(D: number, limit = Infinity): { BQ: number; BKV: number; WG: number } {
  let c: { BQ: number; BKV: number; WG: number };
  if (D <= 64) c = { BQ: 32, BKV: 32, WG: 128 };
  else if (D <= 128) c = { BQ: 16, BKV: 16, WG: 128 };
  else if (D <= 256) c = { BQ: 8, BKV: 8, WG: 64 };
  else throw new Error(`sdpa: head dim ${D} > 256 not supported`);
  while (sdpaBytes(D, c.BQ, c.BKV) > limit) {
    if (c.BKV >= c.BQ && c.BKV > 1) c = { ...c, BKV: c.BKV / 2 };
    else if (c.BQ > 1) c = { ...c, BQ: c.BQ / 2 };
    else throw new Error(`sdpa: head dim ${D} needs ${sdpaBytes(D, 1, 1)} bytes of workgroup memory, the device allows ${limit}`);
  }
  return c;
}

/** Workgroup memory (bytes) of the fast (D = 32 / 64) sdpa kernel. */
export function sdpaFastBytes(D: number, masked: boolean): number {
  const BQ = 32, BKV = 16, WG = 128, D4 = D / 4, KP = D4 + 1;
  return 16 * (BQ * D4 + BKV * KP + BKV * D4) + 4 * (BQ * (BKV + 1) + WG + 3 * BQ) + (masked ? 16 : 0);
}

export function sdpaKernel(q: Kind, k: Kind, v: Kind, mask: Kind | null, out: Kind, D: number, limit = Infinity): KernelSource {
  const { BQ, BKV, WG } = sdpaConfig(D, limit);
  const NS = Math.ceil((BQ * BKV) / WG);
  const NO = Math.ceil((BQ * D) / WG);
  const bindings: BindingSpec[] = [
    { name: "Q", elem: q.st, access: "read" },
    { name: "Kt", elem: k.st, access: "read" },
    { name: "V", elem: v.st, access: "read" },
  ];
  if (mask) bindings.push({ name: "Mk", elem: mask.st, access: "read" });
  bindings.push({ name: "O", elem: out.st, access: "read_write" });
  let decl = "";
  for (let i = 0; i < NO; i++) decl += `  var o${i} = 0.0;\n`;
  let scores = "";
  for (let i = 0; i < NS; i++) {
    scores += `    { let e = lid + ${i * WG}u;
      if (e < ${BQ * BKV}u) {
        let qi = e / ${BKV}u; let kj = e % ${BKV}u;
        let qg = q0 + qi; let kg = kt + kj;
        var s = NEG;
        if (qg < P.Lq && kg < P.Lk) {
          var acc = 0.0;
          for (var d = 0u; d < ${D}u; d++) { acc += Qs[qi * ${D}u + d] * Ks[kj * ${D + 1}u + d]; }
          s = acc;
${mask ? `          if (Mk[mBase + qg * P.msq + kg * P.msk] == 0u) { s = NEG; }\n` : ""}        }
        S[e] = s;
      }
    }\n`;
  }
  let accum = "";
  for (let i = 0; i < NO; i++) {
    accum += `    { let e = lid + ${i * WG}u;
      if (e < ${BQ * D}u) {
        let qi = e / ${D}u; let d = e % ${D}u;
        var acc = o${i} * Al[qi];
        for (var j = 0u; j < ${BKV}u; j++) { acc += S[qi * ${BKV}u + j] * Vs[j * ${D}u + d]; }
        o${i} = acc;
      }
    }\n`;
  }
  let write = "";
  for (let i = 0; i < NO; i++) {
    write += `  { let e = lid + ${i * WG}u;
    if (e < ${BQ * D}u) {
      let qi = e / ${D}u; let d = e % ${D}u;
      if (q0 + qi < P.Lq) { O[oBase + (q0 + qi) * ${D}u + d] = ${st(out, `o${i} / Ls[qi]`, "f32")}; }
    }
  }\n`;
  }
  const body = `${out.bf16 ? HELPERS : ""}
const NEG = -3.0e38;
var<workgroup> Qs: array<f32, ${BQ * D}>;
var<workgroup> Ks: array<f32, ${BKV * (D + 1)}>;
var<workgroup> Vs: array<f32, ${BKV * D}>;
var<workgroup> S: array<f32, ${BQ * BKV}>;
var<workgroup> Ms: array<f32, ${BQ}>;
var<workgroup> Ls: array<f32, ${BQ}>;
var<workgroup> Al: array<f32, ${BQ}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let q0 = wid.x * ${BQ}u; let h = wid.y; let b = wid.z;
  let qBase = P.oq + ((b * P.H + h) * P.Lq) * ${D}u;
  let kBase = P.ok + ((b * P.H + h) * P.Lk) * ${D}u;
  let vBase = P.ov + ((b * P.H + h) * P.Lk) * ${D}u;
  let oBase = ((b * P.H + h) * P.Lq) * ${D}u;
${mask ? "  let mBase = P.om + b * P.msb + h * P.msh;\n" : ""}
  for (var t = lid; t < ${BQ * D}u; t += ${WG}u) {
    let qi = t / ${D}u;
    var x = 0.0;
    if (q0 + qi < P.Lq) { x = ${ld(q, `Q[qBase + q0 * ${D}u + t]`, "f32")} * P.scale; }
    Qs[t] = x;
  }
  if (lid < ${BQ}u) { Ms[lid] = NEG; Ls[lid] = 0.0; }
${decl}
  for (var kt = 0u; kt < P.Lk; kt += ${BKV}u) {
    for (var t = lid; t < ${BKV * D}u; t += ${WG}u) {
      let kj = t / ${D}u;
      var kx = 0.0; var vx = 0.0;
      if (kt + kj < P.Lk) {
        kx = ${ld(k, `Kt[kBase + kt * ${D}u + t]`, "f32")};
        vx = ${ld(v, `V[vBase + kt * ${D}u + t]`, "f32")};
      }
      Ks[kj * ${D + 1}u + t % ${D}u] = kx; Vs[t] = vx;
    }
    workgroupBarrier();
${scores}    workgroupBarrier();
    if (lid < ${BQ}u) {
      let r = lid;
      var mx = NEG;
      for (var j = 0u; j < ${BKV}u; j++) { mx = max(mx, S[r * ${BKV}u + j]); }
      let mprev = Ms[r];
      let mnew = max(mprev, mx);
      var alpha = 1.0;
      var l = Ls[r];
      if (mnew > -1.0e38) {
        alpha = exp(mprev - mnew);
        l = l * alpha;
        for (var j = 0u; j < ${BKV}u; j++) {
          let s = S[r * ${BKV}u + j];
          var p = 0.0;
          if (s > -1.0e38) { p = exp(s - mnew); }
          S[r * ${BKV}u + j] = p;
          l += p;
        }
      } else {
        for (var j = 0u; j < ${BKV}u; j++) { S[r * ${BKV}u + j] = 0.0; }
      }
      Ms[r] = mnew; Ls[r] = l; Al[r] = alpha;
    }
    workgroupBarrier();
${accum}    workgroupBarrier();
  }
${write}}`;
  const params: ParamSpec = [
    ["H", "u32"], ["Lq", "u32"], ["Lk", "u32"], ["oq", "u32"], ["ok", "u32"], ["ov", "u32"], ["scale", "f32"],
    ...(mask ? ([["om", "u32"], ["msb", "u32"], ["msh", "u32"], ["msq", "u32"], ["msk", "u32"]] as const) : []),
  ];
  return {
    key: `sdpa:${kindKey(q)}:${kindKey(k)}:${kindKey(v)}:${mask ? kindKey(mask) : "-"}:${kindKey(out)}:D${D}:${BQ}x${BKV}`,
    bindings,
    params,
    body,
    f16: needsF16(q, k, v, out),
  };
}

/**
 * Register-blocked flash attention for head dims D ∈ {32, 64}: BQ = 32
 * queries × BKV = 16 keys per tile, 128 threads. Scores: each thread a
 * 2×2 block from vec4 tiles; online softmax with 4 lanes per row; P·V:
 * each thread 2 rows × (D/32) vec4. f32 math and accumulation.
 */
export function sdpaFastKernel(q: Kind, k: Kind, v: Kind, mask: Kind | null, out: Kind, D: number): KernelSource {
  const BQ = 32, BKV = 16, WG = 128, D4 = D / 4, KP = D4 + 1, TD = D4 / 8;
  if (D % 32 || D > 64) throw new Error("sdpaFast: D must be 32 or 64");
  const bindings: BindingSpec[] = [
    { name: "Q", elem: `vec4<${q.st}>`, access: "read" },
    { name: "Kt", elem: `vec4<${k.st}>`, access: "read" },
    { name: "V", elem: `vec4<${v.st}>`, access: "read" },
  ];
  if (mask) bindings.push({ name: "Mk", elem: mask.st, access: "read" });
  bindings.push({ name: "O", elem: out.st, access: "read_write" });
  const f4 = (kd: Kind, e: string) => (kd.st === "f32" ? e : `vec4<f32>(${e})`);
  // scores: thread t → rows 2*(t/8) + {0,1}, keys 2*(t%8) + {0,1}
  let score = "";
  for (let a = 0; a < 2; a++) for (let c = 0; c < 2; c++) score += `    var s${a}${c} = 0.0;\n`;
  score += `    for (var d = 0u; d < ${D4}u; d++) {
      let q0 = Qs[(sr) * ${D4}u + d]; let q1 = Qs[(sr + 1u) * ${D4}u + d];
      let k0 = Ks[(sc) * ${KP}u + d]; let k1 = Ks[(sc + 1u) * ${KP}u + d];
      s00 += dot(q0, k0); s01 += dot(q0, k1); s10 += dot(q1, k0); s11 += dot(q1, k1);
    }\n`;
  for (let a = 0; a < 2; a++)
    for (let c = 0; c < 2; c++) {
      score += `    { let qg = q0g + sr + ${a}u; let kg = kt + sc + ${c}u; var s = s${a}${c};
      if (qg >= P.Lq || kg >= P.Lk) { s = NEG; }
${mask ? `      else if (Mk[mBase + qg * P.msq + kg * P.msk] == 0u) { s = NEG; }\n` : ""}      S[(sr + ${a}u) * ${BKV + 1}u + sc + ${c}u] = s; }\n`;
    }
  let decl = "";
  for (let a = 0; a < 2; a++) for (let j = 0; j < TD; j++) decl += `  var o${a}_${j} = vec4<f32>(0.0);\n`;
  let accum = `    let al0 = Al[orow]; let al1 = Al[orow + 1u];\n`;
  for (let j = 0; j < TD; j++) accum += `    o0_${j} *= al0; o1_${j} *= al1;\n`;
  accum += `    for (var j = 0u; j < ${BKV}u; j++) {
      let p0 = S[orow * ${BKV + 1}u + j]; let p1 = S[(orow + 1u) * ${BKV + 1}u + j];\n`;
  for (let j = 0; j < TD; j++) accum += `      { let vv = Vs[j * ${D4}u + ocol + ${j * 8}u]; o0_${j} += p0 * vv; o1_${j} += p1 * vv; }\n`;
  accum += `    }\n`;
  let write = "";
  for (let a = 0; a < 2; a++) {
    write += `  { let qg = q0g + orow + ${a}u;\n    if (qg < P.Lq) {\n      let inv = 1.0 / Ls[orow + ${a}u];\n`;
    for (let j = 0; j < TD; j++) {
      const val = `(o${a}_${j} * inv)`;
      const idx = `oBase + qg * ${D}u + (ocol + ${j * 8}u) * 4u`;
      for (let c = 0; c < 4; c++) write += `      O[${idx} + ${c}u] = ${st(out, `${val}.${"xyzw"[c]}`, "f32")};\n`;
    }
    write += `    }\n  }\n`;
  }
  const body = `${out.bf16 ? HELPERS : ""}
const NEG = -3.0e38;
var<workgroup> Qs: array<vec4<f32>, ${BQ * D4}>;
var<workgroup> Ks: array<vec4<f32>, ${BKV * KP}>;
var<workgroup> Vs: array<vec4<f32>, ${BKV * D4}>;
var<workgroup> S: array<f32, ${BQ * (BKV + 1)}>;
var<workgroup> red: array<f32, ${WG}>;
var<workgroup> Ms: array<f32, ${BQ}>;
var<workgroup> Ls: array<f32, ${BQ}>;
var<workgroup> Al: array<f32, ${BQ}>;
${mask ? "var<workgroup> kRange: array<atomic<u32>, 2>;\nvar<workgroup> kLo: u32;\nvar<workgroup> kHi: u32;\n" : ""}@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let q0g = wid.x * ${BQ}u; let h = wid.y; let b = wid.z;
  let bh = b * P.H + h;
  let qBase = P.oq / 4u + bh * P.Lq * ${D4}u;
  let kBase = P.ok / 4u + bh * P.Lk * ${D4}u;
  let vBase = P.ov / 4u + bh * P.Lk * ${D4}u;
  let oBase = bh * P.Lq * ${D}u;
${mask ? "  let mBase = P.om + b * P.msb + h * P.msh;\n" : ""}  let sr = (lid / 8u) * 2u; let sc = (lid % 8u) * 2u;
  let rr = lid / 4u; let lane = lid % 4u;
  let orow = (lid / 8u) * 2u; let ocol = lid % 8u;
  for (var t = lid; t < ${BQ * D4}u; t += ${WG}u) {
    var x = vec4<f32>(0.0);
    if (q0g + t / ${D4}u < P.Lq) { x = ${f4(q, "Q[qBase + q0g * " + D4 + "u + t]")} * P.scale; }
    Qs[t] = x;
  }
  if (lid < ${BQ}u) { Ms[lid] = NEG; Ls[lid] = 0.0; }
${decl}
${mask ? `  // Key-tile range with any visible key for this query block (sliding-window
  // and padding masks): fully masked tiles contribute nothing, so skip them.
  if (lid == 0u) { atomicStore(&kRange[0], 0xffffffffu); atomicStore(&kRange[1], 0u); }
  workgroupBarrier();
  {
    var lo = 0xffffffffu; var hi = 0u;
    let nq = min(${BQ}u, P.Lq - q0g);
    for (var j = lid; j < P.Lk; j += ${WG}u) {
      var vis = false;
      for (var qi = 0u; qi < nq; qi++) { vis = vis || Mk[mBase + (q0g + qi) * P.msq + j * P.msk] != 0u; if (P.msq == 0u) { break; } }
      if (vis) { lo = min(lo, j); hi = max(hi, j + 1u); }
    }
    if (hi > 0u) { atomicMin(&kRange[0], lo); atomicMax(&kRange[1], hi); }
  }
  workgroupBarrier();
  if (lid == 0u) { kLo = (atomicLoad(&kRange[0]) / ${BKV}u) * ${BKV}u; kHi = atomicLoad(&kRange[1]); }
  let ktLo = workgroupUniformLoad(&kLo);
  let ktHi = workgroupUniformLoad(&kHi);
  for (var kt = ktLo; kt < ktHi; kt += ${BKV}u) {` : `  for (var kt = 0u; kt < P.Lk; kt += ${BKV}u) {`}
    for (var t = lid; t < ${BKV * D4}u; t += ${WG}u) {
      let kj = t / ${D4}u; let d = t % ${D4}u;
      var kx = vec4<f32>(0.0); var vx = vec4<f32>(0.0);
      if (kt + kj < P.Lk) {
        kx = ${f4(k, `Kt[kBase + kt * ${D4}u + t]`)};
        vx = ${f4(v, `V[vBase + kt * ${D4}u + t]`)};
      }
      Ks[kj * ${KP}u + d] = kx; Vs[t] = vx;
    }
    workgroupBarrier();
${score}    workgroupBarrier();
    // online softmax: 4 lanes per row, ${BKV / 4} keys per lane
    let sb = rr * ${BKV + 1}u + lane * ${BKV / 4}u;
    var lm = NEG;
    for (var j = 0u; j < ${BKV / 4}u; j++) { lm = max(lm, S[sb + j]); }
    red[lid] = lm;
    workgroupBarrier();
    let mprev = Ms[rr];
    let mnew = max(mprev, max(max(red[rr * 4u], red[rr * 4u + 1u]), max(red[rr * 4u + 2u], red[rr * 4u + 3u])));
    let live = mnew > -1.0e38;
    var ls = 0.0;
    for (var j = 0u; j < ${BKV / 4}u; j++) {
      let s = S[sb + j];
      var p = 0.0;
      if (live && s > -1.0e38) { p = exp(s - mnew); }
      S[sb + j] = p; ls += p;
    }
    workgroupBarrier();
    red[lid] = ls;
    workgroupBarrier();
    if (lane == 0u) {
      let alpha = select(1.0, exp(mprev - mnew), live);
      Ls[rr] = Ls[rr] * alpha + red[lid] + red[lid + 1u] + red[lid + 2u] + red[lid + 3u];
      Ms[rr] = mnew; Al[rr] = alpha;
    }
    workgroupBarrier();
${accum}    workgroupBarrier();
  }
${write}}`;
  const params: ParamSpec = [
    ["H", "u32"], ["Lq", "u32"], ["Lk", "u32"], ["oq", "u32"], ["ok", "u32"], ["ov", "u32"], ["scale", "f32"],
    ...(mask ? ([["om", "u32"], ["msb", "u32"], ["msh", "u32"], ["msq", "u32"], ["msk", "u32"]] as const) : []),
  ];
  return {
    key: `sdpafast:${kindKey(q)}:${kindKey(k)}:${kindKey(v)}:${mask ? kindKey(mask) : "-"}:${kindKey(out)}:D${D}`,
    bindings,
    params,
    body,
    f16: needsF16(q, k, v, out),
  };
}

// ---------------------------------------------------------------------------
// RoPE (split-half), with f64-precomputed cos/sin tables [L, D/2] (f32).

export function ropeKernel(x: Kind, out: Kind): KernelSource {
  const WG = 256;
  const X = (e: string) => ld(x, `X[P.off + ${e}]`, "f32");
  const body = `const WG = ${WG}u;\n${HELPERS}\n${ENTRY(WG)} {\n  let lid = lid3;${FLAT_IDX}
  // One thread per rotation pair (d, d + D/2): P.n is the number of pairs.
  if (i >= P.n) { return; }
  let half = P.D / 2u;
  let row = i / half;
  let j = i % half;
  let l = row % P.L;
  let c = CS[l * half + j];
  let s = CS[P.L * half + l * half + j];
  let o = row * P.D + j;
  let x0 = ${X("o")};
  let x1 = ${X("o + half")};
  // Explicit fma: the contraction Metal chose for the 0.2.0 one-element-per-
  // thread kernel; keeps f16 results bit-identical to it (and MLX parity).
  outp[o] = ${st(out, "fma(-x1, s, x0 * c)", "f32")};
  outp[o + half] = ${st(out, "fma(x0, s, x1 * c)", "f32")};
}`;
  return {
    key: `rope:${kindKey(x)}:${kindKey(out)}`,
    bindings: [
      { name: "X", elem: x.st, access: "read" },
      { name: "CS", elem: "f32", access: "read" },
      { name: "outp", elem: out.st, access: "read_write" },
    ],
    params: [["n", "u32"], ["D", "u32"], ["L", "u32"], ["off", "u32"]],
    body,
    f16: needsF16(x, out),
  };
}

// ---------------------------------------------------------------------------
// Gathers.

/** out[r, d] = table[toff + ids[ioff + (r / M)*istride... ]] — see backend for the two modes. */
export function gatherKernel(t: Kind, out: Kind, mode: "embedding" | "rows"): KernelSource {
  const WG = 256;
  const rowExpr =
    mode === "embedding"
      ? "let src = u32(clamp(ids[P.oi + r], 0, i32(P.V) - 1));"
      : "let bb = r / P.M; let src = bb * P.V + u32(clamp(ids[P.oi + r], 0, i32(P.V) - 1));";
  const body = `const WG = ${WG}u;\n${HELPERS}\n${ENTRY(WG)} {\n  let lid = lid3;${FLAT_IDX}
  if (i >= P.n) { return; }
  let r = i / P.D; let d = i % P.D;
  ${rowExpr}
  outp[i] = ${st(out, ld(t, "T[P.ot + src * P.D + d]", t.st === "i32" ? "i32" : "f32"), t.st === "i32" ? "i32" : "f32")};
}`;
  return {
    key: `gather:${mode}:${kindKey(t)}:${kindKey(out)}`,
    bindings: [
      { name: "T", elem: t.st, access: "read" },
      { name: "ids", elem: "i32", access: "read" },
      { name: "outp", elem: out.st, access: "read_write" },
    ],
    params: [["n", "u32"], ["D", "u32"], ["V", "u32"], ["M", "u32"], ["ot", "u32"], ["oi", "u32"]],
    body,
    f16: needsF16(t, out),
  };
}

// ---------------------------------------------------------------------------
// Fused GEGLU and masked mean pooling.

export function gegluKernel(x: Kind, out: Kind): KernelSource {
  const WG = 256;
  const body = `const WG = ${WG}u;\n${HELPERS}\n${ENTRY(WG)} {\n  let lid = lid3;${FLAT_IDX}
  if (i >= P.n) { return; }
  let r = i / P.F; let j = i % P.F;
  let base = P.off + r * 2u * P.F;
  let v = ${ld(x, "X[base + j]", "f32")};
  let g = ${ld(x, "X[base + P.F + j]", "f32")};
  outp[i] = ${st(out, "gelu_(v) * g", "f32")};
}`;
  return {
    key: `geglu:${kindKey(x)}:${kindKey(out)}`,
    bindings: [
      { name: "X", elem: x.st, access: "read" },
      { name: "outp", elem: out.st, access: "read_write" },
    ],
    params: [["n", "u32"], ["F", "u32"], ["off", "u32"]],
    body,
    f16: needsF16(x, out),
  };
}

export function meanPoolKernel(x: Kind, m: Kind): KernelSource {
  const WG = 256;
  const body = `const WG = ${WG}u;\n${ENTRY(WG)} {\n  let lid = lid3;${FLAT_IDX}
  if (i >= P.n) { return; }
  let b = i / P.D; let d = i % P.D;
  var s = 0.0; var c = 0.0;
  for (var l = 0u; l < P.L; l++) {
    let w = ${ld(m, "Mk[P.om + b * P.L + l]", "f32")};
    if (w != 0.0) { s += ${ld(x, "X[P.ox + (b * P.L + l) * P.D + d]", "f32")}; c += 1.0; }
  }
  outp[i] = s / max(c, 1.0);
}`;
  return {
    key: `meanpool:${kindKey(x)}:${kindKey(m)}`,
    bindings: [
      { name: "X", elem: x.st, access: "read" },
      { name: "Mk", elem: m.st, access: "read" },
      { name: "outp", elem: "f32", access: "read_write" },
    ],
    params: [["n", "u32"], ["D", "u32"], ["L", "u32"], ["ox", "u32"], ["om", "u32"]],
    body,
    f16: needsF16(x),
  };
}

// ---------------------------------------------------------------------------
// Sort along the last axis: workgroup bitonic sort in shared memory
// (row length ≤ 4096), else a per-row insertion sort fallback.

export function sortKernel(x: Kind, out: Kind, NP: number): KernelSource {
  const WG = Math.min(256, NP / 2);
  const c: CType = x.st === "i32" ? "i32" : "f32";
  const big = c === "i32" ? "2147483647" : "3.4028234e38";
  const body = `${out.bf16 ? HELPERS : ""}
var<workgroup> sh: array<${c}, ${NP}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let row = wid.x + wid.y * nwg.x;
  if (row >= P.rows) { return; }
  let base = P.off + row * P.n;
  for (var t = lid; t < ${NP}u; t += ${WG}u) {
    var v: ${c} = ${big};
    if (t < P.n) { v = ${ld(x, "X[base + t]", c)}; }
    sh[t] = v;
  }
  workgroupBarrier();
  for (var k = 2u; k <= ${NP}u; k = k << 1u) {
    for (var j = k >> 1u; j > 0u; j = j >> 1u) {
      for (var t = lid; t < ${NP}u; t += ${WG}u) {
        let p = t ^ j;
        if (p > t) {
          let a = sh[t]; let b = sh[p];
          let up = (t & k) == 0u;
          if ((a > b) == up) { sh[t] = b; sh[p] = a; }
        }
      }
      workgroupBarrier();
    }
  }
  for (var t = lid; t < P.n; t += ${WG}u) { outp[row * P.n + t] = ${st(out, "sh[t]", c)}; }
}`;
  return {
    key: `sort:${kindKey(x)}:${kindKey(out)}:${NP}`,
    bindings: [
      { name: "X", elem: x.st, access: "read" },
      { name: "outp", elem: out.st, access: "read_write" },
    ],
    params: [["rows", "u32"], ["n", "u32"], ["off", "u32"]],
    body,
    f16: needsF16(x, out),
  };
}

/** Fallback: one thread per row, insertion sort in the output buffer (already a copy of the input). */
export function sortSlowKernel(out: Kind): KernelSource {
  const WG = 64;
  const c: CType = out.st === "i32" ? "i32" : "f32";
  const body = `const WG = ${WG}u;\n${ENTRY(WG)} {\n  let lid = lid3;${FLAT_IDX}
  if (i >= P.rows) { return; }
  let base = i * P.n;
  for (var a = 1u; a < P.n; a++) {
    let v = outp[base + a];
    var b = a;
    loop {
      if (b == 0u) { break; }
      let u = outp[base + b - 1u];
      if (${ld(out, "u", c)} <= ${ld(out, "v", c)}) { break; }
      outp[base + b] = u;
      b--;
    }
    outp[base + b] = v;
  }
}`;
  return {
    key: `sortslow:${kindKey(out)}`,
    bindings: [{ name: "outp", elem: out.st, access: "read_write" }],
    params: [["rows", "u32"], ["n", "u32"]],
    body,
    f16: needsF16(out),
  };
}

// ---------------------------------------------------------------------------
// Linear through subgroup matrices (Dawn `chromium-experimental-subgroup-matrix`,
// Metal simdgroup_matrix on Apple). f32 8×8×8 fragments: operands are converted
// to f32 while staging tiles in workgroup memory, so accumulation stays f32
// (like MLX's f16 GEMM). Subgroup size must be 32.

export interface SgGemmConfig {
  /** Workgroup tile (multiples of 8·SM, 8·SN). */
  BM: number;
  BN: number;
  BK: number;
  /** Subgroups per workgroup along M and N. */
  WM: number;
  WN: number;
  /** Double-buffer the staged K panels (one barrier per panel instead of two; slower on M2). Default false. */
  db?: boolean;
  /**
   * Split-K: when M ≤ maxM and the grid has at most maxGroups workgroups
   * (both default ∞), split K into S chunks written as f32 partials and
   * summed in a fixed order (plus bias, rounded once) by a second kernel.
   * First match wins. Raises the workgroup count of small grids.
   */
  splitK?: { maxM?: number; maxGroups?: number; S: number }[];
  /** Epilogue: "frag" (default; one 8×8 fragment at a time, small scratch) or "block" (whole block, one barrier). */
  epi?: "frag" | "block";
  /** Row padding (floats) of the staged panels (default 4). */
  pad?: number;
  /** f16 operands: load 8 halves per thread as vec4<u32> (needs K and offsets % 8 == 0). Default true. */
  wide?: boolean;
}

/**
 * Subgroup-matrix Linear (transB, K % 4 == 0): each subgroup owns a
 * (BM/WM)×(BN/WN) block of 8×8 f32 result fragments. K panels of BK are
 * converted to f32 and staged in workgroup memory (Dawn only offers f16
 * fragments with f16 accumulation, which would break parity); the next
 * panel is fetched into registers while the current one is multiplied and,
 * with `db`, stored into the other half of a double buffer, so each panel
 * costs one barrier. The epilogue spills each subgroup's whole block to
 * workgroup memory once and stores it row-contiguously (bias added,
 * rounded once). With `split` (split-K), workgroup z covers K range
 * [z·P.kc, (z+1)·P.kc) and writes f32 partials at z·M·N (no bias).
 */
export function gemmSgKernel(a: Kind, b: Kind, bias: Kind | null, out: Kind, cfg: SgGemmConfig, split = false, wide = false, quant: QuantSpec | null = null): KernelSource {
  const { BM, BN, BK, WM, WN } = cfg;
  const db = cfg.db ?? false;
  const epi = cfg.epi ?? "frag";
  const SG = 32, WG = WM * WN * SG;
  const SBM = BM / WM, SBN = BN / WN; // per-subgroup block
  const FM = SBM / 8, FN = SBN / 8; // fragments per subgroup
  if (FM % 1 || FN % 1 || BK % 8) throw new Error("gemmSg: bad tile config");
  if (split && bias) throw new Error("gemmSg: split-K partials take no bias");
  if (wide && (a.st !== "f16" || b.st !== "f16" || quant)) throw new Error("gemmSg: wide loads need f16 operands");
  const BKP = BK + (cfg.pad ?? 4); // padded row stride (floats) of the staged tiles
  // V elements per global load: vec4<f16>/<f32> (4), or 8 f16 as vec4<u32> (wide).
  const V = wide ? 8 : 4;
  const elem = (k: Kind) => (wide ? "vec4<u32>" : `vec4<${k.st}>`);
  const bindings: BindingSpec[] = [
    { name: "A", elem: elem(a), access: "read" },
    ...(quant ? quantBindings(quant) : [{ name: "B", elem: elem(b), access: "read" as const }]),
  ];
  if (bias) bindings.push({ name: "bias", elem: bias.st, access: "read" });
  bindings.push({ name: "C", elem: out.st, access: "read_write" });
  const f4 = (k: Kind, e: string) => (k.st === "f32" ? e : `vec4<f32>(${e})`);
  const KV = BK / V; // loads per staged row
  const TILE = (BM + BN) * BKP; // one A+B panel
  const BOFF = BM * BKP;
  const NA = Math.ceil((BM * KV) / WG), NB = Math.ceil((BN * KV) / WG);
  // guard for the last partial round when rows·KV isn't a multiple of WG
  const guard = (rows: number, v: number) => ((v + 1) * WG > rows * KV ? `if (lid + ${v * WG}u < ${rows * KV}u) ` : "");
  const regs = (p: string, n: number) =>
    Array.from({ length: n }, (_, v) => `  var ${p}${v} = vec4<f32>(0.0);\n${wide ? `  var ${p}${v}h = vec4<f32>(0.0);\n` : ""}`).join("");
  const load = (p: string, v: number, name: string, kind: Kind, idx: string, gr: string, kk: string) =>
    quant && name === "B"
      ? `${p}${v} = qw4(${gr}, (${kk}) * 4u);`
      : wide
      ? `let u = ${name}[${idx}]; ${p}${v} = vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y)); ${p}${v}h = vec4<f32>(unpack2x16float(u.z), unpack2x16float(u.w));`
      : `${p}${v} = ${f4(kind, `${name}[${idx}]`)};`;
  const zero = (p: string, v: number) => `${p}${v} = vec4<f32>(0.0);${wide ? ` ${p}${v}h = vec4<f32>(0.0);` : ""}`;
  const fetch = (p: string, n: number, name: string, kind: Kind, base: string, lim: string, gbase: string, rows: number) =>
    Array.from({ length: n }, (_, v) => `    ${guard(rows, v)}{ let t = lid + ${v * WG}u; let gr = ${base} + t / ${KV}u; let kk = k0n / ${V}u + t % ${KV}u;
      if (gr < ${lim} && kk < kEndV) { ${load(p, v, name, kind, `${gbase} + gr * KV + kk`, "gr", "kk")} } else { ${zero(p, v)} } }\n`).join("");
  const stash = (p: string, n: number, S: string, rows: number) =>
    Array.from({ length: n }, (_, v) => `    ${guard(rows, v)}{ let t = lid + ${v * WG}u; let o = ${S} + (t / ${KV}u) * ${BKP}u + (t % ${KV}u) * ${V}u;
      Sh[o] = ${p}${v}.x; Sh[o + 1u] = ${p}${v}.y; Sh[o + 2u] = ${p}${v}.z; Sh[o + 3u] = ${p}${v}.w;${
        wide ? `\n      Sh[o + 4u] = ${p}${v}h.x; Sh[o + 5u] = ${p}${v}h.y; Sh[o + 6u] = ${p}${v}h.z; Sh[o + 7u] = ${p}${v}h.w;` : ""
      } }\n`).join("");
  let decl = "";
  for (let i = 0; i < FM; i++) for (let j = 0; j < FN; j++) decl += `  var c${i}_${j}: subgroup_matrix_result<f32, 8, 8>;\n`;
  const mma = (buf: string) => {
    let m = "";
    for (let i = 0; i < FM; i++)
      m += `      let a${i} = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>, row_major>(&Sh, ${buf} + (sm + ${i * 8}u) * ${BKP}u + kk, ${BKP}u);\n`;
    for (let j = 0; j < FN; j++)
      m += `      let b${j} = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>, col_major>(&Sh, ${buf} + ${BOFF}u + (sn + ${j * 8}u) * ${BKP}u + kk, ${BKP}u);\n`;
    for (let i = 0; i < FM; i++) for (let j = 0; j < FN; j++) m += `      c${i}_${j} = subgroupMatrixMultiplyAccumulate(a${i}, b${j}, c${i}_${j});\n`;
    return Array.from({ length: BK / 8 }, (_, q) => `    {\n      let kk = ${q * 8}u;\n${m}    }\n`).join("");
  };
  const bv = bias ? ` + ${ld(bias, "bias[P.obias + col]", "f32")}` : "";
  const cIdx = split ? "wid.z * P.M * P.N + row * P.N + col" : "row * P.N + col";
  const PANELS = db ? 2 * TILE : TILE;
  let SH: number, epilogue = "";
  if (epi === "block") {
    // Each subgroup spills its whole SBM×SBN block (row-major) over the
    // panels, then its lanes store it row-contiguously: one barrier.
    SH = Math.max(PANELS, (WG / SG) * SBM * SBN);
    for (let i = 0; i < FM; i++)
      for (let j = 0; j < FN; j++) epilogue += `  subgroupMatrixStore<row_major>(&Sh, sg * ${SBM * SBN}u + ${i * 8 * SBN + j * 8}u, c${i}_${j}, ${SBN}u);\n`;
    epilogue += `  workgroupBarrier();
  for (var e = lane; e < ${SBM * SBN}u; e += 32u) {
    let row = r0 + sm + e / ${SBN}u; let col = c0 + sn + e % ${SBN}u;
    if (row < P.M && col < P.N) { C[${cIdx}] = ${st(out, `Sh[sg * ${SBM * SBN}u + e]${bv}`, "f32")}; }
  }\n`;
  } else {
    // One 8×8 fragment at a time through a private 64-float scratch per
    // subgroup (keeps workgroup memory, hence occupancy, small).
    SH = PANELS + (WG / SG) * 64;
    for (let i = 0; i < FM; i++)
      for (let j = 0; j < FN; j++)
        epilogue += `  subgroupMatrixStore<row_major>(&Sh, scr, c${i}_${j}, 8u);
  workgroupBarrier();
  for (var e = lane; e < 64u; e += 32u) {
    let row = r0 + sm + ${i * 8}u + e / 8u; let col = c0 + sn + ${j * 8}u + e % 8u;
    if (row < P.M && col < P.N) { C[${cIdx}] = ${st(out, `Sh[scr + e]${bv}`, "f32")}; }
  }
  workgroupBarrier();\n`;
  }
  const loop = db
    ? `  {
    let k0n = kBeg;
${fetch("pa", NA, "A", a, "r0", "P.M", "aBase", BM)}${fetch("pb", NB, "B", b, "c0", "P.N", "bBase", BN)}
${stash("pa", NA, "0u", BM)}${stash("pb", NB, `${BOFF}u`, BN)}  }
  workgroupBarrier();
  var cur = 0u;
  for (var k0 = kBeg; k0 < kEnd; k0 += ${BK}u) {
    let k0n = k0 + ${BK}u;
    let more = k0n < kEnd;
    if (more) {
${fetch("pa", NA, "A", a, "r0", "P.M", "aBase", BM)}${fetch("pb", NB, "B", b, "c0", "P.N", "bBase", BN)}    }
${mma("cur")}
    if (more) {
      let nxt = ${TILE}u - cur;
${stash("pa", NA, "nxt", BM)}${stash("pb", NB, `nxt + ${BOFF}u`, BN)}    }
    cur = ${TILE}u - cur;
    workgroupBarrier();
  }
`
    : `  {
    let k0n = kBeg;
${fetch("pa", NA, "A", a, "r0", "P.M", "aBase", BM)}${fetch("pb", NB, "B", b, "c0", "P.N", "bBase", BN)}  }
  for (var k0 = kBeg; k0 < kEnd; k0 += ${BK}u) {
${stash("pa", NA, "0u", BM)}${stash("pb", NB, `${BOFF}u`, BN)}    workgroupBarrier();
    let k0n = k0 + ${BK}u;
    if (k0n < kEnd) {
${fetch("pa", NA, "A", a, "r0", "P.M", "aBase", BM)}${fetch("pb", NB, "B", b, "c0", "P.N", "bBase", BN)}    }
${mma("0u")}
    workgroupBarrier();
  }
`;
  const body = `${out.bf16 ? HELPERS : ""}${quant ? quantHelper(quant) : ""}
// Panels: A at 0, B at ${BOFF}${db ? `, second buffer at ${TILE}` : ""}; epilogue scratch ${epi === "block" ? "reuses them" : `at ${PANELS}`}.
var<workgroup> Sh: array<f32, ${SH}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let KV = P.K / ${V}u;
  let kBeg = wid.z * P.kc; let kEnd = min(P.K, kBeg + P.kc); let kEndV = kEnd / ${V}u;
  let r0 = wid.y * ${BM}u; let c0 = wid.x * ${BN}u;
  let aBase = P.oa / ${V}u; let bBase = P.ob / ${V}u;
  let sg = lid / ${SG}u;
  let sm = (sg / ${WN}u) * ${SBM}u; let sn = (sg % ${WN}u) * ${SBN}u;
  let lane = lid % ${SG}u; let scr = ${PANELS}u + sg * 64u;
${decl}
${regs("pa", NA)}${regs("pb", NB)}${loop}
${epilogue}}`;
  return {
    key: `gemmsg:${kindKey(a)}:${quant ? quantKey(quant) : kindKey(b)}:${bias ? kindKey(bias) : "-"}:${kindKey(out)}:${BM}x${BN}x${BK}/${WM}x${WN}:${db ? "db" : "sb"}:${epi}:${BKP}:${V}${split ? ":split" : ""}`,
    bindings,
    params: [["M", "u32"], ["N", "u32"], ["K", "u32"], ["kc", "u32"], ["oa", "u32"], ["ob", "u32"], ["obias", "u32"]],
    body,
    f16: needsF16(a, out, ...(quant ? [quant.scale, ...(quant.r16 ? [{ st: "f16" as const }] : [])] : [b]), ...(bias ? [bias] : [])),
    enables: ["chromium_experimental_subgroup_matrix"],
    // Offsets differ per subgroup (derived from local_invocation_index), which is fine:
    // each subgroup executes the matrix ops in subgroup-uniform control flow.
    directives: ["diagnostic(off, chromium.subgroup_matrix_uniformity)"],
  };
}

/**
 * Split-K reduction: C[i] = Σ_z Pt[z·n + i] (+ bias[i % N]), f32 sums in a
 * fixed order, rounded once on store. 4 outputs per thread when N % 4 == 0.
 */
export function splitKReduceKernel(bias: Kind | null, out: Kind, S: number): KernelSource {
  const bindings: BindingSpec[] = [{ name: "Pt", elem: "f32", access: "read" }];
  if (bias) bindings.push({ name: "bias", elem: bias.st, access: "read" });
  bindings.push({ name: "C", elem: out.st, access: "read_write" });
  let sum = "    var acc = Pt[i];\n";
  for (let z = 1; z < S; z++) sum += `    acc += Pt[i + ${z}u * P.n];\n`;
  const bv = bias ? ` + ${ld(bias, "bias[P.obias + i % P.N]", "f32")}` : "";
  const body = `${out.bf16 ? HELPERS : ""}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i < P.n) {
${sum}    C[i] = ${st(out, `acc${bv}`, "f32")};
  }
}`;
  return {
    key: `splitk:${bias ? kindKey(bias) : "-"}:${kindKey(out)}:${S}`,
    bindings,
    params: [["n", "u32"], ["N", "u32"], ["obias", "u32"]],
    body,
    f16: needsF16(out, ...(bias ? [bias] : [])),
  };
}

// ---------------------------------------------------------------------------
// General numerics (optional contract ops).

/**
 * erf, the f32 lowering of math-plus tensor-core's canonical erf (issue #122;
 * derived from @johnhenry/backend-cpu's double-precision erf): Maclaurin
 * series (10 terms) for |x| < 1, erfc by the even contraction of Laplace's
 * continued fraction (depth 28, exp(-z²) split to keep z² exact) above,
 * flushed to 0 beyond 10.1. Truncation error < 2^-24 relative
 * (math-plus ERF_F32_PARAMS); ≈1e-7 absolute overall in f32.
 */
export const ERF_HELPERS = /* wgsl */ `
fn erf_mp_series(x: f32) -> f32 {
  let x2 = x * x;
  var term = x;
  var sum = x;
  for (var n = 1; n <= 10; n = n + 1) {
    let nf = f32(n);
    term = term * (-x2 / nf);
    sum = sum + term / (2.0 * nf + 1.0);
  }
  return 1.1283791670955126 * sum;
}
fn erf_mp_erfc(z: f32) -> f32 {
  if (z > 10.1) { return 0.0; }
  let t = 2.0 * z * z + 1.0;
  var f = 0.0;
  for (var n = 28; n >= 1; n = n - 1) {
    let nf = f32(n);
    f = ((2.0 * nf - 1.0) * (2.0 * nf)) / (t + 4.0 * nf - f);
  }
  let s = round(z * 64.0) / 64.0;
  let e = exp(-s * s) * exp(-(z - s) * (z + s));
  return (e * 0.5641895835477563 * 2.0 * z) / (t - f);
}
fn erf_mp(x: f32) -> f32 {
  let ax = abs(x);
  if (ax < 1.0) { return erf_mp_series(x); }
  return sign(x) * (1.0 - erf_mp_erfc(ax));
}`;

/**
 * pow with C/MLX semantics: x^0 = 1, 0^y = 0 (y > 0) or inf (y < 0), a
 * negative base is defined for integral exponents (sign from parity), NaN
 * otherwise. WGSL's builtin pow is undefined for x < 0.
 */
export const POW_HELPERS = /* wgsl */ `
fn pow_(a: f32, b: f32) -> f32 {
  if (b == 0.0) { return 1.0; }
  var infBits = 0x7f800000u;
  var nanBits = 0x7fc00000u;
  if (a == 0.0) { return select(0.0, bitcast<f32>(infBits), b < 0.0); }
  let r = exp2(b * log2(abs(a)));
  if (a > 0.0) { return r; }
  if (b != floor(b)) { return bitcast<f32>(nanBits); }
  let half = b * 0.5;
  return select(r, -r, half != floor(half));
}`;

/** Index (i32) of the first max/min along R of [outer, R, inner]: one thread per output. */
export function argReduceKernel(op: "argmax" | "argmin", inp: Kind): KernelSource {
  const WG = 256;
  const c: CType = inp.st === "i32" || inp.st === "u32" ? "i32" : "f32";
  const better = op === "argmax" ? "v > best" : "v < best";
  const body = `const WG = ${WG}u;\n${ENTRY(WG)} {\n  let lid = lid3;${FLAT_IDX}
  if (i >= P.n) { return; }
  let o = i / P.inner; let inn = i % P.inner;
  let base = P.off + o * P.R * P.inner + inn;
  var best = ${ld(inp, "inp[base]", c)};
  var bi = 0u;
  for (var r = 1u; r < P.R; r++) {
    let v = ${ld(inp, "inp[base + r * P.inner]", c)};
    if (${better}) { best = v; bi = r; }
  }
  outp[i] = i32(bi);
}`;
  return {
    key: `${op}:${kindKey(inp)}`,
    bindings: [
      { name: "inp", elem: inp.st, access: "read" },
      { name: "outp", elem: "i32", access: "read_write" },
    ],
    params: [["n", "u32"], ["R", "u32"], ["inner", "u32"], ["off", "u32"]],
    body,
    f16: needsF16(inp),
  };
}

/** Inclusive prefix sum along R of [outer, R, inner]: one thread per (outer, inner) lane, f32/i32 accumulator. */
export function cumsumKernel(inp: Kind, out: Kind): KernelSource {
  const WG = 256;
  const c: CType = out.st === "i32" ? "i32" : "f32";
  const body = `const WG = ${WG}u;\n${out.bf16 ? HELPERS : ""}\n${ENTRY(WG)} {\n  let lid = lid3;${FLAT_IDX}
  if (i >= P.n) { return; }
  let o = i / P.inner; let inn = i % P.inner;
  let base = o * P.R * P.inner + inn;
  var acc = ${c}(0);
  for (var r = 0u; r < P.R; r++) {
    acc = acc + ${ld(inp, "inp[P.off + base + r * P.inner]", c)};
    outp[base + r * P.inner] = ${st(out, "acc", c)};
  }
}`;
  return {
    key: `cumsum:${kindKey(inp)}:${kindKey(out)}`,
    bindings: [
      { name: "inp", elem: inp.st, access: "read" },
      { name: "outp", elem: out.st, access: "read_write" },
    ],
    params: [["n", "u32"], ["R", "u32"], ["inner", "u32"], ["off", "u32"]],
    body,
    f16: needsF16(inp, out),
  };
}
