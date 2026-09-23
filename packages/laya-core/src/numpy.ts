/**
 * Minimal float32/float64 emulation of the numpy expressions laya-mlx uses to turn
 * logits into published probabilities, so results match Python bit for bit.
 *
 * numpy >= 2 (NEP 50): `f32_array / python_float` casts the float to f32 and computes
 * in f32; `f32.sum()` uses pairwise summation with an f32 accumulator.
 */

export const f32 = Math.fround;

/** numpy `@TYPE@_pairwise_sum` (PW_BLOCKSIZE 128), accumulating with `round` after each add. */
function pairwise(a: ArrayLike<number>, lo: number, n: number, round: (x: number) => number): number {
  if (n < 8) {
    let res = round(0);
    for (let i = 0; i < n; i++) res = round(res + a[lo + i]!);
    return res;
  }
  if (n <= 128) {
    const r = new Array<number>(8);
    for (let j = 0; j < 8; j++) r[j] = round(a[lo + j]!);
    let i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (let j = 0; j < 8; j++) r[j] = round(r[j]! + a[lo + i + j]!);
    }
    let res = round(
      round(round(r[0]! + r[1]!) + round(r[2]! + r[3]!)) + round(round(r[4]! + r[5]!) + round(r[6]! + r[7]!)),
    );
    for (; i < n; i++) res = round(res + a[lo + i]!);
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return round(pairwise(a, lo, n2, round) + pairwise(a, lo + n2, n - n2, round));
}

const id = (x: number) => x;

/** `np.add.reduce` over a contiguous f32 vector (identity 0 + pairwise sum). */
export const sumF32 = (a: ArrayLike<number>): number => f32(0 + pairwise(a, 0, a.length, f32));
/** `np.add.reduce` over a contiguous f64 vector. */
export const sumF64 = (a: ArrayLike<number>): number => 0 + pairwise(a, 0, a.length, id);

export const expF32 = (x: number): number => f32(Math.exp(x));
export const logF32 = (x: number): number => f32(Math.log(x));

/** `p = exp(z - z.max()); p /= p.sum()` in float32, `z = logits / scale` (all f32). */
export function softmaxF32(logits: ArrayLike<number>, scale = 1): Float32Array {
  const k = logits.length;
  const s = f32(scale);
  const z = new Float32Array(k);
  for (let i = 0; i < k; i++) z[i] = f32(logits[i]! / s);
  let m = -Infinity;
  for (let i = 0; i < k; i++) if (z[i]! > m) m = z[i]!;
  const p = new Float32Array(k);
  for (let i = 0; i < k; i++) p[i] = expF32(f32(z[i]! - m));
  const total = sumF32(p);
  for (let i = 0; i < k; i++) p[i] = f32(p[i]! / total);
  return p;
}
