/**
 * Double-precision erf / erfc, accurate to ~1e-15 relative over the real
 * line (checked against libm in test/erf.test.ts). Far more accurate than
 * the f32 GELU in MLX / PyTorch, so the CPU backend is a clean reference.
 *
 * - |x| < 2.5: Maclaurin series (at most ~45 terms, <3 digits of cancellation).
 * - |x| >= 2.5: erfc via the even contraction of the Laplace continued fraction,
 *   evaluated backward with a fixed depth, so there is no cancellation in
 *   the tail (erfc(x) ~ exp(-x²)).
 */

const TWO_OVER_SQRT_PI = 1.1283791670955126;
const INV_SQRT_PI = 0.5641895835477563;

function erfSeries(x: number): number {
  const x2 = x * x;
  let term = x;
  let sum = x;
  for (let n = 1; n < 80; n++) {
    term *= -x2 / n;
    const c = term / (2 * n + 1);
    sum += c;
    if (Math.abs(c) <= 1e-17 * Math.abs(sum)) break;
  }
  return TWO_OVER_SQRT_PI * sum;
}

/** erfc(z) for z >= 2.5 (continued fraction). */
function erfcTail(z: number): number {
  if (z > 27.3) return 0;
  const z2 = z * z;
  const t = 2 * z2 + 1;
  // erfc(z) = exp(-z²)/√π · 2z / (t - 1·2/(t+4 - 3·4/(t+8 - 5·6/(t+12 - ...))))
  let f = 0;
  for (let n = 40; n >= 1; n--) {
    f = ((2 * n - 1) * (2 * n)) / (t + 4 * n - f);
  }
  return (Math.exp(-z2) * INV_SQRT_PI * 2 * z) / (t - f);
}

export function erf(x: number): number {
  if (x !== x) return NaN;
  const ax = Math.abs(x);
  if (ax < 2.5) return erfSeries(x);
  const c = erfcTail(ax);
  return x > 0 ? 1 - c : c - 1;
}

export function erfc(x: number): number {
  if (x !== x) return NaN;
  if (x >= 2.5) return erfcTail(x);
  if (x <= -2.5) return 2 - erfcTail(-x);
  return 1 - erfSeries(x);
}

const INV_SQRT2 = 0.7071067811865476;

/** Exact GELU 0.5·x·(1 + erf(x/√2)) = 0.5·x·erfc(-x/√2), without cancellation for x << 0. */
export function geluScalar(x: number): number {
  const z = -x * INV_SQRT2;
  if (z >= 2.5) return 0.5 * x * erfcTail(z);
  if (z <= -2.5) return x - 0.5 * x * erfcTail(-z);
  return 0.5 * x * (1 + erfSeries(-z));
}
