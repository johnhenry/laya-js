/**
 * A small seeded PRNG (mulberry32) used only for pipe-gap placement.
 * Deterministic within this JS implementation for a given seed -- unlike
 * Snake's `PyRandom`, this is NOT a Python-parity port. There is no Python
 * reference implementation of this game to match; the seed exists so runs
 * are reproducible for tests and `--record`/replay, not for cross-language
 * bit-exactness.
 */
export class FlappyRng {
  #state: number;

  constructor(seed: number) {
    this.#state = seed >>> 0;
  }

  /** Next float in [0, 1). */
  next(): number {
    this.#state = (this.#state + 0x6d2b79f5) | 0;
    let t = this.#state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Integer in [min, max], inclusive. */
  int(min: number, max: number): number {
    if (max < min) throw new RangeError("FlappyRng.int: max must be >= min");
    return min + Math.floor(this.next() * (max - min + 1));
  }
}
