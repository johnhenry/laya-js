/**
 * A small seeded PRNG (mulberry32), used only by the scripted bot actor's
 * tie-breaks. Deterministic within this JS implementation for a given seed.
 */
export class CheckersRng {
  #state: number;

  constructor(seed: number) {
    this.#state = seed >>> 0;
  }

  next(): number {
    this.#state = (this.#state + 0x6d2b79f5) | 0;
    let t = this.#state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Integer in [min, max], inclusive. */
  int(min: number, max: number): number {
    if (max < min) throw new RangeError("CheckersRng.int: max must be >= min");
    return min + Math.floor(this.next() * (max - min + 1));
  }
}
