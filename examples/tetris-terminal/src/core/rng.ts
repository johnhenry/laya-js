/**
 * A small seeded PRNG (mulberry32) for the 7-bag piece randomizer.
 * Deterministic within this JS implementation for a given seed -- not a
 * Python-parity port, there is no Python reference for this game.
 */
import { PIECE_KINDS, type PieceKind } from "./pieces.ts";

export class TetrisRng {
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
    if (max < min) throw new RangeError("TetrisRng.int: max must be >= min");
    return min + Math.floor(this.next() * (max - min + 1));
  }

  /** A fresh 7-bag: a Fisher-Yates permutation of all 7 piece kinds. */
  bag(): PieceKind[] {
    const bag = [...PIECE_KINDS];
    for (let i = bag.length - 1; i > 0; i--) {
      const j = this.int(0, i);
      [bag[i], bag[j]] = [bag[j]!, bag[i]!];
    }
    return bag;
  }
}
