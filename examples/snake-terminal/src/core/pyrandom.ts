/**
 * Python-compatible `random.Random(intSeed)` subset: MT19937 seeded with
 * CPython's `init_by_array`, `getrandbits(k)` and `choice(seq)` via
 * `_randbelow_with_getrandbits`. Lets a JS game reproduce the Python demo's
 * food positions for the same seed, so recorded runs can be replayed.
 */
const N = 624;
const M = 397;

export class PyRandom {
  #mt = new Uint32Array(N);
  #mti = N + 1;

  constructor(seed: number | bigint) {
    this.seed(seed);
  }

  seed(seed: number | bigint): void {
    let n = BigInt(seed);
    if (n < 0n) n = -n;
    const key: number[] = [];
    do {
      key.push(Number(n & 0xffffffffn));
      n >>= 32n;
    } while (n > 0n);
    this.#initByArray(key);
  }

  #initGenrand(s: number): void {
    const mt = this.#mt;
    mt[0] = s >>> 0;
    for (let i = 1; i < N; i++) {
      const prev = mt[i - 1]! ^ (mt[i - 1]! >>> 30);
      mt[i] = (Math.imul(1812433253, prev) + i) >>> 0;
    }
    this.#mti = N;
  }

  #initByArray(key: number[]): void {
    const mt = this.#mt;
    this.#initGenrand(19650218);
    let i = 1;
    let j = 0;
    for (let k = Math.max(N, key.length); k > 0; k--) {
      const prev = mt[i - 1]! ^ (mt[i - 1]! >>> 30);
      mt[i] = ((mt[i]! ^ Math.imul(prev, 1664525)) + key[j]! + j) >>> 0;
      i++;
      j++;
      if (i >= N) {
        mt[0] = mt[N - 1]!;
        i = 1;
      }
      if (j >= key.length) j = 0;
    }
    for (let k = N - 1; k > 0; k--) {
      const prev = mt[i - 1]! ^ (mt[i - 1]! >>> 30);
      mt[i] = ((mt[i]! ^ Math.imul(prev, 1566083941)) - i) >>> 0;
      i++;
      if (i >= N) {
        mt[0] = mt[N - 1]!;
        i = 1;
      }
    }
    mt[0] = 0x80000000;
  }

  /** genrand_uint32 */
  uint32(): number {
    const mt = this.#mt;
    if (this.#mti >= N) {
      let kk = 0;
      for (; kk < N - M; kk++) {
        const y = (mt[kk]! & 0x80000000) | (mt[kk + 1]! & 0x7fffffff);
        mt[kk] = mt[kk + M]! ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      }
      for (; kk < N - 1; kk++) {
        const y = (mt[kk]! & 0x80000000) | (mt[kk + 1]! & 0x7fffffff);
        mt[kk] = mt[kk + (M - N)]! ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      }
      const y = (mt[N - 1]! & 0x80000000) | (mt[0]! & 0x7fffffff);
      mt[N - 1] = mt[M - 1]! ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      this.#mti = 0;
    }
    let y = mt[this.#mti++]!;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** `getrandbits(k)` for 0 < k <= 32. */
  getrandbits(k: number): number {
    if (k <= 0 || k > 32) throw new RangeError("getrandbits: 0 < k <= 32 supported");
    return this.uint32() >>> (32 - k);
  }

  /** `_randbelow_with_getrandbits(n)`. */
  randbelow(n: number): number {
    if (n <= 0) throw new RangeError("randbelow: n must be positive");
    const k = 32 - Math.clz32(n); // n.bit_length()
    let r = this.getrandbits(k);
    while (r >= n) r = this.getrandbits(k);
    return r;
  }

  /** `random.choice(seq)` */
  choice<T>(seq: readonly T[]): T {
    if (seq.length === 0) throw new RangeError("Cannot choose from an empty sequence");
    return seq[this.randbelow(seq.length)]!;
  }

  /** `random.random()` */
  random(): number {
    const a = this.uint32() >>> 5;
    const b = this.uint32() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  }
}
