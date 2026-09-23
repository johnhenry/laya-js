/**
 * Python-identical `json.dumps`, `repr(float)`, `"%g"` and `round()` for JavaScript.
 *
 * JavaScript has one number type; Python has `int` and `float`. `dumps` prints a
 * `number` as an int when `Number.isInteger(x)` holds and as a float otherwise.
 * Wrap a value with {@link pyFloat} to force float formatting (`1` -> `1.0`) and
 * pass a `bigint` for ints beyond 2^53. {@link loads} parses JSON text keeping
 * that distinction, so `dumps(loads(s)) === s` for Python-produced text.
 */

// ---------------------------------------------------------------- float marker

/** A number that must be formatted as a Python float. */
export class PyFloat {
  readonly value: number;
  constructor(value: number) {
    this.value = Number(value);
  }
  valueOf(): number {
    return this.value;
  }
  toJSON(): number {
    return this.value;
  }
}

/** Mark `x` as a Python float, e.g. `dumps(pyFloat(1))` === `"1.0"`. */
export const pyFloat = (x: number): PyFloat => new PyFloat(x);
export const isPyFloat = (x: unknown): x is PyFloat => x instanceof PyFloat;

// ---------------------------------------------------------------- exact decimal helpers

const F64 = new Float64Array(1);
const U64 = new BigUint64Array(F64.buffer);

/** |x| = n / 10^scale exactly, for finite x. */
function exactDecimal(x: number): { neg: boolean; n: bigint; scale: number } {
  F64[0] = x;
  const bits = U64[0]!;
  const neg = bits >> 63n === 1n;
  const e = Number((bits >> 52n) & 0x7ffn);
  let mant = bits & 0xfffffffffffffn;
  let exp = -1074;
  if (e !== 0) {
    mant |= 1n << 52n;
    exp = e - 1075;
  }
  if (exp >= 0) return { neg, n: mant << BigInt(exp), scale: 0 };
  return { neg, n: mant * 5n ** BigInt(-exp), scale: -exp };
}

const pow10 = (n: number): bigint => 10n ** BigInt(n);

/** n / 10^drop rounded half to even. */
function divRoundEven(n: bigint, drop: number): bigint {
  if (drop <= 0) return n * pow10(-drop);
  const d = pow10(drop);
  let q = n / d;
  const r2 = (n % d) * 2n;
  if (r2 > d || (r2 === d && (q & 1n) === 1n)) q += 1n;
  return q;
}

/** Decimal string of `q / 10^scale`. */
function placeDecimal(q: bigint, scale: number): string {
  if (scale <= 0) return (q * pow10(-scale)).toString();
  let s = q.toString();
  if (s.length <= scale) s = "0".repeat(scale - s.length + 1) + s;
  return s.slice(0, s.length - scale) + "." + s.slice(s.length - scale);
}

/**
 * CPython `round(x, ndigits)` for floats: correctly rounded, ties to even on the
 * exact binary value (`round(2.675, 2) == 2.67`, `round(0.125, 2) == 0.12`).
 * Unlike Python, `ndigits` defaults to 0 and a number (never an int) is returned.
 */
export function pyRound(x: number, ndigits = 0): number {
  x = Number(x);
  if (!Number.isInteger(ndigits)) throw new TypeError("ndigits must be an integer");
  if (!Number.isFinite(x) || x === 0) return x;
  if (ndigits > 323) return x; // CPython NDIGITS_MAX shortcut
  if (ndigits < -308) return 0.0 * x;
  const { neg, n, scale } = exactDecimal(x);
  if (scale <= ndigits) return x;
  const q = divRoundEven(n, scale - ndigits);
  const y = Number((neg ? "-" : "") + placeDecimal(q, ndigits));
  if (!Number.isFinite(y)) throw new RangeError("rounded value too large to represent");
  return y;
}

// ---------------------------------------------------------------- float formatting

/** Shortest round-trip digits and exponent: |x| = 0.DIGITS * 10^decpt. */
function shortest(x: number): { digits: string; decpt: number } {
  const s = Math.abs(x).toExponential(); // ECMA-262 shortest round-trip digits, same as repr()
  const e = s.indexOf("e");
  return { digits: s.slice(0, e).replace(".", ""), decpt: Number(s.slice(e + 1)) + 1 };
}

const expSuffix = (e: number) => "e" + (e < 0 ? "-" : "+") + Math.abs(e).toString().padStart(2, "0");
const signOf = (x: number) => (x < 0 || Object.is(x, -0) ? "-" : "");

/** CPython `repr(float)` ('r' format): `1.0`, `1e+16`, `1e-07`, `-0.0`, `nan`, `inf`. */
export function pyFloatRepr(x: number): string {
  x = Number(x);
  if (Number.isNaN(x)) return "nan";
  if (x === Infinity) return "inf";
  if (x === -Infinity) return "-inf";
  const sign = signOf(x);
  if (x === 0) return sign + "0.0";
  const { digits, decpt } = shortest(x);
  if (-4 < decpt && decpt <= 16) {
    if (decpt <= 0) return sign + "0." + "0".repeat(-decpt) + digits;
    if (digits.length <= decpt) return sign + digits + "0".repeat(decpt - digits.length) + ".0";
    return sign + digits.slice(0, decpt) + "." + digits.slice(decpt);
  }
  const mant = digits.length === 1 ? digits : digits[0] + "." + digits.slice(1);
  return sign + mant + expSuffix(decpt - 1);
}

/**
 * CPython `"%.<precision>g" % x` (default precision 6), rounding the exact binary
 * value half to even like C `printf`.
 */
export function pyFormatG(x: number, precision = 6): string {
  x = Number(x);
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  const p = Math.max(1, precision);
  const sign = signOf(x);
  if (x === 0) return sign + "0";
  const { n, scale } = exactDecimal(x);
  const len = n.toString().length;
  let e = len - 1 - scale; // decimal exponent of the leading digit
  let q = divRoundEven(n, len - p); // p significant digits
  if (q.toString().length > p) {
    q /= 10n;
    e += 1;
  }
  const digits = q.toString().replace(/0+$/, "") || "0";
  if (e < -4 || e >= p) {
    const mant = digits.length === 1 ? digits : digits[0] + "." + digits.slice(1);
    return sign + mant + expSuffix(e);
  }
  // fixed notation with trailing zeros stripped
  if (e >= 0) {
    if (digits.length <= e + 1) return sign + digits + "0".repeat(e + 1 - digits.length);
    return sign + digits.slice(0, e + 1) + "." + digits.slice(e + 1);
  }
  return sign + "0." + "0".repeat(-e - 1) + digits;
}

// ---------------------------------------------------------------- dumps

export interface DumpsOptions {
  /** Escape every non-ASCII character as `\uXXXX` (default true, like Python). */
  ensureAscii?: boolean;
  /** `[itemSeparator, keySeparator]`, default `[", ", ": "]`. */
  separators?: readonly [string, string];
  /** Sort object keys by code point (default false). */
  sortKeys?: boolean;
  /** Emit NaN / Infinity tokens (default true, like Python); false throws instead. */
  allowNan?: boolean;
}

const SHORT: Record<string, string> = {
  '"': '\\"',
  "\\": "\\\\",
  "\n": "\\n",
  "\r": "\\r",
  "\t": "\\t",
  "\b": "\\b",
  "\f": "\\f",
};
const hex4 = (c: number) => "\\u" + c.toString(16).padStart(4, "0");

/** Python `json.encoder.py_encode_basestring(_ascii)`. */
export function encodeString(s: string, ensureAscii = true): string {
  let out = '"';
  for (let i = 0; i < s.length; i++) {
    const ch = s[i]!;
    const c = s.charCodeAt(i); // UTF-16 units: astral chars become surrogate pairs, as in Python
    const short = SHORT[ch];
    if (short) out += short;
    else if (c < 0x20 || (ensureAscii && c > 0x7e)) out += hex4(c);
    else out += ch;
  }
  return out + '"';
}

/** Code-point order comparison (Python `str` ordering; JS `<` compares UTF-16 units). */
export function comparePyStr(a: string, b: string): number {
  const ia = a[Symbol.iterator]();
  const ib = b[Symbol.iterator]();
  for (;;) {
    const x = ia.next();
    const y = ib.next();
    if (x.done) return y.done ? 0 : -1;
    if (y.done) return 1;
    const d = x.value.codePointAt(0)! - y.value.codePointAt(0)!;
    if (d) return d;
  }
}

function floatToken(x: number, allowNan: boolean): string {
  if (Number.isFinite(x)) return pyFloatRepr(x);
  if (!allowNan) throw new RangeError("Out of range float values are not JSON compliant: " + pyFloatRepr(x));
  return Number.isNaN(x) ? "NaN" : x > 0 ? "Infinity" : "-Infinity";
}

/** A plain number: int when integral (except -0, which only a float can be), else float. */
function numberToken(x: number, allowNan: boolean): string {
  if (Number.isInteger(x) && !Object.is(x, -0)) return BigInt(x).toString();
  return floatToken(x, allowNan);
}

function keyToken(k: unknown, allowNan: boolean): string {
  if (typeof k === "string") return k;
  if (k instanceof PyFloat) return floatToken(k.value, allowNan);
  if (typeof k === "number") return numberToken(k, allowNan);
  if (typeof k === "bigint") return k.toString();
  if (k === true) return "true";
  if (k === false) return "false";
  if (k === null || k === undefined) return "null";
  throw new TypeError(`keys must be str, int, float, bool or None, not ${typeName(k)}`);
}

function typeName(v: unknown): string {
  if (v === null) return "NoneType";
  if (typeof v === "object") return (v as object).constructor?.name ?? "object";
  return typeof v;
}

/**
 * CPython `json.dumps(value, ensure_ascii=..., separators=..., sort_keys=...)`, byte for byte.
 * Accepts null/undefined (`null`), booleans, numbers, bigints, {@link PyFloat}, strings,
 * arrays, plain objects and `Map`s. Object key order is JavaScript's (integer-like keys
 * first); pass a `Map` when you need Python's insertion order for such keys.
 */
export function dumps(value: unknown, options: DumpsOptions = {}): string {
  const { ensureAscii = true, separators = [", ", ": "], sortKeys = false, allowNan = true } = options;
  const [itemSep, keySep] = separators;
  const stack = new Set<object>();
  const enc = (v: unknown): string => {
    if (v === null || v === undefined) return "null";
    if (v === true) return "true";
    if (v === false) return "false";
    if (typeof v === "string") return encodeString(v, ensureAscii);
    if (typeof v === "number") return numberToken(v, allowNan);
    if (typeof v === "bigint") return v.toString();
    if (v instanceof PyFloat) return floatToken(v.value, allowNan);
    if (typeof v !== "object") throw new TypeError(`Object of type ${typeName(v)} is not JSON serializable`);
    if (stack.has(v)) throw new Error("Circular reference detected");
    stack.add(v);
    try {
      if (Array.isArray(v)) return v.length === 0 ? "[]" : "[" + v.map(enc).join(itemSep) + "]";
      let entries: Array<[unknown, unknown]>;
      if (v instanceof Map) entries = [...v.entries()];
      else {
        const proto = Object.getPrototypeOf(v);
        if (proto !== Object.prototype && proto !== null) {
          throw new TypeError(`Object of type ${typeName(v)} is not JSON serializable`);
        }
        entries = Object.entries(v);
      }
      if (entries.length === 0) return "{}";
      const pairs = entries.map(([k, x]) => [keyToken(k, allowNan), x] as [string, unknown]);
      if (sortKeys) pairs.sort((a, b) => comparePyStr(a[0], b[0]));
      return "{" + pairs.map(([k, x]) => encodeString(k, ensureAscii) + keySep + enc(x)).join(itemSep) + "}";
    } finally {
      stack.delete(v);
    }
  };
  return enc(value);
}

// ---------------------------------------------------------------- loads

export interface LoadsOptions {
  /** Wrap numbers written with `.`/`e`/`E` (and NaN/Infinity) as {@link PyFloat} (default true). */
  wrapFloats?: boolean;
  /** Return integers outside the safe range as `bigint` (default true). */
  bigInts?: boolean;
}

const UNESCAPE: Record<string, string> = { '"': '"', "\\": "\\", "/": "/", b: "\b", f: "\f", n: "\n", r: "\r", t: "\t" };

/**
 * Parse JSON text (plus Python's `NaN`/`Infinity`/`-Infinity` tokens) preserving the
 * int/float distinction: floats become {@link PyFloat}, unsafe ints become `bigint`.
 * With the defaults, `dumps(loads(s)) === s` for any `json.dumps` output whose objects
 * have no integer-like keys after other keys.
 */
export function loads(text: string, options: LoadsOptions = {}): any {
  const { wrapFloats = true, bigInts = true } = options;
  let i = 0;
  const fail = (msg: string): never => {
    throw new SyntaxError(`${msg} at position ${i}`);
  };
  const ws = () => {
    while (i < text.length && " \t\n\r".includes(text[i]!)) i++;
  };
  const flt = (x: number) => (wrapFloats ? new PyFloat(x) : x);
  const lit = (word: string, v: unknown) => {
    if (!text.startsWith(word, i)) fail("Unexpected token");
    i += word.length;
    return v;
  };
  const str = (): string => {
    i++;
    let out = "";
    for (;;) {
      const c = text[i++];
      if (c === undefined) return fail("Unterminated string");
      if (c === '"') return out;
      if (c !== "\\") {
        out += c;
        continue;
      }
      const e = text[i++];
      if (e === "u") {
        const h = text.slice(i, i + 4);
        if (!/^[0-9a-fA-F]{4}$/.test(h)) fail("Bad \\u escape");
        out += String.fromCharCode(parseInt(h, 16));
        i += 4;
      } else if (e !== undefined && e in UNESCAPE) out += UNESCAPE[e];
      else fail("Bad escape");
    }
  };
  const value = (): unknown => {
    ws();
    const c = text[i];
    if (c === "{") {
      i++;
      const obj: Record<string, unknown> = {};
      ws();
      if (text[i] === "}") return i++, obj;
      for (;;) {
        ws();
        if (text[i] !== '"') fail("Expected string key");
        const k = str();
        ws();
        if (text[i++] !== ":") fail("Expected ':'");
        Object.defineProperty(obj, k, { value: value(), enumerable: true, writable: true, configurable: true });
        ws();
        const d = text[i++];
        if (d === "}") return obj;
        if (d !== ",") fail("Expected ',' or '}'");
      }
    }
    if (c === "[") {
      i++;
      const arr: unknown[] = [];
      ws();
      if (text[i] === "]") return i++, arr;
      for (;;) {
        arr.push(value());
        ws();
        const d = text[i++];
        if (d === "]") return arr;
        if (d !== ",") fail("Expected ',' or ']'");
      }
    }
    if (c === '"') return str();
    if (c === "t") return lit("true", true);
    if (c === "f") return lit("false", false);
    if (c === "n") return lit("null", null);
    if (c === "N") return lit("NaN", flt(NaN));
    if (c === "I") return lit("Infinity", flt(Infinity));
    if (text.startsWith("-Infinity", i)) return lit("-Infinity", flt(-Infinity));
    const m = /^-?(?:0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?/.exec(text.slice(i, i + 1100));
    if (!m) return fail("Unexpected token");
    i += m[0].length;
    if (m[1] !== undefined || m[2] !== undefined) return flt(Number(m[0]));
    const n = Number(m[0]) || 0; // an int is never -0
    return bigInts && !Number.isSafeInteger(n) ? BigInt(m[0]) : n;
  };
  const v = value();
  ws();
  if (i !== text.length) fail("Extra data");
  return v;
}
