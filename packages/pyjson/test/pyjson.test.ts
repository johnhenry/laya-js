import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const test = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import assert from "node:assert/strict";
import { loadJson } from "@johnhenry/laya-fixtures";
import {
  comparePyStr,
  dumps,
  loads,
  PyFloat,
  pyFloat,
  pyFloatRepr,
  pyFormatG,
  pyRound,
  type DumpsOptions,
} from "@johnhenry/pyjson";

interface Table {
  variants: Record<string, { ensure_ascii?: boolean; separators?: [string, string] }>;
  dumps: Array<Record<string, string>>;
  round4: Array<{ x: number; round4: number }>;
  float_repr: Array<{ x: number; repr: string }>;
}
const table = await loadJson<Table>("tables", "pyjson.json");

const opts = (v: Table["variants"][string]): DumpsOptions => ({
  ...(v.ensure_ascii !== undefined ? { ensureAscii: v.ensure_ascii } : {}),
  ...(v.separators ? { separators: v.separators } : {}),
});

test("dumps matches CPython json.dumps for every fixture value and variant", () => {
  assert.ok(table.dumps.length >= 30);
  for (const row of table.dumps) {
    const value = loads(row.value_json!);
    for (const [name, v] of Object.entries(table.variants)) {
      assert.equal(dumps(value, opts(v)), row[name], `${row.repr} / ${name}`);
    }
  }
});

test("loads round-trips Python output exactly", () => {
  for (const row of table.dumps) assert.equal(dumps(loads(row.value_json!)), row.value_json);
  assert.ok(loads("1.0") instanceof PyFloat);
  assert.equal(typeof loads("9007199254740993"), "bigint");
  assert.equal(loads("7"), 7);
  assert.ok(Object.is(loads("-0"), 0));
  assert.throws(() => loads("[1,]"), SyntaxError);
  assert.throws(() => loads("{} x"), SyntaxError);
});

test("pyRound matches CPython round(x, 4)", () => {
  assert.ok(table.round4.length > 10);
  for (const { x, round4 } of table.round4) assert.ok(Object.is(pyRound(x, 4), round4), `round(${x}, 4)`);
});

test("pyFloatRepr matches CPython repr(float)", () => {
  for (const { x, repr } of table.float_repr) assert.equal(pyFloatRepr(x), repr);
  assert.equal(pyFloatRepr(-0), "-0.0");
  assert.equal(pyFloatRepr(NaN), "nan");
  assert.equal(pyFloatRepr(-Infinity), "-inf");
});

test("int/float distinction: numbers print as ints when integral, pyFloat forces a float", () => {
  assert.equal(dumps(1), "1");
  assert.equal(dumps(pyFloat(1)), "1.0");
  assert.equal(dumps(1.5), "1.5");
  assert.equal(dumps(-0), "-0.0");
  assert.equal(dumps(2n ** 64n), "18446744073709551616");
  assert.equal(dumps([pyFloat(1e16), 1e16]), "[1e+16, 10000000000000000]");
});

test("keys: sortKeys uses code point order, Map keeps insertion order, non-str keys coerced", () => {
  assert.equal(dumps({ b: 1, a: 2 }, { sortKeys: true }), '{"a": 2, "b": 1}');
  // U+FF61 sorts before U+1F600 by code point, but after its surrogate pair by UTF-16 unit
  assert.equal(dumps({ "\u{1F600}": 1, "｡": 2 }, { sortKeys: true, ensureAscii: false }), '{"｡": 2, "\u{1F600}": 1}');
  assert.ok(comparePyStr("｡", "\u{1F600}") < 0);
  assert.equal(dumps(new Map<any, any>([["b", 1], [1, 2], [pyFloat(2), 3], [true, 4], [null, 5]])), '{"b": 1, "1": 2, "2.0": 3, "true": 4, "null": 5}');
});

test("errors: circular references, unsupported types, allowNan=false", () => {
  const a: unknown[] = [];
  a.push(a);
  assert.throws(() => dumps(a), /Circular reference/);
  assert.throws(() => dumps(new Date()), /not JSON serializable/);
  assert.throws(() => dumps(() => 1), /not JSON serializable/);
  assert.throws(() => dumps(NaN, { allowNan: false }), RangeError);
  assert.equal(dumps([1, { a: [] }], { separators: [",", ":"] }), '[1,{"a":[]}]');
});

test("pyRound edge cases follow CPython", () => {
  // values computed with CPython 3.12 round()
  assert.equal(pyRound(2.675, 2), 2.67);
  assert.equal(pyRound(0.125, 2), 0.12);
  assert.equal(pyRound(0.375, 2), 0.38);
  assert.equal(pyRound(2.5), 2);
  assert.equal(pyRound(3.5), 4);
  assert.equal(pyRound(1234.5, -2), 1200);
  assert.ok(Object.is(pyRound(-0.00001, 4), -0));
  assert.ok(Number.isNaN(pyRound(NaN, 4)));
  assert.equal(pyRound(Infinity, 4), Infinity);
  assert.equal(pyRound(5e-324, 400), 5e-324);
});

test("pyFormatG matches CPython %g", () => {
  // values computed with CPython 3.12 "%.<p>g" % x
  const cases: Array<[number, number, string]> = [
    [0.1006, 4, "0.1006"],
    [0.10058280825614929, 4, "0.1006"],
    [9.0, 4, "9"],
    [123456.0, 4, "1.235e+05"],
    [0.0001234, 4, "0.0001234"],
    [0.00001234, 4, "1.234e-05"],
    [2.5, 1, "2"],
    [1e22, 6, "1e+22"],
    [-0.0, 4, "-0"],
    [100, 3, "100"],
    [1000, 3, "1e+03"],
  ];
  for (const [x, p, want] of cases) assert.equal(pyFormatG(x, p), want, `%.${p}g % ${x}`);
});
