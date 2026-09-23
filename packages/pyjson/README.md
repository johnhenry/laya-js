# @johnhenry/pyjson

Byte-identical CPython `json.dumps`, `repr(float)`, `"%g"` and `round()` for JavaScript.
Zero dependencies; works in browsers, Node, Bun and Deno.

Use it when a JavaScript port must produce exactly the strings or numbers a Python
program would, e.g. prompts that embed `json.dumps(...)` output before tokenization.

```js
import { dumps, loads, pyFloat, pyRound, pyFloatRepr, pyFormatG } from "@johnhenry/pyjson";

dumps({ a: [1, 2.5, null], b: "é" });                 // '{"a": [1, 2.5, null], "b": "\\u00e9"}'
dumps({ b: "é" }, { ensureAscii: false });            // '{"b": "é"}'
dumps([1, 2], { separators: [",", ":"] });            // '[1,2]'
dumps(pyFloat(1));                                    // '1.0'  (a Python float)
dumps(1e16), dumps(pyFloat(1e16));                    // '10000000000000000', '1e+16'
dumps(NaN);                                           // 'NaN'
pyFloatRepr(1e-7);                                    // '1e-07'
pyRound(2.675, 2);                                    // 2.67 (the double is below 2.675)
pyRound(0.125, 2);                                    // 0.12 (ties to even)
pyFormatG(0.10058280825614929, 4);                    // '0.1006'  ("%.4g")
dumps(loads("[1.0, 9007199254740993]"));              // '[1.0, 9007199254740993]'
```

## API

| Export | Python equivalent |
|---|---|
| `dumps(value, { ensureAscii = true, separators = [", ", ": "], sortKeys = false, allowNan = true })` | `json.dumps(value, ensure_ascii=, separators=, sort_keys=, allow_nan=)` |
| `loads(text, { wrapFloats = true, bigInts = true })` | `json.loads` keeping int/float apart: floats (`1.0`, `1e5`, `NaN`) become `PyFloat`, ints beyond 2^53 become `bigint` |
| `pyFloat(x)` / `PyFloat` / `isPyFloat(x)` | mark a number as a Python `float` |
| `pyFloatRepr(x)` | `repr(float(x))` |
| `pyRound(x, ndigits = 0)` | `round(x, ndigits)` for floats (always returns a number) |
| `pyFormatG(x, precision = 6)` | `"%.<precision>g" % x` |
| `encodeString(s, ensureAscii = true)` | `json.encoder.py_encode_basestring(_ascii)` |
| `comparePyStr(a, b)` | Python `str` ordering (code points, not UTF-16 units) |

`dumps` accepts `null`/`undefined` (both `null`), booleans, numbers, `bigint`, `PyFloat`,
strings, arrays, plain objects and `Map`s (keys: string, number, `PyFloat`, `bigint`,
boolean, `null`, coerced like Python). Circular structures throw
`Error("Circular reference detected")`; other values throw a `TypeError` like Python's.

`pyRound`, `pyFloatRepr` and `pyFormatG` are exact: they work on the exact binary value of
the double with BigInt arithmetic, so there is no `Math.round(x * 1e4) / 1e4` drift.
They are tested against CPython on the fixture tables (and were cross-checked against
CPython on 20,000 random values during development).

## Limitations

- **int vs float.** JavaScript has one number type. A plain `number` prints as an int when
  `Number.isInteger(x)` holds (`1.0` in Python becomes `1` here) and as a float otherwise.
  Wrap with `pyFloat(x)` to get `1.0`, or pass a `bigint` for exact ints beyond 2^53 (a plain
  `1e30` prints as `BigInt(1e30)`, i.e. `1000000000000000019884624838656`). `-0` always prints
  as `-0.0` because only a Python float can be negative zero.
- **Key order.** Plain JS objects put integer-like keys (`"1"`, `"42"`) first, in numeric order,
  whatever order they were inserted in; Python dicts keep insertion order. Use a `Map` when
  that matters. `loads` returns plain objects, so `dumps(loads(s)) === s` only holds when
  integer-like keys already come first.
- No `indent`, `default=` or `skipkeys` options.
- `loads` is a strict JSON parser plus the `NaN`/`Infinity`/`-Infinity` tokens; it does not
  accept Python's other extensions.
