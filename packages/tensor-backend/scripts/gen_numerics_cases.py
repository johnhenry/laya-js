"""Generates fixtures/ops-numerics.json: conformance cases for the optional
"general numerics" ops of the tensor-backend contract (comparisons, logical
ops, sqrt/rsqrt/pow/neg/abs/tanh/sigmoid/erf, argmax/argmin/mean/min, cumsum).

Reference outputs come from MLX on Metal (the same library the Python Laya
reference runs on), cross-checked against NumPy where NumPy has the op.

    cd packages/tensor-backend
    uv run --with mlx==0.32.2 --with numpy python scripts/gen_numerics_cases.py

Float inputs are chosen exactly representable in bf16 (small multiples of
powers of two), so the f16 and bf16 conformance runs feed the op the same
values as the f32 run and only the result rounding differs; comparison and
arg-reduction inputs are therefore free of rounding-induced flips.
"""

from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "ops-numerics.json"
rng = np.random.default_rng(20260924)
mx.set_default_device(mx.gpu)


# Contract DType -> (numpy little-endian dtype string, is this "preserve the
# input's own numpy dtype" or "cast to this exact one"). Extends the
# original bool/i32/f32 three-bucket encoding (still the default below) with
# the 8 dtypes added 2026-09-25 for full tensor-core parity.
NP_DTYPE = {
    "bool": "<u1", "u8": "<u1", "i8": "<i1", "u16": "<u2", "i16": "<i2",
    "u32": "<u4", "i32": "<i4", "u64": "<u8", "i64": "<i8",
    "f32": "<f4", "f64": "<f8",
}


def enc(a: np.ndarray, dtype: str | None = None) -> dict:
    """`dtype`, when given, must be a key of NP_DTYPE and forces that exact
    encoding (used for the new dtypes below, which the original inference
    below can't express -- it only ever produces bool/i32/f32)."""
    if dtype is not None:
        data = a.astype(np.bool_).astype(NP_DTYPE["bool"]) if dtype == "bool" else a.astype(NP_DTYPE[dtype])
    elif a.dtype == np.bool_:
        dtype, data = "bool", a.astype(np.uint8)
    elif np.issubdtype(a.dtype, np.integer):
        dtype, data = "i32", a.astype("<i4")
    else:
        dtype, data = "f32", a.astype("<f4")
    return {"dtype": dtype, "shape": list(a.shape), "b64": base64.b64encode(np.ascontiguousarray(data).tobytes()).decode()}


def grid(shape, lo: float, hi: float, step: float = 0.125) -> np.ndarray:
    """Uniform random multiples of `step` in [lo, hi] (bf16-exact for |v/step| <= 256)."""
    n = rng.integers(round(lo / step), round(hi / step) + 1, size=shape)
    return (n * step).astype(np.float32)


def distinct(shape, axis: int) -> np.ndarray:
    """Values distinct along `axis` (permutations of k/4), bf16-exact."""
    a = np.moveaxis(np.zeros(shape, np.float32), axis, -1)
    flat = a.reshape(-1, a.shape[-1])
    for r in range(flat.shape[0]):
        flat[r] = (rng.permutation(flat.shape[1]) - flat.shape[1] // 2) / 4
    return np.moveaxis(flat.reshape(a.shape), -1, axis).copy()


def to_np(y: mx.array, dtype: str | None = None) -> np.ndarray:
    """`dtype`, when given, must be a key of NP_DTYPE and preserves that
    exact dtype's fidelity, bypassing the narrowing below -- needed for the
    8 dtypes added 2026-09-25 (this function predates them: previously every
    non-bool integer MLX dtype, including u32/i64/u64, was force-narrowed to
    i32, which is wrong once those dtypes are tested for real)."""
    mx.eval(y)
    if dtype is not None:
        return np.array(y).astype(NP_DTYPE[dtype])
    if y.dtype in (mx.uint32, mx.int64, mx.uint64, mx.int32):
        return np.array(y.astype(mx.int32))
    if y.dtype == mx.bool_:
        return np.array(y)
    return np.array(y.astype(mx.float32))


cases: list[dict] = []


def case(
    name: str,
    op: str,
    inputs: list[np.ndarray],
    out: np.ndarray,
    *,
    args=None,
    atol=0.0,
    rtol=0.0,
    check=None,
    in_dtypes: list[str] | None = None,
    out_dtype: str | None = None,
    **flags,
):
    """`in_dtypes`/`out_dtype` force exact encoding for the 8 dtypes added
    2026-09-25 that `enc`'s original bool/i32/f32 inference can't express.
    `check` still compares through float64, which is exact for every dtype
    used in this file's new cases (chosen well within float64's 52-bit
    mantissa, not near the true u64/i64 boundary -- a boundary-value case
    that needs bit-exact checking should compare `out` to `check` directly
    instead of routing through this parameter)."""
    if check is not None:
        np.testing.assert_allclose(out.astype(np.float64), np.asarray(check, dtype=np.float64), atol=max(atol, 1e-6), rtol=max(rtol, 1e-5), err_msg=name)
    ins = [enc(x, d) for x, d in zip(inputs, in_dtypes)] if in_dtypes else [enc(x) for x in inputs]
    c = {"name": name, "op": op, "args": args or {}, "inputs": ins, "outputs": [enc(out, out_dtype)], "atol": atol, "rtol": rtol}
    c.update({k: v for k, v in flags.items() if v})
    cases.append(c)


def arr(x: np.ndarray) -> mx.array:
    # mx.array() silently downcasts float64 numpy input to float32 by
    # default (confirmed empirically: every other numpy dtype round-trips
    # exactly, only float64 does not) -- explicit dtype=mx.float64 is
    # required or every f64 fixture is actually only f32-precise, which
    # broke every f64 conformance case at generation time, not at the
    # tensor-cpu consumer (2026-09-25).
    if x.dtype == np.float64:
        return mx.array(x, dtype=mx.float64)
    return mx.array(x)


# ---------------------------------------------------------------- comparisons
CMP = {
    "equal": (mx.equal, np.equal),
    "notEqual": (mx.not_equal, np.not_equal),
    "less": (mx.less, np.less),
    "lessEqual": (mx.less_equal, np.less_equal),
    "greater": (mx.greater, np.greater),
    "greaterEqual": (mx.greater_equal, np.greater_equal),
}
for op, (fm, fn) in CMP.items():
    a, b = grid((4, 6), -2, 2, 0.5), grid((4, 6), -2, 2, 0.5)
    case(f"{op}/same-shape", op, [a, b], to_np(fm(arr(a), arr(b))), check=fn(a, b))
    a, b = grid((2, 3, 4), -2, 2, 0.5), grid((3, 1), -2, 2, 0.5)
    case(f"{op}/broadcast", op, [a, b], to_np(fm(arr(a), arr(b))), check=fn(a, b))
    ai, bi = rng.integers(-3, 4, (5,)).astype(np.int32), rng.integers(-3, 4, (2, 5)).astype(np.int32)
    case(f"{op}/i32", op, [ai, bi], to_np(fm(arr(ai), arr(bi))), check=fn(ai, bi))
ab, bb = rng.integers(0, 2, (3, 4)).astype(np.bool_), rng.integers(0, 2, (3, 4)).astype(np.bool_)
case("equal/bool", "equal", [ab, bb], to_np(mx.equal(arr(ab), arr(bb))), check=np.equal(ab, bb))

# ---------------------------------------------------------------- logical
for op, (fm, fn) in {"logicalAnd": (mx.logical_and, np.logical_and), "logicalOr": (mx.logical_or, np.logical_or)}.items():
    a, b = rng.integers(0, 2, (3, 5)).astype(np.bool_), rng.integers(0, 2, (5,)).astype(np.bool_)
    case(f"{op}/bool-broadcast", op, [a, b], to_np(fm(arr(a), arr(b))), check=fn(a, b))
    af, bf = grid((2, 6), -1, 1, 0.5), grid((2, 6), -1, 1, 0.5)  # nonzero is true
    case(f"{op}/float", op, [af, bf], to_np(fm(arr(af).astype(mx.bool_), arr(bf).astype(mx.bool_))), check=fn(af != 0, bf != 0))
a = rng.integers(0, 2, (4, 3)).astype(np.bool_)
case("logicalNot/bool", "logicalNot", [a], to_np(mx.logical_not(arr(a))), check=np.logical_not(a))
af = grid((7,), -1, 1, 0.5)
case("logicalNot/float", "logicalNot", [af], to_np(mx.logical_not(arr(af).astype(mx.bool_))), check=af == 0)

# ---------------------------------------------------------------- unary math
x = grid((5, 8), 0, 16)
x[0, 0] = 0.0
case("sqrt/nonneg", "sqrt", [x], to_np(mx.sqrt(arr(x))), atol=1e-6, rtol=1e-5, check=np.sqrt(x))
x = grid((5, 8), 0.125, 16)
case("rsqrt/positive", "rsqrt", [x], to_np(mx.rsqrt(arr(x))), atol=1e-6, rtol=1e-5, check=1 / np.sqrt(x.astype(np.float64)))

a, b = grid((4, 6), 0.125, 4), grid((4, 6), -2, 2, 0.25)
case("pow/positive-base", "pow", [a, b], to_np(mx.power(arr(a), arr(b))), atol=1e-6, rtol=1e-5, check=np.power(a.astype(np.float64), b))
a, b = grid((3, 4), 0.25, 3, 0.25), grid((4,), -1, 2, 0.5)
case("pow/broadcast", "pow", [a, b], to_np(mx.power(arr(a), arr(b))), atol=1e-6, rtol=1e-5, check=np.power(a.astype(np.float64), b))
a, b = np.zeros((3,), np.float32), np.array([0.5, 1, 2.5], np.float32)
case("pow/zero-base", "pow", [a, b], to_np(mx.power(arr(a), arr(b))), atol=1e-6, rtol=1e-5, check=np.power(a, b))
a = grid((3, 5), -3, 3, 0.5)
b = rng.integers(0, 4, (3, 5)).astype(np.float32)
case("pow/negative-base-integral-exponent", "pow", [a, b], to_np(mx.power(arr(a), arr(b))), atol=1e-6, rtol=1e-5, check=np.power(a.astype(np.float64), b), nativeOnly=True)

x = grid((3, 7), -4, 4)
case("neg/f32", "neg", [x], to_np(mx.negative(arr(x))), check=-x)
case("abs/f32", "abs", [x], to_np(mx.abs(arr(x))), check=np.abs(x))
xi = rng.integers(-1000, 1000, (2, 5)).astype(np.int32)
case("neg/i32", "neg", [xi], to_np(mx.negative(arr(xi))), check=-xi)
case("abs/i32", "abs", [xi], to_np(mx.abs(arr(xi))), check=np.abs(xi))

x = np.concatenate([grid((41,), -8, 8), np.array([-20, -12, 12, 20, 0, 0.125, -0.125], np.float32)]).reshape(3, 1, -1)
case("tanh/wide", "tanh", [x], to_np(mx.tanh(arr(x))), atol=1e-6, rtol=1e-5, check=np.tanh(x.astype(np.float64)))
case("sigmoid/wide", "sigmoid", [x], to_np(mx.sigmoid(arr(x))), atol=1e-6, rtol=1e-5, check=1 / (1 + np.exp(-x.astype(np.float64))))

x = np.concatenate([np.arange(-64, 65, dtype=np.float32) / 16, np.array([-10, -6, -5, 5, 6, 10, 1 / 64, -1 / 64], np.float32)]).reshape(1, -1)
ref = np.array([math.erf(float(v)) for v in x.ravel()]).reshape(x.shape)
case("erf/dense", "erf", [x], to_np(mx.erf(arr(x))), atol=1e-6, rtol=0.0, check=ref)

# ---------------------------------------------------------------- reductions
for op, fm, fn in (("argmax", mx.argmax, np.argmax), ("argmin", mx.argmin, np.argmin)):
    x = distinct((4, 7), -1)
    case(f"{op}/last-axis", op, [x], to_np(fm(arr(x), axis=-1)), args={"axis": -1, "keepDims": False}, check=fn(x, axis=-1))
    x = distinct((5, 3, 2), 0)
    case(f"{op}/axis0-keepdims", op, [x], to_np(fm(arr(x), axis=0, keepdims=True)), args={"axis": 0, "keepDims": True}, check=fn(x, axis=0, keepdims=True))
    x = grid((3, 6), -1, 1, 0.5)  # ties: the first index wins
    case(f"{op}/ties-first", op, [x], to_np(fm(arr(x), axis=1)), args={"axis": 1, "keepDims": False}, check=fn(x, axis=1))
    xi = rng.integers(-50, 50, (3, 4)).astype(np.int32)
    case(f"{op}/i32", op, [xi], to_np(fm(arr(xi), axis=1)), args={"axis": 1, "keepDims": False}, check=fn(xi, axis=1))

x = grid((3, 4, 5), -4, 4)
case("mean/axis1", "mean", [x], to_np(mx.mean(arr(x), axis=1)), args={"axis": 1, "keepDims": False}, atol=1e-6, rtol=1e-5, check=x.astype(np.float64).mean(axis=1))
case("mean/last-keepdims", "mean", [x], to_np(mx.mean(arr(x), axis=-1, keepdims=True)), args={"axis": -1, "keepDims": True}, atol=1e-6, rtol=1e-5, check=x.astype(np.float64).mean(axis=-1, keepdims=True))
xi = rng.integers(-9, 10, (4, 3)).astype(np.int32)
case("mean/i32", "mean", [xi], to_np(mx.mean(arr(xi), axis=0)), args={"axis": 0, "keepDims": False}, atol=1e-6, rtol=1e-5, check=xi.mean(axis=0))

case("min/axis2", "min", [x], to_np(mx.min(arr(x), axis=2)), args={"axis": 2, "keepDims": False}, check=x.min(axis=2))
case("min/i32-keepdims", "min", [xi], to_np(mx.min(arr(xi), axis=0, keepdims=True)), args={"axis": 0, "keepDims": True}, check=xi.min(axis=0, keepdims=True))

x = grid((3, 6), -4, 4)
case("cumsum/last-axis", "cumsum", [x], to_np(mx.cumsum(arr(x), axis=-1)), args={"axis": -1}, atol=1e-5, rtol=1e-5, check=np.cumsum(x, axis=-1))
x = grid((4, 3, 2), -4, 4)
case("cumsum/axis0", "cumsum", [x], to_np(mx.cumsum(arr(x), axis=0)), args={"axis": 0}, atol=1e-5, rtol=1e-5, check=np.cumsum(x, axis=0))
xi = rng.integers(-20, 20, (2, 5)).astype(np.int32)
case("cumsum/i32", "cumsum", [xi], to_np(mx.cumsum(arr(xi), axis=1)), args={"axis": 1}, check=np.cumsum(xi, axis=1))

# ---------------------------------------------------------------- new dtypes (2026-09-25)
# u8/i8/u16/i16/u32/u64/i64/f64: full tensor-core parity (see tensor-backend's
# DType docs). Comparisons/reductions are dtype-generic in what they compute,
# so what actually needs new coverage is each dtype's real range (not just
# small values) and, for f64, genuine double-precision fidelity. `cast`
# round-trip fixtures belong in ops.json (laya-mlx's generator, not this
# script) since cast is a required base op, not a NUMERICS_OPS entry --
# covered instead by a hand-written backend-mlx test (see its test suite).

INT_RANGES = {"u8": (0, 256), "i8": (-128, 128), "u16": (0, 65536), "i16": (-32768, 32768), "u32": (0, 1_000_000)}
for dt, (lo, hi) in INT_RANGES.items():
    a = rng.integers(lo, hi, (4, 5)).astype(NP_DTYPE[dt])
    b = rng.integers(lo, hi, (4, 5)).astype(NP_DTYPE[dt])
    case(f"equal/{dt}", "equal", [a, b], to_np(mx.equal(arr(a), arr(b)), "bool"), in_dtypes=[dt, dt], out_dtype="bool", check=np.equal(a, b))
    case(f"less/{dt}", "less", [a, b], to_np(mx.less(arr(a), arr(b)), "bool"), in_dtypes=[dt, dt], out_dtype="bool", check=np.less(a, b))
    x = rng.integers(lo, hi, (3, 6)).astype(NP_DTYPE[dt])
    case(f"argmax/{dt}", "argmax", [x], to_np(mx.argmax(arr(x), axis=1), "i32"), args={"axis": 1, "keepDims": False}, in_dtypes=[dt], check=np.argmax(x, axis=1))
    case(f"mean/{dt}", "mean", [x], to_np(mx.mean(arr(x), axis=0)), args={"axis": 0, "keepDims": False}, atol=1e-6, rtol=1e-5, in_dtypes=[dt], check=x.astype(np.float64).mean(axis=0))
    # NOT upcast: MLX's cumsum on a narrow integer dtype accumulates (and
    # wraps) IN that dtype -- confirmed empirically (u8 cumsum wraps at 256,
    # it does not auto-promote the way i32/bool -> i32 does per the existing
    # contract doc comment). Same-dtype numpy cumsum wraps identically, so
    # the two agree exactly as long as neither side upcasts.
    # dtype= forces numpy to wrap the same way MLX does; numpy's own default
    # auto-upcasts narrow integer cumsum to a wider type to avoid overflow,
    # which would silently disagree with MLX's real wrapping behavior.
    case(f"cumsum/{dt}", "cumsum", [x], to_np(mx.cumsum(arr(x), axis=1), dt), args={"axis": 1}, in_dtypes=[dt], out_dtype=dt, check=np.cumsum(x, axis=1, dtype=NP_DTYPE[dt]))

# Signed new dtypes only: neg/abs (unsigned wraparound makes these meaningless).
for dt, (lo, hi) in (("i8", (-128, 128)), ("i16", (-32768, 32768))):
    x = rng.integers(lo, hi, (3, 4)).astype(NP_DTYPE[dt])
    case(f"neg/{dt}", "neg", [x], to_np(mx.negative(arr(x)), dt), in_dtypes=[dt], out_dtype=dt, check=-x)
    case(f"abs/{dt}", "abs", [x], to_np(mx.abs(arr(x)), dt), in_dtypes=[dt], out_dtype=dt, check=np.abs(x))

# u64/i64: moderate ranges only here (values chosen well within float64's
# exact-integer range so the `check` path stays bit-exact); a real u64/i64
# boundary round-trip near 2**63/2**64 is a hand-written backend-mlx test,
# not a JSON-fixture case, since it needs bit-exact (not allclose) comparison.
u64a, u64b = rng.integers(0, 1_000_000, (3, 4)).astype(NP_DTYPE["u64"]), rng.integers(0, 1_000_000, (3, 4)).astype(NP_DTYPE["u64"])
case("equal/u64", "equal", [u64a, u64b], to_np(mx.equal(arr(u64a), arr(u64b)), "bool"), in_dtypes=["u64", "u64"], out_dtype="bool", check=np.equal(u64a, u64b))
case("less/u64", "less", [u64a, u64b], to_np(mx.less(arr(u64a), arr(u64b)), "bool"), in_dtypes=["u64", "u64"], out_dtype="bool", check=np.less(u64a, u64b))
u64x = rng.integers(0, 1_000_000, (3, 6)).astype(NP_DTYPE["u64"])
case("argmax/u64", "argmax", [u64x], to_np(mx.argmax(arr(u64x), axis=1), "i32"), args={"axis": 1, "keepDims": False}, in_dtypes=["u64"], check=np.argmax(u64x, axis=1))
case("cumsum/u64", "cumsum", [u64x], to_np(mx.cumsum(arr(u64x), axis=1), "u64"), args={"axis": 1}, in_dtypes=["u64"], out_dtype="u64", check=np.cumsum(u64x, axis=1, dtype=NP_DTYPE["u64"]))

i64a, i64b = rng.integers(-1_000_000, 1_000_000, (3, 4)).astype(NP_DTYPE["i64"]), rng.integers(-1_000_000, 1_000_000, (3, 4)).astype(NP_DTYPE["i64"])
case("equal/i64", "equal", [i64a, i64b], to_np(mx.equal(arr(i64a), arr(i64b)), "bool"), in_dtypes=["i64", "i64"], out_dtype="bool", check=np.equal(i64a, i64b))
case("less/i64", "less", [i64a, i64b], to_np(mx.less(arr(i64a), arr(i64b)), "bool"), in_dtypes=["i64", "i64"], out_dtype="bool", check=np.less(i64a, i64b))
i64x = rng.integers(-1_000_000, 1_000_000, (3, 4)).astype(NP_DTYPE["i64"])
case("neg/i64", "neg", [i64x], to_np(mx.negative(arr(i64x)), "i64"), in_dtypes=["i64"], out_dtype="i64", check=-i64x)
case("cumsum/i64", "cumsum", [i64x], to_np(mx.cumsum(arr(i64x), axis=1), "i64"), args={"axis": 1}, in_dtypes=["i64"], out_dtype="i64", check=np.cumsum(i64x, axis=1, dtype=NP_DTYPE["i64"]))

# f64: values chosen so correctness genuinely requires double precision --
# a result silently computed in f32 (e.g. by a backend that downcasts f64
# before running the op) would fail these tight tolerances.
xf64 = (rng.integers(-40000, 40000, (4, 5)).astype(np.float64) + rng.random((4, 5))) / 131072
with mx.stream(mx.cpu):  # float64 is CPU-only on MLX -- confirmed, it raises on the GPU stream
    case("sqrt/f64", "sqrt", [np.abs(xf64)], to_np(mx.sqrt(arr(np.abs(xf64))), "f64"), in_dtypes=["f64"], out_dtype="f64", atol=1e-14, rtol=1e-13, check=np.sqrt(np.abs(xf64)))
    case("tanh/f64", "tanh", [xf64], to_np(mx.tanh(arr(xf64)), "f64"), in_dtypes=["f64"], out_dtype="f64", atol=1e-14, rtol=1e-13, check=np.tanh(xf64))
    case("mean/f64", "mean", [xf64], to_np(mx.mean(arr(xf64), axis=0), "f64"), args={"axis": 0, "keepDims": False}, in_dtypes=["f64"], out_dtype="f64", atol=1e-14, rtol=1e-13, check=xf64.mean(axis=0))
    case("cumsum/f64", "cumsum", [xf64], to_np(mx.cumsum(arr(xf64), axis=1), "f64"), args={"axis": 1}, in_dtypes=["f64"], out_dtype="f64", atol=1e-13, rtol=1e-12, check=np.cumsum(xf64, axis=1))

OUT.write_text(json.dumps(cases, indent=1) + "\n")
print(f"wrote {len(cases)} cases over {len({c['op'] for c in cases})} ops to {OUT} (MLX {mx.__version__}, {mx.default_device()})", file=sys.stderr)
