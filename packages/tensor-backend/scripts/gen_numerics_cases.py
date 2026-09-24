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


def enc(a: np.ndarray) -> dict:
    if a.dtype == np.bool_:
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


def to_np(y: mx.array) -> np.ndarray:
    mx.eval(y)
    if y.dtype in (mx.uint32, mx.int64, mx.uint64, mx.int32):
        return np.array(y.astype(mx.int32))
    if y.dtype == mx.bool_:
        return np.array(y)
    return np.array(y.astype(mx.float32))


cases: list[dict] = []


def case(name: str, op: str, inputs: list[np.ndarray], out: np.ndarray, *, args=None, atol=0.0, rtol=0.0, check=None, **flags):
    if check is not None:
        np.testing.assert_allclose(out.astype(np.float64), np.asarray(check, dtype=np.float64), atol=max(atol, 1e-6), rtol=max(rtol, 1e-5), err_msg=name)
    c = {"name": name, "op": op, "args": args or {}, "inputs": [enc(x) for x in inputs], "outputs": [enc(out)], "atol": atol, "rtol": rtol}
    c.update({k: v for k, v in flags.items() if v})
    cases.append(c)


def arr(x: np.ndarray) -> mx.array:
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

OUT.write_text(json.dumps(cases, indent=1) + "\n")
print(f"wrote {len(cases)} cases over {len({c['op'] for c in cases})} ops to {OUT} (MLX {mx.__version__}, {mx.default_device()})", file=sys.stderr)
