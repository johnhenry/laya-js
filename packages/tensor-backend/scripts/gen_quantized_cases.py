"""Generates fixtures/ops-quantized.json: conformance cases for the optional
quantized-weight ops of the tensor-backend contract (`quantizedLinear`,
`quantizedEmbedding`), in the laya-js quantized layout:

- q8 symmetric: w = q · scale, q ∈ [-127, 127] (laya-js q8 checkpoints);
- q4 affine:    w = q · scale + bias, q ∈ [0, 15] (laya-js q4 checkpoints);
- q8 affine and q4 symmetric, for completeness of the contract;

with per-group float16 scales along the input dim (groups of 32/64/128, and
partial last groups). Affine parameters and values come from MLX's own
`mx.quantize`; every reference output is computed in float64 with NumPy and,
where MLX has the configuration (affine or symmetric-as-affine, full groups
of 32/64/128), cross-checked against `mx.quantized_matmul` on Metal.

    cd packages/tensor-backend
    uv run --with mlx==0.32.2 --with numpy python scripts/gen_quantized_cases.py

Inputs per case: quantizedLinear [x, q (i32 values), scales, biases | null,
bias | null]; quantizedEmbedding [q, scales, biases | null, ids]. The
conformance runner packs q into `HostQuantized.data` and uploads it with
`uploadQuantized`, so native kernels and the default composition see the
same bytes.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "ops-quantized.json"
rng = np.random.default_rng(20260925)
mx.set_default_device(mx.gpu)


def enc(a: np.ndarray) -> dict:
    if np.issubdtype(a.dtype, np.integer):
        dtype, data = "i32", a.astype("<i4")
    else:
        dtype, data = "f32", a.astype("<f4")
    return {"dtype": dtype, "shape": list(a.shape), "b64": base64.b64encode(np.ascontiguousarray(data).tobytes()).decode()}


def grid(shape, lo: float, hi: float, step: float = 0.125) -> np.ndarray:
    n = rng.integers(round(lo / step), round(hi / step) + 1, size=shape)
    return (n * step).astype(np.float32)


def unpack_mlx(wq: mx.array, bits: int, K: int) -> np.ndarray:
    """MLX uint32 packing → integer values [N, K]."""
    u = np.array(wq).astype(np.uint64)
    per = 32 // bits
    cols = [(u >> np.uint64(bits * j)) & np.uint64((1 << bits) - 1) for j in range(per)]
    return np.stack(cols, axis=-1).reshape(u.shape[0], -1)[:, :K].astype(np.int64)


def quantize(N: int, K: int, bits: int, g: int, mode: str):
    """(q, scales f16-exact f32, biases or None) for a random [N, K] matrix."""
    G = -(-K // g)
    w = (rng.standard_normal((N, K)) * 0.05).astype(np.float16)
    if mode == "affine" and K % g == 0 and g in (32, 64, 128):
        wq, s, b = mx.quantize(mx.array(w), group_size=g, bits=bits)
        return unpack_mlx(wq, bits, K), np.array(s.astype(mx.float32)), np.array(b.astype(mx.float32))
    q = np.zeros((N, K), np.int64)
    scales = np.zeros((N, G), np.float32)
    biases = np.zeros((N, G), np.float32) if mode == "affine" else None
    lv = (1 << bits) - 1
    for r in range(N):
        for gi in range(G):
            seg = w[r, gi * g:(gi + 1) * g].astype(np.float64)
            if mode == "symmetric":
                qmax = (1 << (bits - 1)) - 1
                s = np.float16(max(np.abs(seg).max(), 1e-4) / qmax)
                q[r, gi * g:(gi + 1) * g] = np.clip(np.round(seg / np.float64(s)), -qmax - 1, qmax)
                scales[r, gi] = s
            else:
                lo, hi = seg.min(), seg.max()
                b = np.float16(lo)
                s = np.float16(max(hi - np.float64(b), 1e-4) / lv)
                q[r, gi * g:(gi + 1) * g] = np.clip(np.round((seg - np.float64(b)) / np.float64(s)), 0, lv)
                scales[r, gi], biases[r, gi] = s, b
    return q, scales, biases


def dequant(q: np.ndarray, scales: np.ndarray, biases, g: int) -> np.ndarray:
    K = q.shape[1]
    idx = np.arange(K) // g
    w = q.astype(np.float64) * scales.astype(np.float64)[:, idx]
    if biases is not None:
        w += biases.astype(np.float64)[:, idx]
    return w


def mlx_check(x: np.ndarray, q, scales, biases, bits: int, g: int, mode: str, want: np.ndarray, bias) -> bool:
    """Cross-check with mx.quantized_matmul where MLX supports the configuration."""
    K = q.shape[1]
    if K % g or g not in (32, 64, 128):
        return False
    qa, ba = q, biases
    if mode == "symmetric":  # MLX has affine only: q_u = q + 2^(bits-1), bias = -2^(bits-1)·scale
        off = 1 << (bits - 1)
        qa, ba = q + off, (-off * scales.astype(np.float64)).astype(np.float32)
    per = 32 // bits
    u = qa.astype(np.uint64).reshape(q.shape[0], K // per, per)
    packed = np.zeros(u.shape[:2], np.uint64)
    for j in range(per):
        packed |= u[:, :, j] << np.uint64(bits * j)
    y = mx.quantized_matmul(mx.array(x), mx.array(packed.astype(np.uint32)), mx.array(scales), mx.array(ba), transpose=True, group_size=g, bits=bits)
    if bias is not None:
        y = y + mx.array(bias)
    np.testing.assert_allclose(np.array(y).astype(np.float64), want, atol=2e-4, rtol=2e-4)
    return True


cases: list[dict] = []
checked = 0


def linear_case(xshape, N: int, bits: int, g: int, mode: str, with_bias: bool):
    global checked
    K = xshape[-1]
    q, scales, biases = quantize(N, K, bits, g, mode)
    x = grid(xshape, -1, 1)
    bias = grid((N,), -0.5, 0.5) if with_bias else None
    want = x.astype(np.float64) @ dequant(q, scales, biases, g).T
    if bias is not None:
        want += bias
    checked += mlx_check(x, q, scales, biases, bits, g, mode, want, bias)
    G = -(-K // g)
    part = "" if K % g == 0 else f", partial last group ({K - (G - 1) * g})"
    name = f"quantizedLinear q{bits} {mode} g{g} x{list(xshape)} w[{N},{K}]{' +bias' if with_bias else ''}{part}"
    cases.append({
        "name": name, "op": "quantizedLinear", "args": {"bits": bits, "groupSize": g, "mode": mode},
        "inputs": [enc(x), enc(q), enc(scales), enc(biases) if biases is not None else None, enc(bias) if bias is not None else None],
        "outputs": [enc(want.astype(np.float32))], "atol": 1e-4, "rtol": 1e-4,
    })


def embedding_case(V: int, D: int, ids_shape, bits: int, g: int, mode: str):
    q, scales, biases = quantize(V, D, bits, g, mode)
    ids = rng.integers(0, V, size=ids_shape).astype(np.int32)
    want = dequant(q, scales, biases, g)[ids]
    cases.append({
        "name": f"quantizedEmbedding q{bits} {mode} g{g} table[{V},{D}] ids{list(ids_shape)}", "op": "quantizedEmbedding",
        "args": {"bits": bits, "groupSize": g, "mode": mode},
        "inputs": [enc(q), enc(scales), enc(biases) if biases is not None else None, enc(ids)],
        "outputs": [enc(want.astype(np.float32))], "atol": 1e-5, "rtol": 1e-5,
    })


# laya-js q8 (symmetric) and q4 (affine), group 64: batch-1-like, skinny and tiled M, rank 3
for bits, mode in ((8, "symmetric"), (4, "affine")):
    linear_case((1, 128), 64, bits, 64, mode, False)
    linear_case((5, 256), 48, bits, 64, mode, True)
    linear_case((2, 9, 192), 40, bits, 64, mode, True)
    linear_case((70, 128), 40, bits, 64, mode, True)
    linear_case((130, 128), 24, bits, 64, mode, False)
    linear_case((3, 96), 40, bits, 64, mode, True)  # partial last group
    linear_case((4, 64), 32, bits, 32, mode, False)
    linear_case((2, 256), 16, bits, 128, mode, True)
    linear_case((3, 40), 24, bits, 16, mode, True)  # group size MLX lacks (and partial)
    embedding_case(50, 128, (2, 5), bits, 64, mode)
    embedding_case(33, 96, (7,), bits, 64, mode)  # partial last group
# the other two contract combinations
linear_case((6, 128), 32, 8, 64, "affine", True)
linear_case((6, 128), 32, 4, 64, "symmetric", True)
embedding_case(20, 64, (4,), 8, 32, "affine")
embedding_case(20, 64, (4,), 4, 32, "symmetric")

assert checked >= 16, checked
OUT.write_text(json.dumps(cases, indent=1) + "\n")
print(f"wrote {len(cases)} cases ({checked} cross-checked against mx.quantized_matmul) to {OUT}")
