"""Python-MLX twin of ops.bench.ts. Run: cd laya-mlx && uv run python <this file>"""
import statistics, time
import mlx.core as mx
import mlx.nn as nn

print("mlx", mx.__version__)
def bench(name, iters, fn, warm=10):
    for _ in range(warm): fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    m = statistics.median(ts)
    print(f"{name:<48} {m:.3f} ms")
    return m
up = lambda shape, s=0.02, dt=mx.float16: (mx.random.uniform(-s, s, shape)).astype(dt)

a, c = up((4,), 1, mx.float32), up((4,), 1, mx.float32)
N = 10000
def adds():
    for _ in range(N): a + c
print(f"  -> {bench('10k tiny adds: build+free (per op, us)', 5, adds) * 1000 / N:.3f} us/op")
def chain():
    x = a
    for _ in range(N): x = x + c
    mx.eval(x)
print(f"  -> {bench('10k chained adds: build+eval (per op, us)', 5, chain) * 1000 / N:.3f} us/op")
bench("1 tiny add + read (round trip)", 200, lambda: (a + c).tolist())

x, w = up((16, 128, 1024), 1), up((1024, 1024))
ms = bench("linear f16 [16,128,1024]x[1024,1024]^T", 50, lambda: mx.eval(x @ w.T))
print(f"  -> {2*16*128*1024*1024/ms/1e9:.2f} TFLOP/s")
B, H, L, D = 16, 16, 128, 64
q, k, v = up((B, H, L, D), 1), up((B, H, L, D), 1), up((B, H, L, D), 1)
mask = (mx.arange(L) < 100)[None, None, None, :] & mx.ones((B, 1, 1, L), dtype=mx.bool_)
bench("sdpa f16 [16,16,128,64] bool mask", 50, lambda: mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask=mask)))

Dm, H, Dh, I = 1024, 16, 64, 2624
W = dict(attnNorm=up((Dm,), 1), Wqkv=up((3*Dm, Dm)), Wo=up((Dm, Dm)), mlpNorm=up((Dm,), 1), Wi=up((2*I, Dm)), Wo2=up((Dm, I)))
def layer(x, mask):
    B, L, _ = x.shape
    h = mx.fast.layer_norm(x, W["attnNorm"], None, 1e-5)
    qkv = (h @ W["Wqkv"].T).reshape(B, L, 3, H, Dh).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = mx.fast.rope(q, Dh, traditional=False, base=160000.0, scale=1.0, offset=0)
    k = mx.fast.rope(k, Dh, traditional=False, base=160000.0, scale=1.0, offset=0)
    att = mx.fast.scaled_dot_product_attention(q, k, v, scale=Dh**-0.5, mask=mask)
    x1 = x + att.transpose(0, 2, 1, 3).reshape(B, L, Dm) @ W["Wo"].T
    value, gate = mx.split(mx.fast.layer_norm(x1, W["mlpNorm"], None, 1e-5) @ W["Wi"].T, 2, axis=-1)
    return x1 + (nn.gelu(value) * gate) @ W["Wo2"].T
for B, L in [(1, 32), (16, 128)]:
    x = up((B, L, Dm), 1)
    m = mx.ones((B, 1, 1, L), dtype=mx.bool_)
    bench(f"encoder layer eager B={B} L={L}", 30, lambda: mx.eval(layer(x, m)))
    cl = mx.compile(layer)
    bench(f"encoder layer compiled B={B} L={L}", 30, lambda: mx.eval(cl(x, m)))
    bench(f"  graph build only (py) B={B} L={L}", 30, lambda: layer(x, m))
