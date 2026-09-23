"""Python-MLX twin of real-english.ts's latency line: the shortest English fixture item, B=1.
Run: cd laya-mlx && uv run python ../laya-js/packages/backend-mlx/bench/python_real_short.py"""
import json, statistics, time, warnings
import numpy as np
from laya_mlx.agent import Agent

warnings.simplefilter("ignore")
fx = json.load(open(__file__.rsplit("/packages/", 1)[0] + "/packages/laya-fixtures/data/real/english.json"))
short = min((it for c in fx["cases"] for it in c["items"]), key=lambda it: len(it["ids"]))
L, M = len(short["ids"]), max(2, len(short["markers"]))
pad = M - len(short["markers"])
batch = dict(input_ids=np.array([short["ids"]], np.int32), attention_mask=np.ones((1, L), np.int32),
             marker_pos=np.array([short["markers"] + [0] * pad], np.int32),
             marker_mask=np.array([[1] * len(short["markers"]) + [0] * pad], np.bool_),
             qtype=np.array([short["qtype"]], np.int32))
for dtype in ["float16", "float32"]:
    ag = Agent(fx["model_dir"], dtype=dtype)  # compile=False, the released default
    for _ in range(5): ag.forward(batch)
    ts = []
    for _ in range(30):
        t0 = time.perf_counter(); r = ag.forward(batch); np.array(r[0]); ts.append((time.perf_counter() - t0) * 1e3)
    print(f"python {dtype} short question ({L} tok) P50 {statistics.median(ts):.2f} ms, min {min(ts):.2f} ms")
