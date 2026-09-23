"""Dump Python Agent.prepare ids for the first recorded snake frames (compact prompt)."""
import json, sys
from collections import deque
from laya_mlx.snake.game import SnakeGame
from laya_mlx.snake.policy import LayaPolicy

src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = [json.loads(l) for l in open(src)]
frames = [l for l in lines if l["type"] == "frame"][:n]
policy = LayaPolicy(None, prompt="compact")
captured = {}
real = policy.agent.predict
def capture(state, questions):
    captured["state"], captured["questions"] = state, questions
    return real(state, questions)
policy.agent.predict = capture
cases = []
for f in frames:
    g = f["game"]
    game = SnakeGame(g["width"], g["height"], g["seed"])
    game.body = deque(tuple(c) for c in g["body"]); game.food = tuple(g["food"])
    d = policy.decide(game)
    items, _ = policy.agent.prepare(captured["state"], captured["questions"])
    cases.append({
        "tick": g["ticks"], "state": captured["state"], "questions": captured["questions"],
        "items": [{"ids": list(map(int, it["ids"])), "markers": list(map(int, it["markers"])), "qtype": int(it["qtype"])} for it in items],
        "probabilities": d.probabilities, "risk_noul": 1 - d.dead_end_risk, "food_noul": d.food_reachable,
    })
json.dump({"source": src, "note": "laya_mlx.snake LayaPolicy (compact, fp16, eager) on recorded boards; ids from Agent.prepare", "cases": cases}, open(out, "w"))
print(len(cases))
