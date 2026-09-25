# Laya Dino Run (terminal: Node and Bun)

A real game of Chrome's offline "no internet" dinosaur runner, driven by
real Laya predictions from [`@johnhenry/laya`](../../packages/laya) on
**MLX**, **WebGPU** (Dawn) or the CPU reference, in a truecolor ANSI
terminal UI with no dependencies. Built in the architecture of
[`examples/snake-terminal`](../snake-terminal): a deterministic, Node-free
`core/` (game rules, a lookahead safety classifier, Laya prompt-building, a
pure shield), then a thin terminal consumer.

**This game is JS-original**, like [`flappy-terminal`](../flappy-terminal),
[`checkers-terminal`](../checkers-terminal) and
[`tetris-terminal`](../tetris-terminal) — there is no Python reference
implementation to port from or match.

## The game

Tick-based, like Flappy Bird, but with **three actions instead of two**:
`JUMP`, `DUCK`, `RUN`. A cactus collides unless you're airborne. A
low-flying pterodactyl collides unless you're ducking — **jumping into
one is fatal**, mirroring the real game's actual "don't jump at every
bird" trap. A high-flying pterodactyl only collides if you jump into it —
`RUN` and `DUCK` are both safe under one. Once airborne, the jump arc is
uncontrollable for its fixed duration (matches the real game and Flappy
Bird's flap-sets-a-fixed-impulse precedent) — **no `predict()` call is
made on airborne ticks at all**, since no decision has any effect
mid-jump; the session only asks the model on grounded ticks, the same way
`tetris-terminal` only asks once per piece rather than once per tick.

## The lookahead shield — built reactive from day one

`core/game.ts`'s `moves()` classifies every action `safe` via a bounded
lookahead: simulate the tested action now, then coast for the rest of the
horizon. **The coast is reactive** (jump for an imminent cactus, duck for
an imminent low pterodactyl, run otherwise) — deliberately not a blind
"always run" coast. A blind coast has exactly the shape of bug
[`flappy-terminal`](../flappy-terminal) shipped and then fixed: if the
coast never reacts to anything, every path eventually runs into *some*
obstacle inside a long enough horizon, making every action look
permanently unsafe.

**This game's own version of that lesson was caught before shipping, not
after.** The reactive coast was written from the start, but a first
empirical simulation (run deliberately *before* writing the formal test
suite, learning directly from the Flappy Bird incident) still found a real
bug: a heuristic policy died to a pterodactyl on every single seed, with
**zero ducks ever chosen**. The reaction window used `obstacle.x >= dinoX`
to decide "is this still ahead of me," but the actual collision zone
extends slightly *behind* the dino too (`|x - dinoX| < 1.5`) — so the
coast let go of `DUCK` exactly while the pterodactyl was still physically
overlapping. Fixed by widening the window to match the real collision
zone; reverified afterward: the same heuristic now survives 5,000 ticks
across 8 seeds with a healthy mix of all three actions.

## Run

The default model is `aac6fef/laya-multilingual-mlx`, loaded **offline**
from the local Hugging Face cache, same convention as the other three demos.

```bash
npm start -w @johnhenry/example-dino-terminal                 # node, backend auto (mlx on Apple silicon)
npm run start:bun -w @johnhenry/example-dino-terminal         # same under Bun
npm run dev -w @johnhenry/example-dino-terminal                # --max-speed
npm start -w @johnhenry/example-dino-terminal -- --backend webgpu --max-speed
npm run bench -w @johnhenry/example-dino-terminal              # 4 x 1200 uncapped ticks, seeds 101-104
# directly:
node --conditions=source src/cli.ts --backend webgpu --headless --episodes 1 --steps 10
```

Keys: **Space** pause, **R** next seed, **Q** or Ctrl-C quit.

Options: `--model <id|dir>`, `--backend auto|mlx|webgpu|cpu`,
`--dtype f16|f32`, `--prompt compact|detailed`, `--optimize`, `--width`,
`--seed`, `--fps` (default 20), `--max-speed`, `--duration`, `--steps`,
`--unassisted`, `--record run.jsonl`, `--headless`, `--episodes N`,
`--no-alt-screen`, `--online`. Full list: `--help`.

## Tests (`npm test` / `npm run test:bun`)

`test/game.test.ts`: per-obstacle-kind collision rules (cactus needs
airborne, low pterodactyl needs ducking, high pterodactyl punishes
jumping), `moves()` empty while airborne, `LayaPolicy.decide()` proven to
never call `predict()` while airborne (a stub that throws if called), the
direct regression test for the reactive-coast bug, jump-arc physics
duration, seed reproducibility, the shield's guard/override/empty-safe-set
behavior, and a 5,000-tick multi-seed survival test that explicitly
asserts both `JUMP` and `DUCK` are meaningfully exercised (not just "the
game didn't crash," which is exactly what the original bug's test
coverage would have missed). **13 cases, 0 skipped** — no Python
reference exists, so there's no fixture-parity tier.

## Verified so far

- `npm run typecheck` clean; `node --test test/*.test.ts` 13/13 pass.
- A real headless run against `aac6fef/laya-multilingual-mlx` on
  **WebGPU** (`--backend webgpu --headless --episodes 1 --steps 10`)
  completed end to end: agent loaded in 2.0 s, `predict()` accepted the
  compact prompt's `choice`+`noul` questions, the dino survived all 10
  ticks with 0 interventions.
- **Not yet done**: an interactive terminal session (this was built and
  checked in a sandbox without an attached TTY), a real CPU/MLX timing
  run (only WebGPU was exercised here), and a longer real-model run to
  see the shield actually intervene against real model output. If you're
  the first to run it interactively or gather more numbers, treat that as
  the real verification and update this section, mirroring
  `snake-terminal/README.md`'s "Measured" table.

## Limits

- No `export` (MP4/GIF) subcommand. `--record` writes `laya-dino-v1`
  JSONL frames, analogous to the other games' `--record`, with no
  existing consumer tool yet.
- Jump duration, reaction window, obstacle spacing/kind weights, and the
  speed ramp were chosen to produce a playable, recognizable game, not
  tuned against the real Chrome dino game's actual constants — a
  reasonable starting point, not a faithful clone of its exact physics.
