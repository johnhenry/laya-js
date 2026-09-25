# Laya Flappy Bird (terminal: Node and Bun)

A real Flappy Bird, driven by real Laya predictions from
[`@johnhenry/laya`](../../packages/laya) on **MLX**, **WebGPU** (Dawn) or the
CPU reference, in a truecolor ANSI terminal UI with no dependencies. Built in
the architecture of [`examples/snake-terminal`](../snake-terminal): a
deterministic, Node-free `core/` (game rules, a two-action lookahead safety
classifier, Laya prompt-building, a pure shield), then a thin terminal
consumer.

**This game is JS-original.** Unlike Snake (a faithful port of laya-mlx's
`snake/` demo, with a Python recording to prove byte-for-bit parity against),
there is no Python reference implementation of this Flappy Bird to port from
or match. The architecture mirrors Snake's on purpose — same shield shape,
same CLI flag family, same terminal UI approach — but the game rules,
physics, prompt text and RNG are original to this repo. See "What's
different from Snake" below for the specific, deliberate divergences.

## The game

Tick-based: the bird falls under gravity; flapping sets its vertical velocity
to a fixed upward impulse (not additive — a real flap always gives the same
kick, matching the original game's feel). Pipes scroll left at a constant
speed with a gap of fixed height, placed at a seeded-random row each spawn
(`core/rng.ts` — a small mulberry32 PRNG, **not** a Python-parity port; it
exists only so a given seed reproduces the same run for tests and
`--record`/replay). Death is ground, ceiling, or a pipe outside its gap —
decided purely by game rules in `core/game.ts`, no model involved.

Every tick, `core/game.ts`'s `moves()` classifies both `FLAP` and `NOFLAP`:
both are always **legal** (there's no illegal input in Flappy Bird); an
action is **safe** if a multi-tick lookahead — simulate the action now, then
coast with no further flaps for up to 20 ticks — never collides. A single
tick isn't enough to classify safety meaningfully (falling one tick rarely
kills you near the top of the screen), so this is a real forward simulation,
not a 1-step check.

## What's ported from Snake vs. original here

| Concept | Snake (`snake-terminal/src/core`) | Here (`src/core`) |
|---|---|---|
| Terminal-state detection | `SnakeGame.step()`, pure rules | `FlappyGame.step()`, pure rules |
| Legal/safe classification | `moves()`: 4 fixed directions, Hamiltonian-cycle planner | `moves()`: 2 fixed actions, multi-tick lookahead simulation |
| Prompt shape | 1 `choice` (4 directions) + `risk`/`food` `noul` | 1 `choice` (FLAP/NOFLAP) + `risk`/`aligned` `noul` |
| Shield | `executed = guarded && !safe.includes(proposed) ? argmax(safe) : proposed`; throws if `safe` is ever empty (a real bug — the cycle planner guarantees a safe move exists) | Same rule, **except**: an empty safe set executes the model's raw choice instead of throwing — Flappy Bird has no such guarantee, and a bad flap history can make death genuinely unavoidable. See the comment in `core/policy.ts`. |
| RNG | `core/pyrandom.ts`: MT19937 + CPython `init_by_array`, byte-exact with `laya-mlx`'s Python for a given seed | `core/rng.ts`: mulberry32, deterministic only within this JS implementation — no Python reference to match |
| Test tiers | pure-logic + Python-fixture-parity (`prompt.test.ts`, `agent.test.ts` cross-language deltas) | pure-logic only (`test/game.test.ts`, 15 cases, 0 skipped) — no parity tier, since there's nothing to compare against |

## Run

The default model is `aac6fef/laya-multilingual-mlx`, loaded **offline** from
the local Hugging Face cache, same convention as Snake. Download it once with
`hf download aac6fef/laya-multilingual-mlx`, or pass `--online`.

```bash
npm start -w @johnhenry/example-flappy-terminal                 # node, backend auto (mlx on Apple silicon)
npm run start:bun -w @johnhenry/example-flappy-terminal         # same under Bun
npm run dev -w @johnhenry/example-flappy-terminal                # --max-speed
npm start -w @johnhenry/example-flappy-terminal -- --backend webgpu --max-speed
npm run bench -w @johnhenry/example-flappy-terminal              # 4 x 1200 uncapped ticks, seeds 101-104
# directly:
node --conditions=source src/cli.ts --backend cpu --headless --episodes 1 --steps 5
```

Use a terminal of at least 130 × 39 (the board defaults to 40 x 20 cells).
Keys: **Space** pause, **↑/↓** (or +/−) speed, **R** next seed, **Q** or
Ctrl-C quit.

Options: `--model <id|dir>`, `--backend auto|mlx|webgpu|cpu`,
`--dtype f16|f32`, `--prompt compact|detailed`, `--optimize`,
`--width/--height`, `--seed`, `--gravity`, `--flap-impulse`, `--pipe-speed`,
`--gap-height`, `--fps` (default 20), `--max-speed`, `--duration`, `--steps`,
`--unassisted`, `--record run.jsonl`, `--headless`, `--episodes N`,
`--no-alt-screen`, `--online`. Full list: `--help`.

`--episodes N --headless` runs N uncapped episodes (seeds `seed … seed+N−1`,
at most `--steps` each, default 1200) and prints a JSON summary: ticks/s,
deaths, interventions, inference p50/p95/p99 per episode and overall.

## Tests (`npm test` / `npm run test:bun`)

`test/game.test.ts`: RNG determinism, ground/ceiling/pipe collision
(constructed scenarios, not just random play), the lookahead safety
classifier (including the "both actions unsafe, no throw" case), the shield
with a stub agent (guarded restriction + intervention reporting, raw
execution when unguarded), invalid-model-probability rejection, seed
reproducibility, and a multi-seed heuristic-play smoke test. **15 cases, 0
skipped** — no Python reference exists, so there's no fixture-parity tier
(contrast with `snake-terminal`'s `prompt.test.ts`/`agent.test.ts`).

## Verified so far

- **A real bug was found and fixed here, from actual play, not from the test
  suite.** The lookahead's "coast" phase originally simulated NOFLAP forever
  after the tested action, and since nothing stops gravity, that path
  eventually hits the ground regardless of what the tested action was —
  `NOFLAP` came back "unsafe" almost every tick, so the shield flapped
  constantly and shot the bird into the ceiling within the first ~20 ticks,
  before a single pipe ever scrolled into view. The 15 pure-logic tests all
  passed the whole time; none of them exercised a long enough play sequence
  to notice. Fixed by making the coast reactive (flap only when actually
  close to the ground) instead of blindly never-flap-again — see the
  comment on `#willCollideWithin` in `core/game.ts`. Two tests needed
  sharper scenarios to still be meaningful after the fix, and one new test
  was added for the "mild fall recovers in time" case the bug had made
  impossible to distinguish from "unavoidable death."
- `npm run typecheck` clean; `node --test test/*.test.ts` 16/16 pass.
- A real headless run against `aac6fef/laya-multilingual-mlx` on the **CPU**
  reference backend (`--backend cpu --headless --episodes 1 --steps 5`)
  completed end to end: agent loaded, `predict()` accepted the compact
  prompt's `choice`+`noul` questions, the shield ran with 0 interventions
  over 5 ticks, and a valid JSON summary was produced. CPU inference was
  ~22.7 s/decision (p50) — the CPU backend is a correctness reference, not
  something to play interactively with, same caveat as Snake's.
- **Not yet done**: an interactive terminal session (this was built and
  checked in a sandbox without an attached TTY), and a real MLX/WebGPU
  timing run (only the CPU backend was exercised here). If you're the first
  to run it interactively or on MLX/WebGPU, treat that as the real
  verification and update this section with real numbers, mirroring
  `snake-terminal/README.md`'s "Measured" table.

## Limits

- No `export` (MP4/GIF) subcommand. `--record` writes `laya-flappy-v1`
  JSONL frames (board + decision + stats per tick), analogous to Snake's
  `--record`, but there is no existing tool that consumes it yet.
- The lookahead horizon (20 ticks) and physics defaults (gravity, flap
  impulse, pipe speed/gap/spacing) were chosen to produce a playable,
  survivable-with-good-play game, not tuned against any reference — treat
  them as a reasonable starting point, adjustable via CLI flags.
