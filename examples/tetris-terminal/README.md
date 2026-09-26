# Laya Tetris (terminal: Node and Bun)

A real game of Tetris, driven by real Laya predictions from
[`@johnhenry/laya`](../../packages/laya) on **MLX**, **WebGPU** (Dawn) or the
CPU reference, in a truecolor ANSI terminal UI with no dependencies. Built in
the architecture of [`examples/snake-terminal`](../snake-terminal): a
deterministic, Node-free `core/` (game rules, a legal/safe placement
classifier, Laya prompt-building, a pure shield), then a thin terminal
consumer.

**This game is JS-original**, like [`flappy-terminal`](../flappy-terminal)
and [`checkers-terminal`](../checkers-terminal) — there is no Python
reference implementation to port from or match.

## Decisions are per gravity STEP, with one last chance before locking

Earlier versions of this demo made one decision per PIECE (enumerate every
reachable final placement, one `choice` picks among them) and only
*animated* the fall afterward, cosmetically. This version makes the
decision real, matching the granularity of actually playing Tetris: **every
time the active piece moves down a row is a step**, and at each step the
model picks one of 4 rotations (0°/90°/180°/270°), one of 3 horizontal
directions (none/left/right) and a distance (0-9 columns, clamped to
whatever's actually free) — three small, independent `choice` questions,
not a lookup into a placement table. When the piece can no longer descend,
it gets **exactly one more** such decision (now also asking the `risk`/
`clears` signals) before that position locks — a bounded "lock delay" /
"extended placement" in [Tetris Wiki](https://tetris.wiki/Lock_delay) /
[Hard Drop wiki](https://harddrop.com/wiki/Lock_delay) terminology. This is
a deliberately bounded reading of that lineage, not a claim of exact
Game Boy/NES parity (those had no formal lock delay at all, per the same
sources) or full Guideline "Infinity" behavior (unbounded resets, which
would risk an adversarial model stalling forever) — one extra decision,
never a repeating cascade of them.

**Every frame is a real decision, not an interpolated one.** `cli.ts` calls
`session.stepDecide()`/`stepAdvance()` once per row and draws immediately;
there's no more pre-baked frame list to play back, and no more "preview the
default rotation before snapping" — every rotation shown was actually
chosen. Holding **↓** still speeds up the pacing between steps, but each
step now costs real inference time, so the felt cadence is `max(0, target
- elapsed)`, not a flat sleep stacked on top of real latency.

**This costs roughly 10-20x more `predict()` calls per piece** (bounded by
how many rows it descends, plus one — so a piece that lands high, near a
tall stack, is actually *cheaper* than one that falls the full board).
Each call's prompt is much smaller now (3-5 fixed small questions vs. a
`choice` over up to ~34 placement criteria before), which partially
offsets it, but wall-clock cost per piece is still substantially higher on
every backend. This is inherent to real per-step control, not a
regression to fix.

## The engine

7 standard tetrominoes (I/O/T/S/Z/J/L). `pieces.ts` only stores
geometrically **distinct** rotation shapes (O has 1, I/S/Z have 2, T/J/L
have 4); `resolveRotation(kind, label)` lets the model freely choose any of
the 4 absolute labels regardless of piece — it maps onto the correct
(possibly repeated) shape because each kind's distinct-rotation count
always evenly divides 4. `spawnColFor(kind)` centers a piece's default
orientation on the board (e.g. I spawns at columns 3-6), the fixed spot a
piece actually enters play at — a genuinely new concept this version
needed, since the old per-piece design tried every column and never needed
just one.

**Game over (block-out)** is now `canSpawn(board, kind)`: does the piece's
literal default spawn configuration collide, right now. This replaced the
old, more permissive "does *any* rotation/column combination fit on row 0
somewhere" check — a real, deliberate behavior change, and a more
realistic one: a piece actually enters play in one fixed orientation and
column, not by trying every hypothetical placement, so that's the
configuration whose collision is a genuine block-out.

**The shield only ever guards the final lock decision**, and its
evaluation set (`optionsAtRow`) is **local**: everything reachable via
rotate+shift *alone* from the piece's current position (via
`sweepColumns`, which walks outward and stops at the first collision each
way — this is what keeps the local set from including geometrically
disconnected shelves the piece could never actually have slid to), not a
global re-search from spawn. `safe` still means the stack height, measured
**after** line-clear resolution, stays within a 4-row margin of the
ceiling — but it's a real, weaker guarantee than before: the shield can fix
a bad final orientation/column, not rescue a trajectory that already
steered too high. That's the honest cost of real per-step control, not a
bug.

**Empty-safe-set handling still follows Flappy Bird's precedent, not
Checkers'** — and is *routine* here, not rare: near the top of a real game,
it's normal for every locally-reachable option to breach the margin while
the game is very much still going (block-out is a separate, narrower
condition — see above). The shield trusts the model's own proposed choice
rather than substituting a different, unshielded one it didn't ask for —
see the comment in `core/policy.ts`.

Piece order: a standard **7-bag** randomizer (a shuffled permutation of all
7 kinds per bag, so no piece is ever absent for more than ~12 spawns),
seeded with a small local mulberry32-style PRNG — deterministic within
this JS implementation, not a Python-parity port (see `core/rng.ts`).

## Run

The default model is `aac6fef/laya-multilingual-mlx`, loaded **offline**
from the local Hugging Face cache, same convention as the other three demos.

```bash
npm start -w @johnhenry/example-tetris-terminal                 # node, backend auto (mlx on Apple silicon)
npm run start:bun -w @johnhenry/example-tetris-terminal         # same under Bun
npm run dev -w @johnhenry/example-tetris-terminal                # --max-speed
npm start -w @johnhenry/example-tetris-terminal -- --backend webgpu --max-speed
npm run bench -w @johnhenry/example-tetris-terminal              # 4 headless episodes, up to 200 pieces each, seeds 101-104
# directly:
node --conditions=source src/cli.ts --backend mlx --headless --episodes 1 --steps 3
```

Keys: **Space** pause, **↓** hold to speed up the current step's pacing
(never instant), **R** next seed, **Q** or Ctrl-C quit.

Options: `--model <id|dir>`, `--backend auto|mlx|webgpu|cpu`,
`--dtype f16|f32`, `--prompt compact|detailed`, `--optimize`, `--seed`,
`--pace` (gravity-steps/s when spectating, default 3 — **not** pieces/s;
a piece now takes many steps), `--max-speed`, `--duration`,
`--steps` (pieces locked, per episode with `--episodes`), `--unassisted`,
`--record run.jsonl` (`laya-tetris-v2`), `--headless`, `--episodes N`,
`--no-alt-screen`, `--online`. Full list: `--help`.

## Tests (`npm test` / `npm run test:bun`)

`test/game.test.ts`: piece-shape correctness (every rotation is exactly 4
cells, no two rotations of the same piece share a cell-set), `resolveRotation`'s
invariant (always resolves to a real shape, geometrically-equivalent labels
map onto each other), `spawnColFor` centering, placement bounds at both
board edges for every kind, a 4-line "Tetris" clear verified against the
**post**-clear board, `canSpawn`'s exact block-out condition, a real
(non-clearing) block-out reached via an actual `applyPlacement()` call —
not just a directly constructed pathological board — 7-bag determinism and
completeness over 1,000+ bags, `optionsAtRow`'s local safe/unsafe
decoration on a critically tall vs. a mostly-empty board, a regression test
for a real bug caught during design review (a naive full-row scan would
have let the shield "jump" a wall to a geometrically disconnected but
same-row-resting shelf — `sweepColumns` fixes this), the per-step engine's
rotate-before-shift ordering (an illegal rotation is rejected using the
*current* column, no wall-kick, even if the shift alone would have made
room), the lock-time shield's guard/override/empty-safe-set behavior
against the new per-step decision shape, and a full-game smoke-play test
(a simple lowest-resulting-height heuristic, using the still-available
global `legalPlacements`, not the shield's local view) across 6 seeds.
**26 cases, 0 skipped** — no Python reference exists, so there's no
fixture-parity tier.

## Verified so far

- `npm run typecheck` clean; `node --test test/*.test.ts` 26/26 pass.
- A real headless run against `aac6fef/laya-multilingual-mlx` on **MLX**
  (`--backend mlx --headless --episodes 1 --steps 2`) completed end to end:
  agent loaded in 1.0 s, 2 pieces locked over 40 real gravity-step
  `predict()` calls (20 steps/piece — each a real rotation/direction/
  distance decision, not an interpolated frame), inference p50 43 ms,
  0 interventions. A longer run (`--episodes 2 --steps 60`, no cap hit)
  reached real block-out (`alive: false`, via the new `canSpawn` check) on
  both episodes after 15 and 20 pieces respectively, with no crashes;
  `--unassisted` (shield off) was also exercised for 8 pieces/139 steps.
  Per this session's own guidance, **the CPU backend was not spot-checked**
  here (already known much slower for this game even at one call per
  piece; now doing 10-20x more calls per piece would make that far worse)
  — WebGPU is untested in this specific sandbox pass but was verified for
  the prior per-piece design and shares the same `predict()` call shape.
- **Not yet done**: an interactive terminal session (this was built and
  checked in a sandbox without an attached TTY). If you're the first to run
  it interactively, treat that as the real verification and update this
  section with real numbers, mirroring `snake-terminal/README.md`'s
  "Measured" table.

## Limits

- No wall kicks (a rotation either fits or it doesn't, checked against the
  piece's *current* column before any shift — there's no SRS kick-table
  attempt to nudge a rotation into a tighter space), and no hold-piece
  mechanic. Both are real simplifications from tournament Tetris, chosen to
  keep the per-step model clean.
- The shield is now a real, weaker guarantee than the old per-piece one: it
  can fix a bad final orientation/column at lock time, not rescue a
  trajectory that already steered too high (see "The engine" above).
- Real per-step control costs roughly 10-20x more `predict()` calls per
  piece than the old per-piece design — expect noticeably slower play,
  especially headless benchmarks (see above).
- No `export` (MP4/GIF) subcommand. `--record` writes `laya-tetris-v2`
  JSONL (bumped from `v1` — a new `"step"` record per gravity-step decision,
  `"piece"` now a per-piece summary written once it locks), analogous to
  the other games' `--record`, with no existing consumer tool yet.
- The 4-row safety margin and 7-bag lookahead depth (5) were chosen to
  produce a playable, reasonably-forgiving game, not tuned against any
  reference — a starting point, not a tournament-accurate constant.
