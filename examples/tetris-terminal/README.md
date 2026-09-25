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

## Decisions are per PIECE, not per tick

A real gravity-tick loop (rotate/shift/soft-drop as separate ticked
decisions) would need 5-15 `predict()` calls just to relocate one piece, for
a game where only the final resting cell set matters. Instead: a piece
spawns, `core/game.ts`'s `legalPlacements` enumerates every reachable
`(rotation, column)` via a drop-simulation, one `predict()` call picks
among them, and the whole rotate+shift+hard-drop+lock+clear sequence
resolves as a single pure `applyPlacement()` transformation — no
animated-movement concept at all, the same way `checkers-terminal` computes
a hop's resulting board directly rather than simulating a slide. This is
exactly why Checkers itself moved to per-turn decisions instead of
per-tick, applied one level further: Tetris's placement space (9-34
reachable placements depending on piece and board state) is naturally
enumerable as `choice` criteria, just like Checkers' legal-hop set.

## The engine

7 standard tetrominoes (I/O/T/S/Z/J/L), with only geometrically **distinct**
rotations generated in the first place (O has 1, I/S/Z have 2, T/J/L have
4) — built into the shape table itself, not enumerated-then-deduplicated.
`legalPlacements(board, kind)` is the complete legal set (no extra
"mandatory" layer the way Checkers has mandatory capture).

The `safe` tier — the one genuinely novel design decision in this
package, since `safe` can't just equal `legal` here (that would make the
shield a no-op): a placement is safe only if the stack height, measured
**after** line-clear resolution, stays within a 4-row margin of the
ceiling. A placement that looks tall pre-clear but completes 1-4 full rows
can legitimately duck back under the margin and count as safe, while a
"boring" non-clearing placement that stacks just as high may not — the
same structural shape as Snake's "would trap the snake" rule and Flappy
Bird's "would collide within the lookahead," just computed from a
lock+clear simulation instead of a physics simulation.

**Empty-safe-set handling follows Flappy Bird's precedent, not Checkers'.**
In Checkers, `legal` empty *is* the loss condition. In Tetris, `legalPlacements`
empty only happens when the piece can't even spawn (the real game-over
check — block-out); `safe` can be empty while `legal` is nonempty and the
game is very much still going (every reachable placement might breach the
margin while cells are open elsewhere). That's "in real trouble," not
"already lost," so the shield executes the model's raw choice instead of
throwing — see the comment in `core/policy.ts`.

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
npm run bench -w @johnhenry/example-tetris-terminal              # 4 headless episodes, 200 pieces each, seeds 101-104
# directly:
node --conditions=source src/cli.ts --backend cpu --headless --episodes 1 --steps 3
```

Keys: **Space** pause, **R** next seed, **Q** or Ctrl-C quit.

Options: `--model <id|dir>`, `--backend auto|mlx|webgpu|cpu`,
`--dtype f16|f32`, `--prompt compact|detailed`, `--optimize`, `--seed`,
`--pace` (pieces/s when spectating, default 3), `--max-speed`, `--duration`,
`--steps`, `--unassisted`, `--record run.jsonl`, `--headless`,
`--episodes N`, `--no-alt-screen`, `--online`. Full list: `--help`.

## Tests (`npm test` / `npm run test:bun`)

`test/game.test.ts`: piece-shape correctness (every rotation is exactly 4
cells, no two rotations of the same piece share a cell-set), placement
bounds at both board edges for every kind (I horizontal/vertical, T/J/L
across all 4 rotations, O's single rotation), a 4-line "Tetris" clear
verified against the **post**-clear board, a constructed
safe-empty-but-not-game-over scenario, a real (non-clearing) block-out
reached via an actual `applyPlacement()` call — not just a directly
constructed pathological board — 7-bag determinism and completeness over
1,000+ bags, the shield's guard/override/empty-safe-set behavior, and a
full-game smoke-play test (a simple lowest-resulting-height heuristic)
across 6 seeds. **19 cases, 0 skipped** — no Python reference exists, so
there's no fixture-parity tier.

## Verified so far

- `npm run typecheck` clean; `node --test test/*.test.ts` 19/19 pass.
- A real headless run against `aac6fef/laya-multilingual-mlx` on **WebGPU**
  (`--backend webgpu --headless --episodes 1 --steps 3`) completed end to
  end: agent loaded in 1.5 s, `predict()` accepted the compact prompt's
  `choice`+`noul` questions over the enumerated placement set, and the
  shield ran against real output — 3 pieces, 0 interventions, inference
  p50 591 ms. The CPU reference backend was tried first but abandoned for
  this smoke test: Tetris's prompt is far larger than Snake's/Flappy
  Bird's/Checkers' (up to ~34 `choice` criteria vs. single digits), and a
  `--steps 3` CPU run was still running after several minutes with no
  sign of finishing, so it was killed rather than left blocking — a real,
  worth-noting cost of this game's larger placement-enumeration prompt
  that the other three don't have. CPU is presumably still correct, just
  meaningfully slower here than for the other games; someone with more
  patience (or a machine without a WebGPU-limited CPU-reference-only path)
  should confirm it directly and update this note.
- **Not yet done**: an interactive terminal session (this was built and
  checked in a sandbox without an attached TTY), and a real MLX timing
  run. If you're the first to run it interactively or on MLX, treat that
  as the real verification and update this section with real numbers,
  mirroring
  `snake-terminal/README.md`'s "Measured" table.

## Limits

- No wall kicks (a rotation either fits or it doesn't — there's no SRS
  kick-table attempt to nudge a rotation into a tighter space), and no
  hold-piece mechanic. Both are real simplifications from tournament
  Tetris, chosen to keep the placement-enumeration model clean.
- No `export` (MP4/GIF) subcommand. `--record` writes `laya-tetris-v1`
  JSONL, analogous to the other games' `--record`, with no existing
  consumer tool yet.
- The 4-row safety margin and 7-bag lookahead depth (5) were chosen to
  produce a playable, reasonably-forgiving game, not tuned against any
  reference — a starting point, not a tournament-accurate constant.
