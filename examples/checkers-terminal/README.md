# Laya Checkers (terminal: Node and Bun)

A real game of American-style checkers, driven by real Laya predictions from
[`@johnhenry/laya`](../../packages/laya) on **MLX**, **WebGPU** (Dawn) or the
CPU reference, in a truecolor ANSI terminal UI with no dependencies. Built in
the architecture of [`examples/snake-terminal`](../snake-terminal): a
deterministic, Node-free `core/` (game rules, a legal/compliant move
classifier, Laya prompt-building, a pure shield), then a thin terminal
consumer.

**This game is JS-original**, like [`flappy-terminal`](../flappy-terminal) —
there is no Python reference implementation to port from or match. Unlike
Snake and Flappy Bird, which are single-player (the model plays against the
game itself), Checkers needs an opponent: this demo supports three
interchangeable seats — a real Laya policy, a zero-inference scripted bot,
or a human typing at the keyboard — chosen independently per side.

## The rules

8x8 board, single-square kings (no flying kings), **mandatory capture** (if
any capture is available anywhere on the board, only capture moves are
legal that turn) and **multi-jump chains** (a piece that captures must keep
capturing with that same piece if another capture is immediately available
from its new square). A man reaching the back row is crowned king
immediately — **including mid-chain, where kinging stops the chain even if
a further jump would otherwise have been available**, the standard rule
(verified deliberately, not simplified away — see `test/moves.test.ts`'s
"kinging stops the chain" test). The game ends when the player to move has
no legal hops.

## The engine

`core/moves.ts`'s `legalHops` enumerates every geometrically legal hop (a
simple step or one capture jump) for the player to move, ignoring mandatory
capture — the "legal" tier, Checkers' analog of Snake's per-direction
`legal` flag. `compliantHops` (the "safe" analog) filters that down to what
mandatory capture / an in-progress chain actually permits this turn.
`core/game.ts`'s `applyHop` is where the chain/kinging interaction lives,
documented in detail in its header comment. One **hop** is one decision
point — a simple move or a single capture jump — never a whole multi-jump
chain collapsed into one option; `core/session.ts`'s turn loop is
deliberately chain-unaware, running the identical decide→apply cycle every
time regardless of whether the same player is being asked again.

`core/policy.ts`'s `buildPrompt` shows the model a `choice` question whose
`criteria` covers **every legal hop this turn**, including ones that violate
mandatory capture (tagged "Illegal: ...") — the variable-cardinality analog
of Snake showing all 4 directions including blocked ones. The shield then
restricts an unsafe top-1 to the compliant subset's argmax, the same rule
as Snake's and Flappy Bird's.

## The three opponents

`--red`/`--black` each independently pick `laya`, `bot`, or `human`
(default: red is `laya`, black is `bot`):

- **`laya`**: a real `LayaPolicy` decision, shielded.
- **`bot`**: zero-inference — prefers a capture (an arbitrary but seeded,
  reproducible tie-break) when one is compliant, otherwise a random
  compliant hop.
- **`human`**: prints the numbered list of this turn's legal hops (by
  notation, e.g. `1. c3xd4`) and reads a line of input — a plain numbered
  menu, not chess notation typing, kept as bare-bones as Snake's arrow keys.

## Run

The default model is `aac6fef/laya-multilingual-mlx`, loaded **offline**
from the local Hugging Face cache, same convention as Snake/Flappy Bird.
Only loaded at all if at least one seat is `laya`.

```bash
npm start -w @johnhenry/example-checkers-terminal                           # laya (red) vs bot (black)
npm start -w @johnhenry/example-checkers-terminal -- --red bot --black bot  # spectate two bots
npm start -w @johnhenry/example-checkers-terminal -- --red human --black bot
npm run bench -w @johnhenry/example-checkers-terminal                       # 4 headless games, bot vs bot
# directly:
node --conditions=source src/cli.ts --backend cpu --headless --episodes 1 --red laya --black bot --steps 3
```

Options: `--model <id|dir>`, `--backend auto|mlx|webgpu|cpu`,
`--dtype f16|f32`, `--prompt compact|detailed`, `--optimize`, `--red`/
`--black laya|bot|human`, `--seed` (the bot's tie-break RNG; round r uses
seed + r - 1), `--pace` (turns/s when spectating, default 2), `--max-speed`,
`--duration`, `--steps`, `--record run.jsonl`, `--headless`,
`--episodes N` (bot/laya seats only — a human seat needs a real display),
`--online`. Full list: `--help`. Ctrl-C quits.

## Tests (`npm test` / `npm run test:bun`)

- `test/moves.test.ts` — the move-generation edge-case checklist: no legal
  moves (no pieces / fully blocked), off-board and occupied capture
  landings, multi-piece simultaneous mandatory capture, branching captures
  and chain continuations, **kinging stops the chain**, a near-miss chain
  that correctly keeps going, turn-boundary state clearing, a shield
  override mid-chain, and board-edge geometry (no wraparound). 18 cases.
- `test/game.test.ts` — the three actors, `CheckersSession`'s turn loop and
  auto-round-advance, and **full bot-vs-bot games completing cleanly with a
  winner across 8 seeds** (the strongest general-correctness signal, same
  spirit as Snake's "12 seeds of arbitrary shielded play reach a full
  board"). 7 cases.

**25 cases total, 0 skipped.** No Python reference exists, so there's no
fixture-parity tier, same as Flappy Bird.

## Verified so far

- `npm run typecheck` clean; `node --test test/*.test.ts` 25/25 pass.
- A real headless CLI run, `--backend cpu --episodes 5 --red bot --black
  bot --steps 500`, via the actual `cli.ts` (not just the test suite):
  5 games, all completed (47-98 turns each), 3 red wins / 2 black wins, 0
  interventions (expected — the bot never proposes a non-compliant hop).
- A short real-inference run (`--backend cpu --red laya --black bot --steps
  3`) to confirm the model actually accepts the checkers prompt shape and
  the shield runs against real output, same spirit as the other two demos'
  CPU smoke tests.
- **Not yet done**: an interactive terminal session with a human seat (this
  was built and checked in a sandbox without an attached TTY), and a real
  MLX/WebGPU timing run. If you're the first to play it, treat that as the
  real verification and add measured numbers here.

## Limits

- No flying kings, no forced-maximum-capture rule (either legal capture may
  be chosen when multiple are available, not necessarily the longest chain)
  — a deliberate simplification to the more common American ruleset rather
  than international draughts.
- No `--unassisted` — see the comment at the top of `cli.ts` for why.
- `--record` writes `laya-checkers-v1` JSONL, analogous to Snake's/Flappy
  Bird's `--record`, with no existing consumer tool yet.
