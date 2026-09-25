# Laya Tetris (WebGPU, browser)

The [Tetris demo](../tetris-terminal) in the browser. The game, placement
enumeration, safety-margin shield, and the exact prompt are the same code
as the terminal demo: `../tetris-terminal/src/core`, imported relatively
(that directory has no Node imports, so it bundles cleanly for the
browser — same arrangement as `snake-web`/`flappy-web`/`checkers-web`). The
model runs through `@johnhenry/laya` on WebGPU in f16. The board is drawn
on a DPR-aware canvas with the classic per-piece guideline colors (I cyan,
O yellow, T purple, S green, Z red, J blue, L orange); the side panel
follows the terminal layout.

- **Decisions are per piece, not per tick** — see
  `../tetris-terminal/README.md`'s explanation of why; there's nothing
  browser-specific about this, it's the shared core's design.
- **The fall is animated, the decision isn't** — once the model picks a
  placement, the piece visibly descends from the spawn row to its resting
  row (as an overlay drawn on top of the locked board, not yet part of the
  real game state) instead of snapping there instantly. Hold **↓** to
  speed the fall up (a soft drop, `~12 ms`/row instead of `~45 ms`) — there
  is no hard-drop key, so it's never literally instant.
- **Side panel**: active piece and next-5 preview, the executed placement
  (with a `SHIELD` tag when the shield overrode the model), topping-out
  risk, line-clear likelihood, inference ms, a live decisions/s counter,
  input tokens, shield interventions, deaths and pieces placed.
- **Controls**: Space pauses and resumes; R resets to the next seed. "Max
  speed" moves as soon as each inference completes (and skips the fall
  animation entirely). "Shield off" executes raw top-1 and stops at the
  first block-out until you press R or Space, same as the terminal demo's
  `--unassisted`.
- A warmup of 6 decisions on a fixed seed compiles the GPU pipelines, as
  the terminal demo does.

## Run

```bash
npm run dev -w @johnhenry/example-tetris-web   # bun build -> dist/, then serve http://localhost:5177/
npm run build -w @johnhenry/example-tetris-web
npm start -w @johnhenry/example-tetris-web
```

Open <http://localhost:5177/>, pick a checkpoint (Multilingual is the
default and the smallest) and press **Load & play**. The weights (≈644 MB)
download once from huggingface.co into the Cache API.

The build and serve scripts are shared with the playground
(`../web-playground/scripts/{build,serve}.ts`). The build fails if the
bundle references a Node-only or native module.

## Verified so far

- `npm run typecheck` clean.
- `bun ../web-playground/scripts/build.ts --root .` succeeds (263 KiB JS)
  and passes the build's native/Node-module leak check.
- Structural check only (no browser in this sandbox): served the built
  `dist/`, confirmed every DOM id referenced from `main.ts` exists in the
  built HTML, and `node --check`ed the bundle for syntax validity.
- **Not yet done**: any actual interactive/visual browser session — loading
  a real checkpoint, watching Laya place pieces, clicking the controls. If
  you're the first to run this in a real browser, treat that as the real
  verification and add a "Measured" section here with real numbers.

## Limits

- WebGPU only. There is no CPU fallback in this demo. `tetris-terminal`'s
  own CPU-backend smoke test was abandoned after several minutes with no
  result (Tetris's prompt is far larger than the other games' -- up to
  ~34 `choice` criteria per piece), while WebGPU completed the same
  smoke test in 1.5 s with ~591 ms per decision -- so CPU is presumably
  correct but meaningfully slower here, not something you'd want as a
  browser fallback regardless.
- The Cache API is per origin, so locally weights cached by the other
  demos (ports 5173-5176) are not reused here (port 5177). On GitHub Pages
  all demos share the `johnhenry.github.io` origin and one cached copy.
