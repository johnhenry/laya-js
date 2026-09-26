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

- **Decisions are per gravity STEP, with one last chance before locking** —
  see `../tetris-terminal/README.md`'s explanation of why; there's nothing
  browser-specific about this, it's the shared core's design. Every frame
  drawn is a real decision (rotation, direction, distance), not an
  interpolated one — the old "animate an already-decided placement" overlay
  is gone; `session.stepDecide()`/`stepAdvance()` runs once per row and the
  canvas draws immediately. Hold **↓** to speed up the pacing between
  steps — each step now costs real inference time, so this is no longer a
  flat sleep on top of nothing.
- **Pace is gravity-steps/s now, not pieces/s** — a piece takes many steps
  (roughly 10-20x more `predict()` calls per piece than the old per-piece
  design; each call's questions are much smaller, which partially offsets
  it, but expect play to feel slower per piece overall).
- **Every question carries a real, grounded hint, not a bare label** —
  shared core, so this is identical to the terminal demo: each rotation
  option reports whether it actually fits at the piece's current column,
  each direction option reports how many columns are actually free that
  way, and the state text reports the real per-column height profile, not
  just one stack-height number. See `../tetris-terminal/README.md`'s "The
  engine" section for why this was added.
- **Side panel**: active piece and next-5 preview, the executed placement
  (with a `SHIELD` tag when the shield overrode the model), topping-out
  risk, line-clear likelihood, inference ms, live steps/s and pieces/s
  counters, input tokens, shield interventions, deaths and pieces placed.
- **Controls**: Space pauses and resumes; R resets to the next seed. "Max
  speed" moves as soon as each step's inference completes (skips the pacing
  delay, not the decisions -- every step is still real). "Shield off"
  executes the model's raw lock choice and stops at the first block-out
  until you press R or Space, same as the terminal demo's `--unassisted`.
- A warmup of a handful of steps on a fixed seed compiles the GPU
  pipelines, as the terminal demo does.

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
- `bun ../web-playground/scripts/build.ts --root .` succeeds (266 KiB JS)
  and passes the build's native/Node-module leak check -- confirmed again
  after the per-step redesign (`session.stepDecide()`/`stepAdvance()`
  replaced the old `decide()`/`advance()`/`animateDrop()` call shape;
  nothing here imports anything new that could leak a Node/native module).
- Structural check only (no browser in this sandbox): served the built
  `dist/`, confirmed every DOM id referenced from `main.ts` exists in the
  built HTML (including the new `piece-rate` stat), and `node --check`ed
  the bundle for syntax validity.
- The per-step redesign itself was verified for real on the **terminal**
  demo (`tetris-terminal`'s own headless MLX runs -- see that README) since
  this sandbox has no browser; the shared `core/` is identical code, and
  this demo's own rendering/pacing changes were re-typechecked and rebuilt
  as above, but not run interactively.
- **Not yet done**: any actual interactive/visual browser session — loading
  a real checkpoint, watching Laya place pieces, clicking the controls. If
  you're the first to run this in a real browser, treat that as the real
  verification and add a "Measured" section here with real numbers.

## Limits

- WebGPU only. There is no CPU fallback in this demo. `tetris-terminal`'s
  own CPU-backend smoke test was abandoned after several minutes with no
  result even at the OLD one-call-per-piece rate (Tetris's prompt is far
  larger than the other games'); the new per-step design does roughly
  10-20x more calls per piece, so CPU is even less practical here now, not
  something you'd want as a browser fallback regardless.
- The Cache API is per origin, so locally weights cached by the other
  demos (ports 5173-5176) are not reused here (port 5177). On GitHub Pages
  all demos share the `johnhenry.github.io` origin and one cached copy.
