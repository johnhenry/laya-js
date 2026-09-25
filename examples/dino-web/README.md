# Laya Dino Run (WebGPU, browser)

The [Dino Run demo](../dino-terminal) in the browser. The game, reactive
lookahead planner, shield and the exact 3-question compact prompt are the
same code as the terminal demo: `../dino-terminal/src/core`, imported
relatively (that directory has no Node imports, so it bundles cleanly for
the browser — same arrangement as `snake-web`/`flappy-web`/`checkers-web`/
`tetris-web`). The model runs through `@johnhenry/laya` on WebGPU in f16.
The board is a DPR-aware canvas with three height bands (sky, air, ground)
so pterodactyl height and the dino's jump/duck state read at a glance; the
side panel follows the terminal layout.

- **Three actions**: JUMP, DUCK, RUN — richer than Flappy Bird's binary
  FLAP/NOFLAP. A cactus needs a jump, a low pterodactyl needs a duck
  (jumping into one is fatal), a high pterodactyl only punishes jumping.
- **No prediction while airborne**: the jump arc is uncontrollable once
  started, so no decision is requested mid-jump — the panel shows
  "AIRBORNE" instead of a probability list during those ticks.
- **Side panel**: the three action probabilities (the proposed action is
  marked), the executed action with a `SHIELD` tag when the shield
  overrode the model, collision risk, whether the nearest obstacle is a
  low one, inference ms, a live decisions/s counter, input tokens, shield
  interventions, deaths and steps.
- **Controls**: Space pauses and resumes; R resets to the next seed. "Max
  speed" moves as soon as each inference completes. "Shield off" executes
  raw top-1 and stops at the first death until you press R or Space, same
  as the terminal demo's `--unassisted`.

This game is JS-original (no Python reference) — see
`../dino-terminal/README.md` for the full story of a real bug (the
reactive coast letting go of `DUCK` one tick too early) found by
simulating real play *before* writing the formal test suite, applying the
lesson from Flappy Bird's own shipped-then-fixed bug directly.

## Run

```bash
npm run dev -w @johnhenry/example-dino-web   # bun build -> dist/, then serve http://localhost:5178/
npm run build -w @johnhenry/example-dino-web
npm start -w @johnhenry/example-dino-web
```

Open <http://localhost:5178/>, pick a checkpoint (Multilingual is the
default and the smallest) and press **Load & play**. The weights (≈644 MB)
download once from huggingface.co into the Cache API.

The build and serve scripts are shared with the playground
(`../web-playground/scripts/{build,serve}.ts`), same as the other browser
examples. The build fails if the bundle references a Node-only or native
module.

## Verified so far

- `npm run typecheck` clean.
- `bun ../web-playground/scripts/build.ts --root .` succeeds (264 KiB JS)
  and passes the build's native/Node-module leak check.
- Structural check only (no browser in this sandbox): served the built
  `dist/`, confirmed every DOM id referenced from `main.ts` exists in the
  built HTML, and `node --check`ed the bundle for syntax validity.
- **Not yet done**: any actual interactive/visual browser session — loading
  a real checkpoint, watching the dino play, clicking the controls. If
  you're the first to run this in a real browser, treat that as the real
  verification and add a "Measured" section here with real numbers.

## Limits

- WebGPU only. There is no CPU fallback in this demo, same reasoning as
  Snake's and Flappy Bird's — CPU-backend latency would not be a game in
  a browser.
- The Cache API is per origin, so locally weights cached by the other
  demos (ports 5173-5177) are not reused here (port 5178). On GitHub Pages
  all demos share the `johnhenry.github.io` origin and one cached copy.
