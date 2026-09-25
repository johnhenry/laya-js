# Laya Flappy Bird (WebGPU, browser)

The [Flappy Bird demo](../flappy-terminal) in the browser. The game,
lookahead safety classifier, shield and the exact 3-question compact prompt
are the same code as the terminal demo: `../flappy-terminal/src/core`,
imported relatively (that directory has no Node imports, so it bundles
cleanly for the browser — same arrangement as `snake-web`/`snake-terminal`).
The model runs through `@johnhenry/laya` on WebGPU in f16. The board is
drawn on a DPR-aware canvas; the side panel follows the terminal layout.

- **Board**: 40 x 20 cells, seed 7, same defaults as the terminal demo.
  Round r uses seed + r − 1.
- **Side panel**: the two action probabilities (FLAP/NOFLAP, the model's raw
  output; the proposed action is marked), the executed action with a
  `SHIELD` tag when the shield overrode the model, collision risk (the
  model's own `risk` answer), gap alignment (`aligned`), inference ms, a
  live decisions/s counter (rolling 60 decisions), input tokens, shield
  interventions, deaths and steps.
- **Controls**: Space pauses and resumes; ↑/↓ (or +/−) change the pace by
  2/s (1-60, default 20); R resets to the next seed. "Max speed" moves as
  soon as each inference completes. "Shield off" executes raw top-1 and
  stops at the first death until you press R or Space, same as the terminal
  demo's `--unassisted`.
- A warmup of 6 decisions on seed + 10000 compiles the GPU pipelines, as the
  terminal demo does.

This game is JS-original (no Python reference) — see
`../flappy-terminal/README.md`'s "What's ported from Snake vs. original
here" table for exactly what's shared architecture versus original content.

## Run

```bash
npm run dev -w @johnhenry/example-flappy-web   # bun build -> dist/, then serve http://localhost:5175/
npm run build -w @johnhenry/example-flappy-web
npm start -w @johnhenry/example-flappy-web
```

Open <http://localhost:5175/>, pick a checkpoint (Multilingual is the
default and the smallest) and press **Load & play**. The weights (≈644 MB)
download once from huggingface.co into the Cache API.

The build and serve scripts are shared with the playground
(`../web-playground/scripts/{build,serve}.ts`), same as `snake-web`. The
build fails if the bundle references a Node-only or native module.

## Verified so far

- `npm run typecheck` clean.
- `bun ../web-playground/scripts/build.ts --root .` succeeds (265 KiB JS)
  and passes the build's native/Node-module leak check.
- Structural check only (no browser in this sandbox): served the built
  `dist/`, confirmed every DOM id referenced from `main.ts` exists in the
  built HTML, and `node --check`ed the bundle for syntax validity.
- **Not yet done**: any actual interactive/visual browser session — loading
  a real checkpoint, watching the bird play, clicking the controls. If
  you're the first to run this in a real browser, treat that as the real
  verification and add a "Measured" section here with real numbers,
  mirroring `snake-web/README.md`'s.

## Limits

- WebGPU only. There is no CPU fallback in this demo — the CPU reference
  backend measured ~23 s/decision in `flappy-terminal`'s smoke test, which
  would not be a game in a browser either. An unsupported browser gets an
  explanation.
- The Cache API is per origin, so locally weights cached by the playground
  (port 5173) or Snake (port 5174) are not reused here (port 5175). On
  GitHub Pages all three demos share the `johnhenry.github.io` origin and
  one cached copy.
