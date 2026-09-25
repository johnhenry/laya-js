# Laya Checkers (WebGPU, browser)

The [Checkers demo](../checkers-terminal) in the browser. The rules,
mandatory-capture/chain/kinging state machine, shield, and the three
interchangeable opponent actors (`LayaActor`/`BotActor`/`HumanActor`) are
the same code as the terminal demo: `../checkers-terminal/src/core`,
imported relatively (that directory has no Node imports, so it bundles
cleanly for the browser — same arrangement as `snake-web`/`flappy-web`).
The board is a click-to-move canvas instead of the terminal's numbered
menu; everything else (rules, shield, prompt) is identical.

**Unlike Snake and Flappy Bird, a Laya seat here is optional.**
Bot-vs-bot, human-vs-bot, and human-vs-human all play fully without
WebGPU — only picking `laya` for a seat needs it, and the app tells you so
rather than blocking the whole page behind a WebGPU requirement.

- **Seats**: pick Red and Black independently (Laya / Bot / Human) before
  starting. Loading a checkpoint is skipped entirely if neither seat is
  Laya.
- **Human play**: click one of your highlighted pieces, then click a
  highlighted destination square. Squares light up only for this turn's
  actually-legal (mandatory-capture-compliant) hops — the same set the
  terminal demo's numbered menu offers, just shown visually instead of as
  text.
- **Side panel**: whose turn, the last move (with a `SHIELD` tag when the
  shield overrode a Laya seat's raw choice), material-at-risk, inference
  ms, decisions/s, input tokens, shield interventions, and the win tally.
- **Controls**: "New game" restarts with the same seat configuration (to
  change seats, reload the page — see Limits); "Max speed" removes the
  pacing delay between non-human turns; "Pace" sets turns/s when
  spectating bot/Laya play (1-10, default 2).

## Run

```bash
npm run dev -w @johnhenry/example-checkers-web   # bun build -> dist/, then serve http://localhost:5176/
npm run build -w @johnhenry/example-checkers-web
npm start -w @johnhenry/example-checkers-web
```

Open <http://localhost:5176/>, choose seats, and press **Play**. If either
seat is Laya, the weights (≈644 MB) download once from huggingface.co into
the Cache API.

The build and serve scripts are shared with the playground
(`../web-playground/scripts/{build,serve}.ts`), same as the other browser
examples. The build fails if the bundle references a Node-only or native
module.

## Verified so far

- `npm run typecheck` clean.
- `bun ../web-playground/scripts/build.ts --root .` succeeds (265 KiB JS)
  and passes the build's native/Node-module leak check.
- Structural check only (no browser in this sandbox): served the built
  `dist/`, confirmed every DOM id referenced from `main.ts` exists in the
  built HTML, and `node --check`ed the bundle for syntax validity.
- **Not yet done**: any actual interactive/visual browser session —
  clicking through a human-vs-bot game, loading a real checkpoint, watching
  Laya play. If you're the first to run this in a real browser, treat that
  as the real verification and add measured numbers here.

## Limits

- "New game" reuses the seat configuration from the last **Play** (or page
  load); to change who's playing which side, reload the page. A fuller
  implementation would re-show the seat pickers without a reload — not
  done here to keep the control surface simple.
- The Cache API is per origin, so locally weights cached by the playground
  (port 5173), Snake (port 5174), or Flappy Bird (port 5175) are not reused
  here (port 5176). On GitHub Pages all demos share the
  `johnhenry.github.io` origin and one cached copy.
- No forced-maximum-capture rule, no flying kings — see
  `../checkers-terminal/README.md`'s Limits for the full ruleset notes.
