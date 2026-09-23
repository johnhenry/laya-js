# @johnhenry/laya-router

Language and task routing across the three Laya checkpoints. This is a port of
laya-mlx `router.py`: the same routing decisions and reasons, the same
precedence and the same LRU residency, made async-safe.

```ts
import { Router } from "@johnhenry/laya-router";

const router = new Router({ mlxRepos: true, loadOptions: { backend: "auto", dtype: "f16" } });
const r = await router.predict({ message: "Mein Konto wurde zweimal belastet" }, questions);
r.routing; // { model: "multilingual", repo, reason: "Latin script but language looks like 'de', not English", detection, workflow }
router.route("I was charged twice"); // decision only; nothing is loaded
await router.preload(["english", "multilingual"]); // keep both resident
router.unload();
```

## API

- `DEFAULT_MODELS` is the bundle repo `convaiinnovations/laya` with
  subfolders, as in Python. `STANDALONE_MODELS` lists the per-checkpoint repos.
  `MLX_MODELS` lists the laya-mlx fp16 MLX exports (`aac6fef/*-mlx`), which
  are the checkpoints laya-js is validated on; it is not in Python.
- `ALIASES`, `TYPED_DECISION_WORKFLOWS`, `normaliseName` (also exported as
  `normalizeName`), `matchTypedDecisionsWorkflow`, `repoStr`, `splitSpec`.
- `new Router({ models?, maxLoaded = 1, default = "english", autoTaskDetection = false, standaloneRepos?, mlxRepos?, loadOptions?, loader? })`.
  - `loadOptions` is passed to every load, for example `{ backend, dtype, offline, token }`.
  - `loader(repo, opts)` defaults to `load` from `@johnhenry/laya`.
- `route(state, questions?, { model?, task?, lang? }) → RouteDecision`
  (`{ model, repo, reason, detection, workflow }`). Precedence: explicit
  `model` > explicit `task` > detected workflow (opt-in) > explicit `lang` >
  detected script or language > default. Reasons are Python's strings.
- `predict(state, questions, { model?, task?, lang? })` (alias `systemOne`)
  returns the agent's result plus `routing`.
- `load(name)`, `attach(name, agent)`, `preload(names?)`, `unload(name?)`, `loaded`, `maxLoaded`.

## Concurrency semantics

Python guards loading with an `RLock`. Here:

- Concurrent `load()` calls for one model share a single in-flight promise,
  so the model is built once and every caller gets the same agent. A failed
  build rejects every waiter and is not cached, so the next call retries.
- `maxLoaded` caps residency. The least-recently-used agent is evicted, but an
  agent with a `predict()` in flight through the router is never evicted or
  disposed. Residency can exceed `maxLoaded` until that call settles; then
  eviction resumes.
- The router disposes agents it built when they are evicted or unloaded. For an
  agent in use, disposal waits until the call settles. Attached agents belong
  to the caller and are only dropped. An agent that `load()` returned and that
  you use directly is not protected. Use `predict()`, or raise `maxLoaded`.
- `unload()` does not cancel builds that are already in flight.

## Tests

`test/router.test.ts` ports the routing tests from laya-mlx `test_router.py`
and `test_runtime.py`. The two thread-safety tests (#95) become async
concurrency tests. Further tests cover failed builds, eviction while a call is
in use, and deferred disposal. Language detection is tested in
`@johnhenry/langdetect-lite`.

## Limitations and differences from Python

- An auto-detected workflow reports `repo` as a string
  (`convaiinnovations/laya/typed-decisions`). Python puts the raw
  `(repo, subfolder)` tuple there, unlike every other branch.
- The constructor has no `preload=True`, because constructors cannot await.
  Call `await router.preload()`. It has no `device`, `dtype` or `token`
  arguments either; pass them in `loadOptions`.
- `DEFAULT_MODELS` points at the upstream PyTorch-trained bundle.
  `@johnhenry/laya` reads it through the upstream-name mapping, but only the
  `MLX_MODELS` repos were verified in this repository.
