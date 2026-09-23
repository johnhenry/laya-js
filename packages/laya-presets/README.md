# @johnhenry/laya-presets

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Flaya-presets.svg)](https://www.npmjs.com/package/@johnhenry/laya-presets)

Ready-made Laya question sets and email cleaning: a port of laya-mlx
`presets.py` and `email.py`, with the exact question text.

## Install

```bash
npm install @johnhenry/laya-presets
bun add @johnhenry/laya-presets
deno add jsr:@johnhenry/laya-presets
```

Pure data and string functions: Node ≥ 24, Bun ≥ 1.2, Deno and browsers.

```ts
import { emailQuestions, emailState } from "@johnhenry/laya-presets";

const state = emailState("Duplicate charge", rawBody, { sender: "jane@example.com" });
const result = await agent.predict(state, emailQuestions());
```

## API

- `triageQuestions()`, `emailQuestions(categories?)`, `guardQuestions()`,
  `moderationQuestions()`, `routerQuestions()`, and `EMAIL_CATEGORIES`. Each
  call returns a fresh object.
- `cleanEmailBody(body, maxChars = 3000)` removes quoted history, `>` lines,
  late signatures and disclaimer sentences. A paragraph is dropped whole only
  when all of it is boilerplate (upstream #94).
- `emailState(subject, body, { sender?, clean = true, extra? })` builds
  `{ subject, body, from?, ...extra }`. Null or undefined values in `extra`
  are dropped.

## Tests

- `cleanEmailBody` against the Python table in `@johnhenry/laya-fixtures`.
- The ported `test_email.py`.
- A live differential run against `presets.py` and `email.py`, loaded by file
  path, so MLX is not needed. It covers every preset (deep-equal) and 19 email
  bodies, and needs `python3` and a laya-mlx checkout (`../laya-mlx` or
  `$LAYA_MLX_DIR`). It is skipped otherwise.

## Limitations

- Regexes are ported to JS with the `u` flag. Python's `\w` in the signature
  pattern becomes `[\p{L}\p{N}_]`. `\s` and `str.strip()` differ from Python
  on a few control characters (U+001C–U+001F count as whitespace in Python but
  not in JS).
- `maxChars` and the 40-character signature limit count code points, as Python does.

## Family

Part of **[laya-js](https://github.com/johnhenry/laya-js#readme)**, Laya typed decisions in JavaScript on MLX, WebGPU and CPU — see its [package map](https://github.com/johnhenry/laya-js#which-package-do-i-want) and [results](https://github.com/johnhenry/laya-js#results).

- Produces question objects for [`@johnhenry/laya`](https://github.com/johnhenry/laya-js/tree/main/packages/laya)'s `predict()` and [`@johnhenry/laya-router`](https://github.com/johnhenry/laya-js/tree/main/packages/laya-router); types come from [`@johnhenry/laya-core`](https://github.com/johnhenry/laya-js/tree/main/packages/laya-core).

## License

Apache-2.0. Ports logic from [laya-mlx](https://github.com/mizorewww/laya-mlx) and [Laya](https://github.com/NandhaKishorM/laya) (both Apache-2.0); see [NOTICE](NOTICE).
