# @johnhenry/laya-presets

Ready-made Laya question sets and email cleaning: a port of laya-mlx
`presets.py` and `email.py`, with the exact question text.

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
