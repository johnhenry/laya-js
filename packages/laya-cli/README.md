# @johnhenry/laya-cli

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Flaya-cli.svg)](https://www.npmjs.com/package/@johnhenry/laya-cli)

The `laya` command runs Laya typed decisions from a shell. It works under
Node ≥ 24 and Bun.

## Install

```bash
npm install @johnhenry/laya-cli
bun add @johnhenry/laya-cli
```

Node ≥ 24 or Bun ≥ 1.2. For GPU inference also install a backend next to it (`@johnhenry/backend-mlx` on Apple Silicon, `@johnhenry/backend-webgpu` elsewhere); without one it runs on the CPU reference.

```bash
npx @johnhenry/laya-cli predict --state "I was billed twice." --questions @questions.json
bunx @johnhenry/laya-cli bench --backend mlx
```

```bash
laya predict --model aac6fef/laya-mlx \
  --state "I was billed twice. Please refund the duplicate." \
  --questions '{"department": {"type": "choice", "instructions": "Who should handle this?", "criteria": ["billing", "technical", "sales"]}}'

laya predict --route --state @email.json --questions @questions.json   # pick the checkpoint by language
laya bench --backend mlx                                                # latency and throughput
```

## `laya predict`

`laya predict` prints the result as JSON on stdout, in the same shape and
formatting as `laya-mlx predict` (`json.dumps(result, ensure_ascii=False,
indent=2)`; floats print as floats, e.g. `1.0`). With `--backend mlx` the
output is byte-identical to the Python CLI for the README example.

| flag | |
|---|---|
| `--model <id\|path>` | Hub repo or local checkpoint (default `aac6fef/laya-mlx`) |
| `--state <json\|@file\|text>` | parsed as JSON when it is valid JSON, `@file` reads a JSON file, anything else is plain text |
| `--state-file <file>` | JSON state file (the laya-mlx flag) |
| `--questions <json\|@file\|file>` | question definitions |
| `--route` | choose the checkpoint with `@johnhenry/laya-router` (the `MLX_MODELS` repos); the result gains `routing`. `--model` then names a checkpoint (`english`, `ml`, `typed`, …); `--lang` and `--task` are routing hints |
| `--backend auto\|mlx\|webgpu\|cpu` | default `auto`: MLX, then WebGPU, then CPU |
| `--dtype f16\|f32` | default `f16`; `float16` and `float32` are accepted too |
| `--device gpu\|cpu` | MLX device |
| `--subfolder`, `--revision`, `--batch-size` (16), `--offline` | |

Exit codes: 0 on success, 2 for usage errors, 1 for runtime errors (message on
stderr). The temperature-clamping warning goes to stderr.

## `laya bench`

`laya bench` mirrors laya-mlx `benchmarks/worker.py`. The workload is
laya-mlx's `examples/state.json` with 1 question (the choice) or 50 questions
(the three example questions, cycled), and `--batch-size` defaults to 64. It
measures end-to-end `predict` wall time after `--warmup` (5) runs, over
`--iterations` (50) runs, and reports P50 and P95 for one question, throughput
for 50 questions (50 000 / mean ms), load time, peak MLX allocation, and
backend/device/host details. `--json` prints the report as JSON.

## Running from a checkout

`bin/laya.js` runs `dist/bin.js` when the package is built. In a workspace
checkout without a build, it re-executes itself under
`--conditions=source` so the TypeScript sources are used directly.
`node packages/laya-cli/bin/laya.js …` and `bun packages/laya-cli/bin/laya.js …`
both work.

## Limitations

- `--state` that happens to be valid JSON (for example `123` or `"quoted"`)
  is parsed as JSON. Use `--state-file` or `@file` to be explicit.
- There is no `convert` command (laya-mlx `convert` writes MLX checkpoints;
  use the Python tool).

## Family

Part of **[laya-js](https://github.com/johnhenry/laya-js#readme)**, Laya typed decisions in JavaScript on MLX, WebGPU and CPU — see its [package map](https://github.com/johnhenry/laya-js#which-package-do-i-want) and [results](https://github.com/johnhenry/laya-js#results).

- A thin shell over [`@johnhenry/laya`](https://github.com/johnhenry/laya-js/tree/main/packages/laya) (`load`/`predict`) and [`@johnhenry/laya-router`](https://github.com/johnhenry/laya-js/tree/main/packages/laya-router) (`--route`), printing JSON with [`@johnhenry/pyjson`](https://github.com/johnhenry/laya-js/tree/main/packages/pyjson) exactly as `laya-mlx predict` does.

## License

MIT. Ports logic from [laya-mlx](https://github.com/mizorewww/laya-mlx) and [Laya](https://github.com/NandhaKishorM/laya) (both Apache-2.0); those portions keep their notice: see [NOTICE](NOTICE) and [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0).
