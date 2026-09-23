# laya-js

Laya typed decisions (choice / score / noul) in JavaScript, loading the
published MLX fp16 safetensors and running on **native MLX** (Node, Bun) or
**WebGPU** (browsers, Node/Bun via Dawn), with a pure-TS CPU reference.

This is a *family of packages*, not a single port. The generic layers are
meant to be useful without Laya:

| Package | Role |
|---|---|
| `@johnhenry/tensor-backend` | Backend op contract + conformance suite |
| `@johnhenry/backend-cpu` / `-mlx` / `-webgpu` | Implementations |
| `@johnhenry/modernbert` | ModernBERT / mmBERT encoder on any backend |
| `@johnhenry/pyjson` | Python-identical `json.dumps` + banker's rounding |
| `@johnhenry/hf-cache` | Hugging Face file resolution + caching |
| `@johnhenry/langdetect-lite` | Script + Latin language detection |
| `@johnhenry/laya-core` | Tensor-free prompts, collation, calibration |
| `@johnhenry/laya` | `load()` / `predict()` |
| `@johnhenry/laya-router`, `-presets`, `-cli` | Routing, presets, CLI |
| `@johnhenry/math-plus-safetensors` | (lives in [math-plus](https://github.com/johnhenry/math-plus)) |

Status: under construction. Parity target: every answer of the 63-question
laya-mlx validation set on all three checkpoints (fp32 1e-4, fp16 0.02).

Golden data comes from `laya-mlx/scripts/dump_js_fixtures.py`
(`npm run fixtures`). See [AGENTS.md](AGENTS.md) for conventions.

Ports logic from [laya-mlx](https://github.com/mizorewww/laya-mlx) and
[Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0); see NOTICE.
