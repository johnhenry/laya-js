# Changelog

## 0.1.0 (Unreleased)

First npm distribution of `@johnhenry/modernbert` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **ModernBERT / mmBERT encoder on any tensor backend**, loading safetensors; every stage within 1e-6 of MLX fp32 on the tiny checkpoint. [9bfc247](https://github.com/johnhenry/laya-js/commit/9bfc247).
- **`@johnhenry/math-plus-safetensors` moved to devDependencies**: the encoder only uses a structural `SafetensorsFile` type, so it has no runtime dependency on it.
