# Changelog

## 0.1.0

### Minor Changes

- 7fecf44: Initial release: the laya-js package family (0.1.0). See each package's CHANGELOG.md for what ships, and the root README for parity results.


First npm distribution of `@johnhenry/hf-cache` (never published under any other name). Part of the initial laya-js release; see the [root README](https://github.com/johnhenry/laya-js#readme) for the family and its parity results.

- **Hugging Face Hub file resolution with the `huggingface_hub` cache layout** on Node/Bun/Deno (files Python downloaded are reused offline and vice versa) and the Cache API in browsers. [af2d5dc](https://github.com/johnhenry/laya-js/commit/af2d5dc).
- **No runtime dependencies**: the unused `@huggingface/hub` dependency was removed.
