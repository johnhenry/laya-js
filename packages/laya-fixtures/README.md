# @johnhenry/laya-fixtures

Golden fixtures generated from Python laya-mlx, used by the tests of every
other package. **Private: never published.**

## Install

Workspace-only. Other packages list it as a devDependency.

```ts
import { DATA_DIR } from "@johnhenry/laya-fixtures";
// DATA_DIR/tiny/…  the tiny fixture checkpoint and its expected outputs
```

Regenerate with `npm run fixtures` at the repo root, which runs
`../laya-mlx/scripts/dump_js_fixtures.py`. Never hand-edit `data/`.
Large fixtures (`data/large/`) are gitignored and regenerable.

## Limitations

- Needs a sibling `laya-mlx` checkout with its Python environment to
  regenerate.

## Family

Part of **[laya-js](https://github.com/johnhenry/laya-js#readme)**. The oracle for [`@johnhenry/laya-core`](https://github.com/johnhenry/laya-js/tree/main/packages/laya-core), [`@johnhenry/laya`](https://github.com/johnhenry/laya-js/tree/main/packages/laya), [`@johnhenry/modernbert`](https://github.com/johnhenry/laya-js/tree/main/packages/modernbert) and the backends.

## License

Apache-2.0.
