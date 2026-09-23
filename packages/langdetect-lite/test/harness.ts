/**
 * `test()` that runs under both `node --test` and `bun test`.
 *
 * Bun 1.2's node:test shim only registers tests from the first file of a run, so
 * under Bun each test file must import bun:test itself (a shared module's import
 * would bind every registration to the first file) and pass it to {@link makeTest}:
 *
 *     // @ts-ignore -- bun types are not installed
 *     const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
 *     const test = makeTest(bun);
 */
import * as nodeTest from "node:test";

export type TestFn = (name: string, fn: () => unknown, opts?: { skip?: string | false | null }) => void;

/** Register tests with bun:test when given its module, else node:test. `skip` is a reason. */
export function makeTest(bunTest: unknown): TestFn {
  const bun = bunTest as { test: ((n: string, f: () => unknown) => void) & { skip: (n: string, f: () => unknown) => void } } | null;
  return (name, fn, opts = {}) => {
    if (bun) (opts.skip ? bun.test.skip : bun.test)(name, fn);
    else nodeTest.test(name, { skip: opts.skip || false }, fn as () => void);
  };
}
