/** `test()` that runs under both `node --test` and `bun test` (same pattern as `snake-terminal/test/harness.ts`). */
import * as nodeTest from "node:test";

export type TestFn = (name: string, fn: () => unknown, opts?: { skip?: string | false | null; timeout?: number }) => void;

export function makeTest(bunTest: unknown): TestFn {
  const bun = bunTest as {
    test: ((n: string, f: () => unknown, timeout?: number) => void) & { skip: (n: string, f: () => unknown) => void };
  } | null;
  return (name, fn, opts = {}) => {
    if (bun) opts.skip ? bun.test.skip(`${name} (skipped: ${opts.skip})`, fn) : bun.test(name, fn, opts.timeout ?? 600_000);
    else nodeTest.test(name, { skip: opts.skip || false, timeout: opts.timeout ?? 600_000 }, fn as () => void);
  };
}
