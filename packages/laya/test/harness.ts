/**
 * `test()` that runs under both `node --test` and `bun test`.
 *
 * Bun 1.2's node:test shim only registers tests from the first file of a run, so
 * under Bun each test file must import bun:test itself and pass it here:
 *
 *     // @ts-ignore -- bun types are not installed
 *     const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
 *     const test = makeTest(bun);
 */
import * as nodeTest from "node:test";

export interface TestOpts {
  /** Skip with this reason. */
  skip?: string | false | null;
  /** Milliseconds (Bun's default is 5 s; node:test has none). */
  timeout?: number;
}
export type TestFn = (name: string, fn: () => unknown, opts?: TestOpts) => void;

type BunTest = ((n: string, f: () => unknown, ms?: number) => void) & { skip: (n: string, f: () => unknown) => void };

export function makeTest(bunTest: unknown): TestFn {
  const bun = bunTest as { test: BunTest } | null;
  return (name, fn, opts = {}) => {
    if (bun) {
      if (opts.skip) bun.test.skip(`${name} [${opts.skip}]`, fn);
      else bun.test(name, fn, opts.timeout ?? 60_000);
    } else nodeTest.test(name, { skip: opts.skip || false, ...(opts.timeout ? { timeout: opts.timeout } : {}) }, fn as () => void);
  };
}

export const env: Record<string, string | undefined> =
  (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env ?? {};
