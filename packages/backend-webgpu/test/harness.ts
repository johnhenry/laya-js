/**
 * describe/it/after from bun:test under Bun (Bun 1.2's node:test shim only
 * registers tests from the first file of a run), node:test elsewhere.
 * bun:test binds to the importing file, so each test file imports it itself:
 *   const t = harness(isBun ? await import("bun:test") : null);
 */
import * as nodeTest from "node:test";

type Fn = () => unknown;
interface Api {
  describe(name: string, fn: () => void): void;
  it(name: string, fn: Fn): void;
  after(fn: Fn): void;
  skip(name: string, reason: string): void;
}

export const isBun = (globalThis as { Bun?: unknown }).Bun !== undefined;

export const harness = (bun: Record<string, any> | null): Api =>
  bun
  ? {
      describe: (n, f) => bun.describe(n, f),
      it: (n, f) => bun.it(n, f, 120_000),
      after: (f) => bun.afterAll(f),
      skip: (n, r) => bun.it.skip(`${n} [${r}]`, () => {}),
    }
  : {
      describe: (n, f) => void nodeTest.describe(n, f),
      it: (n, f) => void nodeTest.it(n, f as () => void | Promise<void>),
      after: (f) => void nodeTest.after(f as () => void),
      skip: (n, r) => void nodeTest.it(n, { skip: r }, () => {}),
    };
