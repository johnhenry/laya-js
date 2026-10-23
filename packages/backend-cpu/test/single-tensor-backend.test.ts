import * as nodeTest from "node:test";
// @ts-ignore -- bun types are not installed
const bunTest: unknown = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const { it: test } = (bunTest ?? nodeTest) as Pick<typeof nodeTest, "it">;
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";

test("backend-cpu's dependency ranges admit one tensor-backend major.minor", () => {
  const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
  const mine = pkg.dependencies["@johnhenry/tensor-backend"];
  const range = pkg.dependencies["@johnhenry/math-plus-tensor-cpu"];
  const npm = (...a: string[]) => JSON.parse(execFileSync("npm", ["view", ...a, "--json"], { encoding: "utf8" }));
  const versions: string[] = npm("@johnhenry/math-plus-tensor-cpu", "versions");
  const semver = (v: string) => v.split(".").map(Number);
  // highest published version that satisfies backend-cpu's own range (caret on 0.x pins the minor)
  const [, minor] = semver(range.replace(/^\^/, ""));
  const best = versions.filter((v) => semver(v)[0] === 0 && semver(v)[1] === minor).at(-1);
  const dep = npm(`@johnhenry/math-plus-tensor-cpu@${best}`, "dependencies");
  assert.equal(dep["@johnhenry/tensor-backend"], mine, `math-plus-tensor-cpu@${best} wants tensor-backend ${dep["@johnhenry/tensor-backend"]}, backend-cpu wants ${mine}`);
});
