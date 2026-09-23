/**
 * Manifest-drift guard (mirrors math-plus's test/manifest-drift.test.ts).
 *
 * Hand-maintained manifests with no validating test silently drift from
 * reality. This repo has several: the JSR package list, per-package
 * package.json metadata, the engines floor repeated in every manifest and
 * in CI, LICENSE/NOTICE copies, and the npm workspaces exclusion for the
 * native platform package. Each test below names what to fix.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
// @ts-expect-error — plain .mjs script without type declarations
import { JSR_EXCLUDED, PACKAGE_DIRS, jsrConfigFor } from "../scripts/sync-jsr-configs.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

interface Pkg {
  name: string;
  version: string;
  private?: boolean;
  license?: string;
  homepage?: string;
  repository?: { directory?: string; url?: string };
  exports?: Record<string, unknown>;
  files?: string[];
  engines?: Record<string, string>;
  os?: string[];
  cpu?: string[];
  publishConfig?: { access?: string };
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
  optionalDependencies?: Record<string, string>;
  workspaces?: string[];
}

const readJson = <T = Pkg>(p: string): T => JSON.parse(readFileSync(join(ROOT, p), "utf8")) as T;
const root = readJson("package.json");

/** Native platform packages: no TypeScript, MIT (Apple binaries), outside npm workspaces. */
const PLATFORM_PACKAGES = new Set(["packages/backend-mlx-darwin-arm64"]);

function discover(globRoot: string): { dir: string; pkg: Pkg }[] {
  return readdirSync(join(ROOT, globRoot), { withFileTypes: true })
    .filter((e) => e.isDirectory() && existsSync(join(ROOT, globRoot, e.name, "package.json")))
    .map((e) => ({ dir: `${globRoot}/${e.name}`, pkg: readJson(`${globRoot}/${e.name}/package.json`) }));
}
const packages = discover("packages");
const examples = discover("examples");
const published = packages.filter((p) => !p.pkg.private);

test("every TypeScript package has the tsconfig.json + tsconfig.typecheck.json pair", () => {
  const missing = packages
    .filter((p) => !PLATFORM_PACKAGES.has(p.dir))
    .flatMap((p) => ["tsconfig.json", "tsconfig.typecheck.json"].filter((f) => !existsSync(join(ROOT, p.dir, f))).map((f) => `${p.dir}/${f}`));
  assert.deepEqual(missing, [], `missing tsconfig files: ${missing.join(", ")} -- copy an existing package's pair`);
});

test("every package is either in sync-jsr-configs.mjs PACKAGE_DIRS or explicitly in JSR_EXCLUDED", () => {
  const onJsr = new Set<string>(PACKAGE_DIRS);
  const excluded = new Set(Object.keys(JSR_EXCLUDED));
  const dirs = packages.map((p) => p.dir);
  const neither = dirs.filter((d) => !onJsr.has(d) && !excluded.has(d));
  const both = dirs.filter((d) => onJsr.has(d) && excluded.has(d));
  const stale = [...onJsr, ...excluded].filter((d) => !dirs.includes(d));
  assert.deepEqual(neither, [], `package(s) in neither PACKAGE_DIRS nor JSR_EXCLUDED: ${neither.join(", ")}`);
  assert.deepEqual(both, [], `package(s) in both lists: ${both.join(", ")}`);
  assert.deepEqual(stale, [], `stale JSR list entries (dir no longer exists): ${stale.join(", ")}`);
  for (const d of excluded) assert.ok(String((JSR_EXCLUDED as Record<string, string>)[d]).length > 10, `${d}: give a real reason`);
});

test("every committed jsr.json matches what sync-jsr-configs.mjs generates", () => {
  for (const dir of PACKAGE_DIRS as string[]) {
    const path = join(ROOT, dir, "jsr.json");
    assert.ok(existsSync(path), `${dir}/jsr.json missing -- run npm run sync:jsr`);
    assert.deepEqual(JSON.parse(readFileSync(path, "utf8")), jsrConfigFor(dir), `${dir}/jsr.json is stale -- run npm run sync:jsr`);
  }
  for (const dir of Object.keys(JSR_EXCLUDED)) {
    assert.ok(!existsSync(join(ROOT, dir, "jsr.json")), `${dir} is JSR_EXCLUDED but has a jsr.json`);
  }
});

test("every package has README.md and CHANGELOG.md; published ones ship LICENSE (and NOTICE when listed)", () => {
  const rootLicense = readFileSync(join(ROOT, "LICENSE"), "utf8");
  const rootNotice = readFileSync(join(ROOT, "NOTICE"), "utf8");
  for (const { dir, pkg } of packages) {
    for (const f of ["README.md", "CHANGELOG.md"]) assert.ok(existsSync(join(ROOT, dir, f)), `${dir}/${f} missing`);
    if (pkg.private) continue;
    assert.ok(existsSync(join(ROOT, dir, "LICENSE")), `${dir}/LICENSE missing (npm ships it automatically)`);
    assert.ok(pkg.files?.includes("CHANGELOG.md"), `${dir}: add CHANGELOG.md to "files"`);
    if (PLATFORM_PACKAGES.has(dir)) continue;
    assert.equal(readFileSync(join(ROOT, dir, "LICENSE"), "utf8"), rootLicense, `${dir}/LICENSE differs from the root LICENSE`);
    if (pkg.files?.includes("NOTICE")) {
      assert.ok(existsSync(join(ROOT, dir, "NOTICE")), `${dir}: "files" lists NOTICE but the file is missing`);
      assert.equal(readFileSync(join(ROOT, dir, "NOTICE"), "utf8"), rootNotice, `${dir}/NOTICE differs from the root NOTICE`);
      // Apache-2.0 §4(a): derived portions must ship with a copy of that license.
      assert.ok(pkg.files?.includes("LICENSE-APACHE-2.0"), `${dir}: ships NOTICE, so "files" must list LICENSE-APACHE-2.0`);
      assert.equal(readFileSync(join(ROOT, dir, "LICENSE-APACHE-2.0"), "utf8"), readFileSync(join(ROOT, "LICENSE-APACHE-2.0"), "utf8"), `${dir}/LICENSE-APACHE-2.0 differs from the root copy`);
    } else {
      assert.ok(!existsSync(join(ROOT, dir, "NOTICE")), `${dir}/NOTICE exists but "files" does not ship it`);
    }
  }
});

test("engines agree everywhere (root, packages, examples) and with the CI matrix", () => {
  const floor = root.engines;
  assert.ok(floor?.node && floor.bun, "root package.json needs engines.node and engines.bun");
  const bad = [...packages, ...examples].filter((p) => JSON.stringify(p.pkg.engines) !== JSON.stringify(floor)).map((p) => p.dir);
  assert.deepEqual(bad, [], `engines differ from root ${JSON.stringify(floor)} in: ${bad.join(", ")}`);
  const nodeMajor = floor.node.match(/(\d+)/)![1];
  const bunMinor = floor.bun.match(/(\d+\.\d+)/)![1];
  assert.equal(readFileSync(join(ROOT, ".nvmrc"), "utf8").trim(), nodeMajor, ".nvmrc must equal the engines.node major");
  const ci = readFileSync(join(ROOT, ".github/workflows/ci.yml"), "utf8");
  assert.match(ci, new RegExp(`node-version: \\[${nodeMajor}\\]`), `ci.yml must test the engines floor Node ${nodeMajor}`);
  assert.match(ci, new RegExp(`bun-version: "?${bunMinor.replace(".", "\\.")}`), `ci.yml must test the engines floor Bun ${bunMinor}`);
  for (const wf of ["ci.yml", "release.yml"]) {
    const text = readFileSync(join(ROOT, ".github/workflows", wf), "utf8");
    const versions = [...text.matchAll(/node-version: "?(\d+)"?/g)].map((m) => m[1]);
    assert.ok(versions.every((v) => v === nodeMajor), `${wf} uses node ${versions.join(",")} but engines.node is ${floor.node}`);
  }
});

test("published package metadata is complete and consistent", () => {
  for (const { dir, pkg } of published) {
    const where = `${dir}/package.json`;
    assert.equal(pkg.repository?.directory, dir, `${where}: repository.directory`);
    assert.equal(pkg.repository?.url, "git+https://github.com/johnhenry/laya-js.git", `${where}: repository.url`);
    assert.ok(pkg.homepage, `${where}: homepage`);
    assert.equal(pkg.publishConfig?.access, "public", `${where}: publishConfig.access must be public (scoped packages default to private)`);
    assert.equal(pkg.license, "MIT", `${where}: license`);
    assert.ok(pkg.version === "0.0.0" || /^\d+\.\d+\.\d+/.test(pkg.version), `${where}: version`);
    if (!PLATFORM_PACKAGES.has(dir)) {
      assert.ok(pkg.files?.includes("dist"), `${where}: "files" must include dist`);
      // The "source" export condition must resolve in the tarball too: a consumer
      // running with --conditions=source would otherwise hit ERR_MODULE_NOT_FOUND.
      assert.ok(pkg.files?.includes("src"), `${where}: "files" must include src (exports use the "source" condition)`);
      assert.match(
        (pkg as Pkg & { scripts?: Record<string, string> }).scripts?.build ?? "",
        /rewrite-dts-extensions\.mjs dist$/,
        `${where}: build must end with scripts/rewrite-dts-extensions.mjs (tsc leaves ".ts" specifiers in .d.ts files)`,
      );
      for (const [sub, entry] of Object.entries(pkg.exports ?? {})) {
        if (typeof entry !== "object" || entry === null) continue;
        const keys = Object.keys(entry);
        const first = keys[0] === "browser" ? Object.keys((entry as Record<string, object>).browser)[0] : keys[0];
        assert.equal(first, "source", `${where}: exports["${sub}"] must list the "source" condition first`);
      }
    }
  }
});

test("internal dependencies use semver ranges (no file:, link:, workspace:, *)", () => {
  const offenders: string[] = [];
  for (const { dir, pkg } of [...packages, ...examples]) {
    for (const field of ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"] as const) {
      for (const [name, range] of Object.entries(pkg[field] ?? {})) {
        if (!/^[\^~]?\d/.test(range)) offenders.push(`${dir} ${field}.${name}=${range}`);
      }
    }
  }
  assert.deepEqual(offenders, [], `non-semver dependency ranges: ${offenders.join("; ")}`);
});

test("native platform packages stay out of npm workspaces (EBADPLATFORM) and match backend-mlx's optional range", () => {
  for (const { dir, pkg } of packages.filter((p) => p.pkg.os || p.pkg.cpu)) {
    assert.ok(PLATFORM_PACKAGES.has(dir), `${dir} has os/cpu but is not in PLATFORM_PACKAGES`);
    assert.ok(root.workspaces?.includes(`!${dir}`), `root workspaces must exclude "!${dir}" -- npm hard-fails with EBADPLATFORM on other platforms for os/cpu-gated workspace members`);
  }
  // In this repo the platform package is linked through a root optionalDependency
  // (file:) + override, so the lockfile stays valid for `npm ci` before it is on
  // npm, and npm skips it silently on other platforms (it is optional there).
  const r = root as Pkg & { overrides?: Record<string, string> };
  for (const dir of PLATFORM_PACKAGES) {
    const name = readJson(`${dir}/package.json`).name;
    assert.equal(r.optionalDependencies?.[name], `file:${dir}`, `root optionalDependencies must link ${name} as file:${dir}`);
    assert.equal(r.overrides?.[name], `$${name}`, `root overrides must map ${name} to the root's file: link`);
  }
  const mlx = readJson("packages/backend-mlx/package.json");
  const plat = readJson("packages/backend-mlx-darwin-arm64/package.json");
  assert.equal(mlx.optionalDependencies?.[plat.name], `^${plat.version}`, "backend-mlx optionalDependencies must pin ^<platform package version>");
  assert.equal(mlx.version, plat.version, "backend-mlx and its platform package are versioned together");
});

test("the vendored math-plus-safetensors tarball backs the root devDependency + override", () => {
  const spec = (root as Pkg & { devDependencies: Record<string, string> }).devDependencies["@johnhenry/math-plus-safetensors"];
  if (!spec) return; // math-plus published: override removed, nothing to check
  const file = spec.replace(/^file:/, "");
  assert.ok(existsSync(join(ROOT, file)), `${file} missing -- run npm run vendor:math-plus`);
  const overrides = (root as { overrides?: Record<string, string> }).overrides ?? {};
  assert.equal(overrides["@johnhenry/math-plus-safetensors"], "$@johnhenry/math-plus-safetensors", "root overrides must point workspace ranges at the vendored tarball");
});
