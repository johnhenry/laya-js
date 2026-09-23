#!/usr/bin/env node
// Refuses to pack/publish an incomplete or modified native bundle: every file
// listed in lib/SHA256SUMS (written by build-mlxc.sh) must exist and match,
// the four runtime files must all be listed, and Apple's MIT notices must be
// present. Runs as `prepack`, so `npm pack` / `npm publish` cannot ship an
// empty package by accident (lib/ is gitignored and built in CI).
import { createHash } from "node:crypto";
import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const lib = fileURLToPath(new URL("../lib/", import.meta.url));
const REQUIRED = ["libmlxc.dylib", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib"];
const fail = (msg) => {
  console.error(`backend-mlx-darwin-arm64: ${msg}\nBuild the bundle first: npm run build:native (in packages/backend-mlx-darwin-arm64).`);
  process.exit(1);
};

if (!existsSync(join(lib, "SHA256SUMS"))) fail("lib/SHA256SUMS is missing");
for (const f of ["VERSION", "licenses/MLX.LICENSE", "licenses/mlx-c.LICENSE"]) {
  if (!existsSync(join(lib, f))) fail(`lib/${f} is missing`);
}
const sums = new Map(
  readFileSync(join(lib, "SHA256SUMS"), "utf8")
    .trim()
    .split("\n")
    .map((l) => l.trim().split(/\s+/))
    .map(([hash, file]) => [file, hash]),
);
for (const f of REQUIRED) if (!sums.has(f)) fail(`lib/SHA256SUMS does not list ${f}`);
const sha256 = (p) =>
  new Promise((res, rej) => {
    const h = createHash("sha256");
    createReadStream(p).on("data", (d) => h.update(d)).on("end", () => res(h.digest("hex"))).on("error", rej);
  });
let total = 0;
for (const [file, hash] of sums) {
  const p = join(lib, file);
  if (!existsSync(p)) fail(`lib/${file} is missing`);
  const got = await sha256(p);
  if (got !== hash) fail(`lib/${file} sha256 ${got} does not match SHA256SUMS (${hash})`);
  total += statSync(p).size;
}
console.log(`backend-mlx-darwin-arm64: ${sums.size} files verified (${(total / 1e6).toFixed(1)} MB), ${readFileSync(join(lib, "VERSION"), "utf8").trim()}`);
