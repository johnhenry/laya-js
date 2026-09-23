#!/usr/bin/env bash
# Refreshes vendor/johnhenry-math-plus-safetensors-<version>.tgz from a math-plus
# checkout, for local development and CI until @johnhenry/math-plus-safetensors
# is published to npm.
#
# Why a vendored tarball: every package that uses safetensors declares the
# semver range it will get from npm ("^0.0.0"). Until math-plus publishes it,
# the root package.json `overrides` entry maps that range to this tarball, so
# `npm ci` works in a fresh clone, in CI, and on any machine without a
# math-plus checkout at a particular relative path. `overrides` only apply to
# this repo's own install; they are never published, so consumers resolve
# the real registry package.
#
# After math-plus publishes: delete the `overrides` entry and vendor/, run
# `npm install`, commit the lockfile (see AGENTS.md "Releases").
#
# Env: MATH_PLUS_DIR  math-plus checkout (default: the f16-safetensors worktree
#                     next to this repo's parent, ../@johnhenry/math-plus.worktrees/f16-safetensors)
#      BUILD=1        run the package's build before packing
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mp="${MATH_PLUS_DIR:-$root/../@johnhenry/math-plus.worktrees/f16-safetensors}"
pkg="$mp/packages/safetensors"
[[ -f "$pkg/package.json" ]] || { echo "math-plus safetensors package not found at $pkg (set MATH_PLUS_DIR)" >&2; exit 1; }
if [[ "${BUILD:-0}" == 1 ]]; then (cd "$mp" && npm run build -w @johnhenry/math-plus-safetensors); fi
[[ -f "$pkg/dist/index.js" ]] || { echo "$pkg/dist missing: build math-plus first (BUILD=1)" >&2; exit 1; }
rm -f "$root"/vendor/johnhenry-math-plus-safetensors-*.tgz
(cd "$pkg" && npm pack --silent --pack-destination "$root/vendor" >/dev/null)
tgz="$(ls "$root"/vendor/johnhenry-math-plus-safetensors-*.tgz)"
{
  echo "package: @johnhenry/math-plus-safetensors"
  echo "source:  $(git -C "$mp" remote get-url origin 2>/dev/null || echo "$mp") @ $(git -C "$mp" rev-parse HEAD) ($(git -C "$mp" rev-parse --abbrev-ref HEAD))"
  echo "dirty:   $(git -C "$mp" status --porcelain -- packages/safetensors | wc -l | tr -d ' ') file(s)"
  echo "sha256:  $(shasum -a 256 "$tgz" | cut -d' ' -f1)  $(basename "$tgz")"
} > "$root/vendor/SOURCE"
cat "$root/vendor/SOURCE"
echo "now run: npm install   (refreshes package-lock.json)"
