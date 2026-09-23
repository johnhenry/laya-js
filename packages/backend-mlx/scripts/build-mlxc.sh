#!/usr/bin/env bash
# Builds libmlxc.dylib (mlx-c) against the MLX 0.32.2 binaries from the
# `mlx` / `mlx-metal` Python wheel and assembles a relocatable bundle in
# prebuilds/darwin-arm64/:
#   libmlxc.dylib  libmlx.dylib  libjaccl.dylib  mlx.metallib
# Only mlx-c's thin C++ wrapper is compiled (~10 s); MLX itself and its
# Metal kernels are Apple's prebuilt wheel binaries.
#
# Env:
#   MLX_PY_DIR  path to an installed `mlx` package (site-packages/mlx).
#               Default: ask python3 (`python3 -c 'import mlx.core'`).
#   MLXC_REF    mlx-c git ref (default: d4afaec, "Support MLX v0.32.2").
#   SDKROOT     macOS SDK to link against (default: first SDK that links).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
out="$here/prebuilds/darwin-arm64"
ref="${MLXC_REF:-d4afaec5cc5c9ffbe58f37fdc038b2faaedc6e70}"
[[ "$(uname -s)/$(uname -m)" == "Darwin/arm64" ]] || { echo "macOS/arm64 only" >&2; exit 1; }
command -v cmake >/dev/null || { echo "cmake is required (brew install cmake)" >&2; exit 1; }

mlx="${MLX_PY_DIR:-$(python3 -c 'import mlx.core, os; print(os.path.dirname(mlx.core.__file__))' 2>/dev/null || true)}"
[[ -f "$mlx/lib/libmlx.dylib" && -d "$mlx/share/cmake/MLX" ]] || {
  echo "MLX wheel not found (set MLX_PY_DIR to site-packages/mlx; pip install mlx==0.32.2)" >&2; exit 1; }
ver="$(grep -E 'define MLX_VERSION_(MAJOR|MINOR|PATCH)' "$mlx/include/mlx/version.h" | awk '{print $3}' | paste -sd. -)"
echo "MLX wheel: $mlx (v$ver)"

# Some Command Line Tools SDKs ship .tbd stubs the linker rejects; pick one that links.
if [[ -z "${SDKROOT:-}" ]]; then
  tmp="$(mktemp -d)"; echo 'int main(){return 0;}' > "$tmp/t.cc"
  for sdk in "$(xcrun --show-sdk-path 2>/dev/null)" /Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk \
             /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX*.sdk; do
    [[ -d "$sdk" ]] || continue
    if c++ -isysroot "$sdk" "$tmp/t.cc" -o "$tmp/t" 2>/dev/null; then SDKROOT="$sdk"; break; fi
  done
  rm -rf "$tmp"
fi
[[ -n "${SDKROOT:-}" ]] || { echo "no macOS SDK that links; set SDKROOT" >&2; exit 1; }
export SDKROOT; echo "SDK: $SDKROOT"

work="${TMPDIR:-/tmp}/laya-mlxc-build"
rm -rf "$work"; mkdir -p "$work"
git -C "$work" init -q src
git -C "$work/src" fetch -q --depth 1 https://github.com/ml-explore/mlx-c "$ref"
git -C "$work/src" checkout -q FETCH_HEAD
cmake -S "$work/src" -B "$work/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_SYSROOT="$SDKROOT" \
  -DBUILD_SHARED_LIBS=ON -DMLX_C_BUILD_EXAMPLES=OFF -DMLX_C_USE_SYSTEM_MLX=ON -DMLX_DIR="$mlx/share/cmake/MLX" >/dev/null
cmake --build "$work/build" -j "$(sysctl -n hw.ncpu)" >/dev/null

mkdir -p "$out"
cp "$work/build/libmlxc.dylib" "$out/"
cp "$mlx/lib/libmlx.dylib" "$mlx/lib/libjaccl.dylib" "$mlx/lib/mlx.metallib" "$out/"
chmod u+w "$out"/*.dylib
# Make the bundle relocatable: resolve @rpath/libmlx.dylib next to libmlxc.
while read -r rp; do install_name_tool -delete_rpath "$rp" "$out/libmlxc.dylib"; done \
  < <(otool -l "$out/libmlxc.dylib" | awk '/LC_RPATH/{getline; getline; print $2}')
install_name_tool -add_rpath @loader_path "$out/libmlxc.dylib"
install_name_tool -id @rpath/libmlxc.dylib "$out/libmlxc.dylib"
install_name_tool -add_rpath @loader_path "$out/libmlx.dylib" 2>/dev/null || true
codesign --force -s - "$out"/*.dylib >/dev/null 2>&1
echo "mlx-c $ref + MLX $ver" > "$out/VERSION"
echo "built $out:"; ls -la "$out"
