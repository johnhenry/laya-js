// Paths into the prebuilt MLX bundle. @johnhenry/backend-mlx resolves
// `@johnhenry/backend-mlx-darwin-arm64/package.json` and loads lib/libmlxc.dylib;
// importing this module is only needed by tools that want the paths.
import { fileURLToPath } from "node:url";

/** Directory holding libmlxc.dylib, libmlx.dylib, libjaccl.dylib and mlx.metallib. */
export const libDir = fileURLToPath(new URL("./lib/", import.meta.url));
/** Absolute path of libmlxc.dylib (pass to `createMlxBackend({ libPath })` or `$LAYA_MLXC_PATH`). */
export const libmlxc = fileURLToPath(new URL("./lib/libmlxc.dylib", import.meta.url));
