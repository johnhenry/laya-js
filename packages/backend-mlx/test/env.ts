import { libCandidates, mlxPlatformSupported } from "../src/lib.ts";

/** Why MLX tests cannot run here, or null when they can. */
export const skipReason: string | null = !mlxPlatformSupported()
  ? "MLX needs macOS on Apple Silicon"
  : libCandidates().some((c) => c.exists)
    ? null
    : "libmlxc.dylib not found (see README: Install)";
