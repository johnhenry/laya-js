/**
 * Thin, runtime-neutral binding to the subset of mlx-c this backend uses.
 *
 * - Bun: `bun:ffi` (built in). Node: `koffi` (prebuilt N-API addon).
 * - Every mlx-c handle (`mlx_array`, `mlx_stream`, `mlx_vector_array`,
 *   `mlx_closure`) is a one-pointer struct passed by value, which on arm64 is
 *   ABI-identical to passing the pointer itself, so handles cross as JS
 *   numbers (macOS user-space addresses are < 2^47, exact in a double).
 * - Out-params (`mlx_array* res`) point into one long-lived scratch
 *   ArrayBuffer whose address is computed once. ArrayBuffer backing stores
 *   created with `new ArrayBuffer(n)` are off-heap and never move in V8 or
 *   JSC, so the address stays valid for the life of the buffer (which we
 *   retain forever).
 * - mlx-c's default error handler prints and calls exit(); we install one
 *   that records the message so ops throw ordinary JS errors instead.
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

type ArgT = "h" | "buf" | "i32" | "bool" | "f32" | "usize" | "optf";
type RetT = "h" | "i32" | "usize" | "void";
interface Sym {
  args: ArgT[];
  ret: RetT;
}

const SYMBOLS = {
  mlx_set_error_handler: { args: ["h", "h", "h"], ret: "void" },
  mlx_array_new_data: { args: ["buf", "h", "i32", "i32"], ret: "h" },
  mlx_array_free: { args: ["h"], ret: "i32" },
  mlx_array_ndim: { args: ["h"], ret: "usize" },
  mlx_array_shape: { args: ["h"], ret: "h" },
  mlx_array_dtype: { args: ["h"], ret: "i32" },
  mlx_array_eval: { args: ["h"], ret: "i32" },
  mlx_array_data_uint8: { args: ["h"], ret: "h" },
  mlx_array_nbytes: { args: ["h"], ret: "usize" },
  mlx_eval: { args: ["h"], ret: "i32" },
  mlx_vector_array_new_data: { args: ["h", "usize"], ret: "h" },
  mlx_vector_array_set_data: { args: ["h", "h", "usize"], ret: "i32" },
  mlx_vector_array_free: { args: ["h"], ret: "i32" },
  mlx_vector_array_size: { args: ["h"], ret: "usize" },
  mlx_vector_array_get: { args: ["h", "h", "usize"], ret: "i32" },
  mlx_default_gpu_stream_new: { args: [], ret: "h" },
  mlx_default_cpu_stream_new: { args: [], ret: "h" },
  mlx_stream_free: { args: ["h"], ret: "i32" },
  mlx_synchronize: { args: ["h"], ret: "i32" },
  mlx_get_active_memory: { args: ["h"], ret: "i32" },
  mlx_get_peak_memory: { args: ["h"], ret: "i32" },
  mlx_clear_cache: { args: [], ret: "i32" },
  // ops
  mlx_add: { args: ["h", "h", "h", "h"], ret: "i32" },
  mlx_subtract: { args: ["h", "h", "h", "h"], ret: "i32" },
  mlx_multiply: { args: ["h", "h", "h", "h"], ret: "i32" },
  mlx_divide: { args: ["h", "h", "h", "h"], ret: "i32" },
  mlx_maximum: { args: ["h", "h", "h", "h"], ret: "i32" },
  mlx_where: { args: ["h", "h", "h", "h", "h"], ret: "i32" },
  mlx_exp: { args: ["h", "h", "h"], ret: "i32" },
  mlx_log: { args: ["h", "h", "h"], ret: "i32" },
  mlx_erf: { args: ["h", "h", "h"], ret: "i32" },
  mlx_astype: { args: ["h", "h", "i32", "h"], ret: "i32" },
  mlx_reshape: { args: ["h", "h", "h", "usize", "h"], ret: "i32" },
  mlx_transpose_axes: { args: ["h", "h", "h", "usize", "h"], ret: "i32" },
  mlx_slice: { args: ["h", "h", "h", "usize", "h", "usize", "h", "usize", "h"], ret: "i32" },
  mlx_split: { args: ["h", "h", "i32", "i32", "h"], ret: "i32" },
  mlx_concatenate_axis: { args: ["h", "h", "i32", "h"], ret: "i32" },
  mlx_sum_axis: { args: ["h", "h", "i32", "bool", "h"], ret: "i32" },
  mlx_max_axis: { args: ["h", "h", "i32", "bool", "h"], ret: "i32" },
  mlx_softmax_axis: { args: ["h", "h", "i32", "bool", "h"], ret: "i32" },
  mlx_sort_axis: { args: ["h", "h", "i32", "h"], ret: "i32" },
  mlx_matmul: { args: ["h", "h", "h", "h"], ret: "i32" },
  mlx_addmm: { args: ["h", "h", "h", "h", "f32", "f32", "h"], ret: "i32" },
  mlx_take_axis: { args: ["h", "h", "h", "i32", "h"], ret: "i32" },
  mlx_take_along_axis: { args: ["h", "h", "h", "i32", "h"], ret: "i32" },
  mlx_expand_dims: { args: ["h", "h", "i32", "h"], ret: "i32" },
  mlx_contiguous: { args: ["h", "h", "bool", "h"], ret: "i32" },
  mlx_fast_layer_norm: { args: ["h", "h", "h", "h", "f32", "h"], ret: "i32" },
  mlx_fast_rope: { args: ["h", "h", "i32", "bool", "optf", "f32", "i32", "h", "h"], ret: "i32" },
  // compile
  mlx_closure_new_func_payload: { args: ["h", "usize", "h"], ret: "h" },
  mlx_closure_free: { args: ["h"], ret: "i32" },
  mlx_closure_apply: { args: ["h", "h", "h"], ret: "i32" },
  mlx_compile: { args: ["h", "h", "bool"], ret: "i32" },
  memcpy: { args: ["buf", "h", "usize"], ret: "h" },
} satisfies Record<string, Sym>;

/** ABI-specific symbols: mlx-c < 0.6 (MLX 0.32.1) vs mlx-c >= 0.6 (MLX 0.32.2). */
const ABI_OLD: Record<string, Sym> = {
  mlx_detail_compile_clear_cache: { args: [], ret: "i32" },
};
const ABI_NEW: Record<string, Sym> = {
  mlx_detail_compile_cache: { args: ["h"], ret: "i32" },
  mlx_detail_compile_clear_cache: { args: ["h"], ret: "i32" },
  mlx_compile_cache_free: { args: ["h"], ret: "i32" },
};

const SDPA_OLD: Sym = { args: ["h", "h", "h", "h", "f32", "buf", "h", "h", "h"], ret: "i32" };
const SDPA_NEW: Sym = { args: ["h", "h", "h", "h", "f32", "buf", "h", "h", "bool", "h"], ret: "i32" };

type Fn = (...a: any[]) => any;
export type Native = { [K in keyof typeof SYMBOLS]: Fn } & {
  mlx_fast_scaled_dot_product_attention: Fn;
  mlx_detail_compile_clear_cache: Fn;
  mlx_detail_compile_cache?: Fn;
  mlx_compile_cache_free?: Fn;
  /** mlx-c ≥ 0.6 (MLX 0.32.2) added `bool force_fused` to sdpa. */
  readonly sdpaForceFused: boolean;
  readonly runtime: "bun" | "node";
  readonly libPath: string;
  /** Address of a typed array's first byte (the array must stay alive). */
  addressOf(a: ArrayBufferView): number;
  /** Takes and clears the last error message recorded by the handler. */
  takeError(): string | null;
  /** C trampoline `int (*)(mlx_vector_array*, const mlx_vector_array, void*)`. */
  closureTrampoline(fn: (res: number, input: number, payload: number) => number): number;
};

const isBun = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";

let lastError: string | null = null;

function openBun(path: string): Native {
  const ffi = require("bun:ffi");
  const T: Record<string, string> = { h: "ptr", buf: "ptr", i32: "i32", bool: "bool", f32: "f32", usize: "usize", optf: "u64", void: "void" };
  let probe = false;
  try {
    const p = ffi.dlopen(path, { mlx_compile_cache_new: { args: [], returns: "ptr" } });
    probe = true;
    p.close();
  } catch {
    probe = false;
  }
  const specs: Record<string, Sym> = { ...SYMBOLS, ...(probe ? ABI_NEW : ABI_OLD), mlx_fast_scaled_dot_product_attention: probe ? SDPA_NEW : SDPA_OLD };
  const defs: Record<string, { args: string[]; returns: string }> = {};
  for (const [k, s] of Object.entries(specs)) defs[k] = { args: s.args.map((a) => T[a]!), returns: T[s.ret]! };
  const lib = ffi.dlopen(path, defs);
  const out: Record<string, unknown> = {};
  for (const [k, s] of Object.entries(specs)) {
    const fn = lib.symbols[k] as Fn;
    out[k] = s.ret === "h" ? (...a: unknown[]) => Number(fn(...a) ?? 0) : s.ret === "usize" ? (...a: unknown[]) => Number(fn(...a)) : fn;
  }
  const keep: unknown[] = [];
  const errCb = new ffi.JSCallback(
    (msg: number | null) => {
      try {
        lastError = msg ? new ffi.CString(msg).toString() : "unknown mlx-c error";
      } catch {
        lastError = "mlx-c error (message not decodable)";
      }
    },
    { args: ["ptr", "ptr"], returns: "void" },
  );
  keep.push(errCb);
  (out.mlx_set_error_handler as Fn)(errCb.ptr, 0, 0);
  return Object.assign(out, {
    sdpaForceFused: probe,
    runtime: "bun" as const,
    libPath: path,
    _keep: keep,
    addressOf: (a: ArrayBufferView) => Number(ffi.ptr(a)),
    takeError: () => {
      const e = lastError;
      lastError = null;
      return e;
    },
    closureTrampoline(fn: (res: number, input: number, payload: number) => number) {
      const cb = new ffi.JSCallback((r: number | null, i: number | null, p: number | bigint) => fn(r ?? 0, i ?? 0, Number(p)), {
        args: ["ptr", "ptr", "u64"],
        returns: "i32",
      });
      keep.push(cb);
      return Number(cb.ptr);
    },
  }) as unknown as Native;
}

function openNode(path: string): Native {
  const koffi = require("koffi");
  const T: Record<string, string> = { h: "uintptr_t", buf: "void *", i32: "int32_t", bool: "bool", f32: "float", usize: "size_t", optf: "uint64_t", void: "void" };
  const lib = koffi.load(path);
  let probe = false;
  try {
    lib.func("mlx_compile_cache_new", "uintptr_t", []);
    probe = true;
  } catch {
    probe = false;
  }
  const specs: Record<string, Sym> = { ...SYMBOLS, ...(probe ? ABI_NEW : ABI_OLD), mlx_fast_scaled_dot_product_attention: probe ? SDPA_NEW : SDPA_OLD };
  const out: Record<string, unknown> = {};
  for (const [k, s] of Object.entries(specs)) {
    const fn = lib.func(k, T[s.ret], s.args.map((a) => T[a]));
    out[k] = s.ret === "usize" || s.ret === "h" ? (...a: unknown[]) => Number(fn(...a)) : fn;
  }
  const keep: unknown[] = [];
  const ErrProto = koffi.proto("void laya_mlx_err_cb(const char *msg, void *data)");
  const errCb = koffi.register((msg: string | null) => {
    lastError = msg ?? "unknown mlx-c error";
  }, koffi.pointer(ErrProto));
  keep.push(errCb);
  (out.mlx_set_error_handler as Fn)(Number(koffi.address(errCb)), 0, 0);
  const ClosureProto = koffi.proto("int laya_mlx_closure_cb(uintptr_t res, uintptr_t input, uintptr_t payload)");
  return Object.assign(out, {
    sdpaForceFused: probe,
    runtime: "node" as const,
    libPath: path,
    _keep: keep,
    addressOf: (a: ArrayBufferView) => Number(koffi.address(a)),
    takeError: () => {
      const e = lastError;
      lastError = null;
      return e;
    },
    closureTrampoline(fn: (res: number, input: number, payload: number) => number) {
      const cb = koffi.register((r: number | bigint, i: number | bigint, p: number | bigint) => fn(Number(r), Number(i), Number(p)), koffi.pointer(ClosureProto));
      keep.push(cb);
      return Number(koffi.address(cb));
    },
  }) as unknown as Native;
}

const cache = new Map<string, Native>();

/** Opens (once per path per process) the mlx-c library. */
export function openNative(path: string): Native {
  let n = cache.get(path);
  if (!n) {
    n = isBun ? openBun(path) : openNode(path);
    cache.set(path, n);
    installExitHook(n);
  }
  return n;
}

/**
 * MLX's global compile cache holds traced graphs (and our C trampolines).
 * Destroying it during C++ static teardown, after the Metal device and the
 * FFI callback trampolines are gone, segfaults at process exit. Python MLX
 * avoids this with an atexit hook calling compile_clear_cache; so do we.
 */
function installExitHook(n: Native): void {
  const proc = (globalThis as { process?: { on?(ev: string, fn: () => void): void } }).process;
  proc?.on?.("exit", () => clearCompileCache(n));
}

export function clearCompileCache(n: Native): void {
  try {
    if (n.sdpaForceFused && n.mlx_detail_compile_cache && n.mlx_compile_cache_free) {
      const slot = new Uint32Array(new ArrayBuffer(16));
      const addr = n.addressOf(slot);
      if (n.mlx_detail_compile_cache(addr) === 0) {
        const c = slot[0]! + slot[1]! * 4294967296;
        n.mlx_detail_compile_clear_cache(c);
        n.mlx_compile_cache_free(c);
      }
    } else {
      n.mlx_detail_compile_clear_cache();
    }
  } catch {
    // best effort at exit
  }
}
