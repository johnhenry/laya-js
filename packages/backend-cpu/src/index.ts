/**
 * @johnhenry/backend-cpu — now a thin re-export of
 * `@johnhenry/math-plus-tensor-cpu`, the CPU reference `Backend` that
 * math-plus owns (math-plus RFC 0001 §12 Q3, johnhenry/math-plus#144). It is
 * built on math-plus tensor-core's kernels, so the GEMM and the other kernels
 * that used to live here exist once, in math-plus.
 *
 * Kept so existing imports keep working; new code should depend on
 * `@johnhenry/math-plus-tensor-cpu` directly. This package will be
 * deprecated in a later release.
 *
 * Removed relative to 0.2.0: the raw `gemmNT` export (the GEMM is in
 * `@johnhenry/math-plus-tensor-core/kernels`, over packed f64 panels).
 */
export { createCpuBackend, CpuTensor, erf, erfc, geluScalar, type CpuBackend } from "@johnhenry/math-plus-tensor-cpu";
