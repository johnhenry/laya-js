/**
 * C[i, j] (+)= Σ_k A[i, k] · B[j, k]   (A · Bᵀ, both row-major, "NT" layout).
 *
 * This is the only matmul kernel: `linear` (PyTorch [out, in] weights) uses
 * it directly, `matmul` and attention transpose their right operand once.
 * Both operands stream contiguously; a 4×4 register block gives 16
 * independent accumulators per k step, and j is blocked so a panel of B
 * stays cache-resident while all rows of A pass over it. Accumulation is in
 * f64 (JS numbers) and rounded once on store.
 */
export function gemmNT(
  A: Float32Array, aOff: number, lda: number,
  B: Float32Array, bOff: number, ldb: number,
  C: Float32Array, cOff: number, ldc: number,
  M: number, N: number, K: number,
): void {
  const JB = 64;
  for (let j0 = 0; j0 < N; j0 += JB) {
    const j1 = Math.min(N, j0 + JB);
    let i = 0;
    for (; i + 4 <= M; i += 4) {
      const a0 = aOff + i * lda, a1 = a0 + lda, a2 = a1 + lda, a3 = a2 + lda;
      const c0 = cOff + i * ldc, c1 = c0 + ldc, c2 = c1 + ldc, c3 = c2 + ldc;
      let j = j0;
      for (; j + 4 <= j1; j += 4) {
        const b0 = bOff + j * ldb, b1 = b0 + ldb, b2 = b1 + ldb, b3 = b2 + ldb;
        let s00 = 0, s01 = 0, s02 = 0, s03 = 0;
        let s10 = 0, s11 = 0, s12 = 0, s13 = 0;
        let s20 = 0, s21 = 0, s22 = 0, s23 = 0;
        let s30 = 0, s31 = 0, s32 = 0, s33 = 0;
        for (let k = 0; k < K; k++) {
          const x0 = A[a0 + k]!, x1 = A[a1 + k]!, x2 = A[a2 + k]!, x3 = A[a3 + k]!;
          const y0 = B[b0 + k]!, y1 = B[b1 + k]!, y2 = B[b2 + k]!, y3 = B[b3 + k]!;
          s00 += x0 * y0; s01 += x0 * y1; s02 += x0 * y2; s03 += x0 * y3;
          s10 += x1 * y0; s11 += x1 * y1; s12 += x1 * y2; s13 += x1 * y3;
          s20 += x2 * y0; s21 += x2 * y1; s22 += x2 * y2; s23 += x2 * y3;
          s30 += x3 * y0; s31 += x3 * y1; s32 += x3 * y2; s33 += x3 * y3;
        }
        C[c0 + j] = s00; C[c0 + j + 1] = s01; C[c0 + j + 2] = s02; C[c0 + j + 3] = s03;
        C[c1 + j] = s10; C[c1 + j + 1] = s11; C[c1 + j + 2] = s12; C[c1 + j + 3] = s13;
        C[c2 + j] = s20; C[c2 + j + 1] = s21; C[c2 + j + 2] = s22; C[c2 + j + 3] = s23;
        C[c3 + j] = s30; C[c3 + j + 1] = s31; C[c3 + j + 2] = s32; C[c3 + j + 3] = s33;
      }
      for (; j < j1; j++) {
        const b0 = bOff + j * ldb;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let k = 0; k < K; k++) {
          const y = B[b0 + k]!;
          s0 += A[a0 + k]! * y; s1 += A[a1 + k]! * y; s2 += A[a2 + k]! * y; s3 += A[a3 + k]! * y;
        }
        C[c0 + j] = s0; C[c1 + j] = s1; C[c2 + j] = s2; C[c3 + j] = s3;
      }
    }
    for (; i < M; i++) {
      const a0 = aOff + i * lda, c0 = cOff + i * ldc;
      let j = j0;
      for (; j + 4 <= j1; j += 4) {
        const b0 = bOff + j * ldb, b1 = b0 + ldb, b2 = b1 + ldb, b3 = b2 + ldb;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let k = 0; k < K; k++) {
          const x = A[a0 + k]!;
          s0 += x * B[b0 + k]!; s1 += x * B[b1 + k]!; s2 += x * B[b2 + k]!; s3 += x * B[b3 + k]!;
        }
        C[c0 + j] = s0; C[c0 + j + 1] = s1; C[c0 + j + 2] = s2; C[c0 + j + 3] = s3;
      }
      for (; j < j1; j++) {
        const b0 = bOff + j * ldb;
        let s = 0;
        for (let k = 0; k < K; k++) s += A[a0 + k]! * B[b0 + k]!;
        C[c0 + j] = s;
      }
    }
  }
}

/** Row-major [R, Cn] block at `off` → [Cn, R] into `out` at 0. */
export function transpose2d(src: Float32Array, off: number, R: number, Cn: number, out: Float32Array): void {
  for (let r = 0; r < R; r++) {
    const s = off + r * Cn;
    for (let c = 0; c < Cn; c++) out[c * R + r] = src[s + c]!;
  }
}
