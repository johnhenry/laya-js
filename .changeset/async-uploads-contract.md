---
"@johnhenry/tensor-backend": minor
---

**Breaking: `Backend.fromHost` is async** (`fromHost(t): Promise<T>`), like `read`, so device transfers are async-visible in both directions (math-plus RFC 0001 §12 Q2, closes #1). The conformance harness awaits uploads (and starts a case's uploads together), and the `compose.ts` helpers no longer upload: `meanPool`'s constant is derived on the device, with the new `zerosLike` / `onesLike` / `fullLike` helpers, so every helper stays synchronous and traceable by `compile`.

Migration: `const x = await backend.fromHost(h)`; batch independent uploads with `Promise.all`. Code that uploaded constants in the middle of a synchronous op sequence (inside `scope` or a `compile`d function) should upload them beforehand or use `onesLike` / `fullLike`. Backend implementers: make `fromHost` `async` (copying the host data at call time is fine).
