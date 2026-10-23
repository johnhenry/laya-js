---
"@johnhenry/backend-cpu": patch
---

Depend on `@johnhenry/math-plus-tensor-cpu` `^0.3.0`, which uses `tensor-backend` `^0.4.0`, so a fresh install has one copy of `tensor-backend` instead of a nested 0.3.0 next to 0.4.0.
