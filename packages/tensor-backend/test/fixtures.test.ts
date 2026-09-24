import { test } from "node:test";
import assert from "node:assert/strict";
import { OP_CASE_FILES, decodeTensor, loadOpCases } from "../src/conformance.ts";
import { sizeOf } from "../src/host.ts";

test("every op fixture decodes to its declared shape", async () => {
  const cases = await loadOpCases();
  assert.ok(cases.length > 40, `only ${cases.length} cases`);
  for (const c of cases) {
    for (const e of [...c.inputs, ...c.outputs]) {
      if (!e) continue;
      const t = decodeTensor(e);
      assert.equal(t.data.length, sizeOf(t.shape), c.name);
    }
  }
});

test("the numerics fixtures cover every optional numerics op, in their own file", async () => {
  const { NUMERICS_OPS, NATIVE_ONLY_OPS } = await import("../src/compose.ts");
  const numerics = await loadOpCases(OP_CASE_FILES[1]!);
  const ops = new Set(numerics.map((c) => c.op));
  for (const op of NUMERICS_OPS) assert.ok(ops.has(op), `no case for ${op}`);
  assert.deepEqual([...NATIVE_ONLY_OPS], ["cumsum"]);
  const all = await loadOpCases();
  assert.equal(all.length, (await loadOpCases(OP_CASE_FILES[0]!)).length + numerics.length, "loadOpCases() loads both files");
});
