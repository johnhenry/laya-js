import { test } from "node:test";
import assert from "node:assert/strict";
import { decodeTensor, loadOpCases } from "../src/conformance.ts";
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
