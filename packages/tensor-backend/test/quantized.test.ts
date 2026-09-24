import * as nodeTest from "node:test";
// Bun 1.2's node:test shim only registers tests from the first file of a run.
// @ts-ignore -- bun types are not installed
const bunTest: unknown = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const { test } = (bunTest ?? nodeTest) as Pick<typeof nodeTest, "test">;
import assert from "node:assert/strict";
import { packQuantized, unpackQuantized, validateQuantized, quantGroups } from "../src/host.ts";
import { hasNativeQuantized, isQuantized, QUANTIZED_OPS } from "../src/compose.ts";
import type { HostQuantized } from "../src/index.ts";

const scales = (N: number, G: number) => ({ dtype: "f16" as const, shape: [N, G], data: new Float16Array(N * G).fill(0.5) });

test("packQuantized / unpackQuantized round-trip the laya-js layout (low nibble = even column)", () => {
  const q4 = [0, 15, 7, 8, 1, 2, 3, 4];
  const d4 = packQuantized(q4, 4);
  assert.deepEqual([...d4], [0xf0, 0x87, 0x21, 0x43]);
  const h4: HostQuantized = { shape: [1, 8], bits: 4, groupSize: 8, mode: "affine", data: d4, scales: scales(1, 1), biases: scales(1, 1) };
  assert.deepEqual([...unpackQuantized(h4)], q4);
  // symmetric nibbles are two's complement
  const s4 = [-8, 7, -1, 0, 1, -2, 3, -4];
  assert.deepEqual([...unpackQuantized({ ...h4, mode: "symmetric", biases: null, data: packQuantized(s4, 4) })], s4);
  const q8 = [-127, 127, -1, 0];
  const h8: HostQuantized = { shape: [1, 4], bits: 8, groupSize: 4, mode: "symmetric", data: packQuantized(q8, 8), scales: scales(1, 1), biases: null };
  assert.deepEqual([...unpackQuantized(h8)], q8);
  // read as little-endian u32 words, value j sits at bit (j mod 8)·4 of word ⌊j/8⌋ (MLX packing)
  const word = new DataView(d4.buffer).getUint32(0, true);
  for (let j = 0; j < 8; j++) assert.equal((word >>> (4 * j)) & 15, q4[j]);
});

test("validateQuantized checks bytes, groups (partial last group) and biases", () => {
  const ok: HostQuantized = { shape: [2, 96], bits: 8, groupSize: 64, mode: "symmetric", data: new Uint8Array(192), scales: scales(2, 2), biases: null };
  assert.equal(quantGroups(96, 64), 2);
  validateQuantized(ok);
  assert.throws(() => validateQuantized({ ...ok, data: new Uint8Array(10) }), /bytes/);
  assert.throws(() => validateQuantized({ ...ok, scales: scales(2, 1) }), /scales/);
  assert.throws(() => validateQuantized({ ...ok, mode: "affine" }), /biases/);
  assert.throws(() => validateQuantized({ ...ok, biases: scales(2, 2) }), /no biases/);
  assert.throws(() => validateQuantized({ ...ok, shape: [2, 3], data: new Uint8Array(6), scales: scales(2, 1) }), /multiple of 32/);
});

test("isQuantized / hasNativeQuantized", () => {
  assert.equal(isQuantized({ shape: [1], dtype: "f32" } as any), false);
  assert.equal(isQuantized({ w: {}, scales: {}, biases: null, native: false } as any), true);
  const fake = Object.fromEntries(QUANTIZED_OPS.map((o) => [o, () => null]));
  assert.equal(hasNativeQuantized(fake as any), true);
  assert.equal(hasNativeQuantized({ quantizedLinear: () => null } as any), false);
});
