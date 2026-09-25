import assert from "node:assert/strict";
import { test } from "node:test";
import { batchEligible, jsonToStateFields, newStateField, stateFieldsToJson, type SFDraft } from "../src/lib/state-fields.ts";

test("stateFieldsToJson collapses a single default text field to a plain string", () => {
  const fields: SFDraft[] = [{ key: "text", label: "Text", type: "text", value: "hello" }];
  assert.equal(stateFieldsToJson(fields), "hello");
});

test("stateFieldsToJson keeps a single non-default-key text field as an object", () => {
  const fields: SFDraft[] = [{ key: "message", label: "Message", type: "text", value: "hi" }];
  assert.deepEqual(stateFieldsToJson(fields), { message: "hi" });
});

test("stateFieldsToJson builds an object from multiple typed fields", () => {
  const fields: SFDraft[] = [
    { key: "message", label: "Message", type: "text", value: "hi" },
    { key: "priority", label: "Priority", type: "number", value: "3" },
    { key: "urgent", label: "Urgent", type: "boolean", value: "true" },
  ];
  assert.deepEqual(stateFieldsToJson(fields), { message: "hi", priority: 3, urgent: true });
});

test("stateFieldsToJson rejects a blank key", () => {
  assert.throws(() => stateFieldsToJson([{ key: " ", label: "", type: "text", value: "x" }]), /needs a key/);
});

test("stateFieldsToJson rejects duplicate keys", () => {
  const fields: SFDraft[] = [
    { key: "a", label: "", type: "text", value: "x" },
    { key: "a", label: "", type: "text", value: "y" },
  ];
  assert.throws(() => stateFieldsToJson(fields), /Duplicate state field key "a"/);
});

test("stateFieldsToJson rejects an empty text value", () => {
  assert.throws(() => stateFieldsToJson([{ key: "a", label: "", type: "text", value: "  " }]), /needs a value/);
});

test("stateFieldsToJson rejects a non-numeric number field", () => {
  assert.throws(() => stateFieldsToJson([{ key: "a", label: "", type: "number", value: "abc" }]), /needs a valid number/);
});

test("stateFieldsToJson rejects an empty field list", () => {
  assert.throws(() => stateFieldsToJson([]), /Add at least one state field/);
});

test("jsonToStateFields infers types from an object state", () => {
  const fields = jsonToStateFields({ message: "hi", priority: 3, urgent: true });
  assert.deepEqual(fields, [
    { key: "message", label: "message", type: "text", value: "hi" },
    { key: "priority", label: "priority", type: "number", value: "3" },
    { key: "urgent", label: "urgent", type: "boolean", value: "true" },
  ]);
});

test("jsonToStateFields wraps a plain string as a single text field", () => {
  assert.deepEqual(jsonToStateFields("free text"), [{ key: "text", label: "Text", type: "text", value: "free text" }]);
});

test("jsonToStateFields stashes non-representable states (array, null) as JSON text", () => {
  assert.deepEqual(jsonToStateFields([1, 2, 3]), [{ key: "text", label: "Text", type: "text", value: "[1,2,3]" }]);
  assert.deepEqual(jsonToStateFields(null), [{ key: "text", label: "Text", type: "text", value: "null" }]);
});

test("jsonToStateFields is the inverse of stateFieldsToJson for object states", () => {
  const original = { a: "x", b: 1, c: false };
  assert.deepEqual(stateFieldsToJson(jsonToStateFields(original)), original);
});

test("newStateField picks the next free field key, skipping keys already in use", () => {
  const existing: SFDraft[] = [{ key: "field0", label: "Field 1", type: "text", value: "" }];
  const field = newStateField(existing, "number");
  assert.equal(field.key, "field1");
  assert.equal(field.value, "");
});

test("newStateField defaults a boolean field's value to 'false'", () => {
  assert.equal(newStateField([], "boolean").value, "false");
});

test("batchEligible is true only for exactly one text field", () => {
  assert.equal(batchEligible([{ key: "text", label: "Text", type: "text", value: "" }]), true);
  assert.equal(batchEligible([{ key: "n", label: "N", type: "number", value: "" }]), false);
  assert.equal(
    batchEligible([
      { key: "a", label: "A", type: "text", value: "" },
      { key: "b", label: "B", type: "text", value: "" },
    ]),
    false,
  );
  assert.equal(batchEligible([]), false);
});
