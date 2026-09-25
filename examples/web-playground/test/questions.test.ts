import assert from "node:assert/strict";
import { test } from "node:test";
import { fromDrafts, newDraft, toDrafts, type QDraft } from "../src/lib/questions.ts";

test("toDrafts/fromDrafts round-trip a choice question", () => {
  const questions = {
    dept: { type: "choice", instructions: "Which department?", criteria: { billing: "invoices", technical: "bugs" } },
  };
  const drafts = toDrafts(questions);
  assert.equal(drafts.length, 1);
  assert.deepEqual(drafts[0], {
    id: "dept",
    type: "choice",
    instructions: "Which department?",
    rows: [{ label: "billing", desc: "invoices" }, { label: "technical", desc: "bugs" }],
    yes: "",
    no: "",
  });
  assert.deepEqual(fromDrafts(drafts), questions);
});

test("toDrafts handles a choice question with array criteria (no descriptions)", () => {
  const questions = { color: { type: "choice", instructions: "Pick one", criteria: ["red", "blue"] } };
  const drafts = toDrafts(questions);
  assert.deepEqual(drafts[0]!.rows, [{ label: "red", desc: "" }, { label: "blue", desc: "" }]);
  assert.deepEqual(fromDrafts(drafts), questions);
});

test("toDrafts/fromDrafts round-trip a score question", () => {
  const questions = { urgency: { type: "score", instructions: "How urgent?", criteria: ["low", "medium", "high"] } };
  const drafts = toDrafts(questions);
  assert.deepEqual(fromDrafts(drafts), questions);
});

test("toDrafts/fromDrafts round-trip a noul question with criteria", () => {
  const questions = { refund: { type: "noul", instructions: "Wants a refund?", criteria: { true: "asks for money back", false: "does not" } } };
  const drafts = toDrafts(questions);
  assert.equal(drafts[0]!.yes, "asks for money back");
  assert.equal(drafts[0]!.no, "does not");
  assert.deepEqual(fromDrafts(drafts), questions);
});

test("toDrafts/fromDrafts round-trip a noul question with no criteria", () => {
  const questions = { flagged: { type: "noul", instructions: "Is this urgent?" } };
  assert.deepEqual(fromDrafts(toDrafts(questions)), questions);
});

test("fromDrafts rejects a blank id", () => {
  const drafts: QDraft[] = [{ id: "  ", type: "noul", instructions: "x", rows: [], yes: "", no: "" }];
  assert.throws(() => fromDrafts(drafts), /needs an id/);
});

test("fromDrafts rejects duplicate ids", () => {
  const drafts: QDraft[] = [
    { id: "q0", type: "noul", instructions: "a", rows: [], yes: "", no: "" },
    { id: "q0", type: "noul", instructions: "b", rows: [], yes: "", no: "" },
  ];
  assert.throws(() => fromDrafts(drafts), /Duplicate question id "q0"/);
});

test("fromDrafts rejects missing instructions", () => {
  const drafts: QDraft[] = [{ id: "q0", type: "noul", instructions: "  ", rows: [], yes: "", no: "" }];
  assert.throws(() => fromDrafts(drafts), /needs instructions/);
});

test("fromDrafts rejects a choice with fewer than two options", () => {
  const drafts: QDraft[] = [{ id: "q0", type: "choice", instructions: "x", rows: [{ label: "only", desc: "" }], yes: "", no: "" }];
  assert.throws(() => fromDrafts(drafts), /at least two options/);
});

test("fromDrafts rejects a score with fewer than two levels", () => {
  const drafts: QDraft[] = [{ id: "q0", type: "score", instructions: "x", rows: [{ label: "low", desc: "" }], yes: "", no: "" }];
  assert.throws(() => fromDrafts(drafts), /at least two levels/);
});

test("fromDrafts rejects an empty draft list", () => {
  assert.throws(() => fromDrafts([]), /Add at least one question/);
});

test("newDraft picks the next free id per type, skipping ids already in use", () => {
  const existing: QDraft[] = [
    { id: "choice0", type: "choice", instructions: "", rows: [], yes: "", no: "" },
    { id: "choice1", type: "choice", instructions: "", rows: [], yes: "", no: "" },
  ];
  const draft = newDraft(existing, "choice");
  assert.equal(draft.id, "choice2");
  assert.equal(draft.rows.length, 2);
});

test("newDraft uses 'yesno' as the id base for noul questions", () => {
  assert.equal(newDraft([], "noul").id, "yesno0");
});
