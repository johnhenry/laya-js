import assert from "node:assert/strict";
import { test } from "node:test";
import { computePriorityValue, csvCell, queueToCsv, queueToJson, stateSummary, type QueueEntry } from "../src/lib/queue.ts";

function entry(overrides: Partial<QueueEntry> = {}): QueueEntry {
  return {
    id: "q0",
    timestamp: 1700000000000,
    state: { message: "hello" },
    questions: {
      urgency: { type: "score", instructions: "How urgent?", criteria: ["low", "medium", "high"] },
      refund: { type: "noul", instructions: "Wants a refund?" },
    },
    outcomes: [
      {
        label: "WebGPU f16",
        ms: 12.3,
        result: {
          answers: {
            urgency: { type: "score", confidence: 0.9, action: { act_probability: 0.5 }, score: 2, probabilities: { "0": 0.1, "1": 0.2, "2": 0.7 } },
            refund: { type: "noul", confidence: 0.8, action: { act_probability: 0.5 }, noul: 0.75 },
          },
          usage: { input_tokens: 42 },
        },
      },
    ],
    ...overrides,
  };
}

test("computePriorityValue returns null when no priority question is selected", () => {
  assert.equal(computePriorityValue(entry(), ""), null);
});

test("computePriorityValue reads a noul answer directly as 0..1", () => {
  assert.equal(computePriorityValue(entry(), "refund"), 0.75);
});

test("computePriorityValue normalizes a score answer to 0..1 by level count", () => {
  // score 2 out of levels [low, medium, high] (3 levels, max index 2) -> 2/2 = 1
  assert.equal(computePriorityValue(entry(), "urgency"), 1);
});

test("computePriorityValue returns null for a priority question not present on the entry", () => {
  assert.equal(computePriorityValue(entry(), "missing"), null);
});

test("stateSummary reads the first value of an object state, truncated to 80 chars", () => {
  assert.equal(stateSummary({ message: "hello" }), "hello");
  assert.equal(stateSummary("plain text"), "plain text");
  const long = "x".repeat(200);
  assert.equal(stateSummary(long), "x".repeat(80));
});

test("stateSummary falls back to '(empty)' for an empty value", () => {
  assert.equal(stateSummary(""), "(empty)");
});

test("csvCell quotes values containing commas, quotes, or newlines", () => {
  assert.equal(csvCell("plain"), "plain");
  assert.equal(csvCell("a,b"), '"a,b"');
  assert.equal(csvCell('say "hi"'), '"say ""hi"""');
  assert.equal(csvCell("line1\nline2"), '"line1\nline2"');
  assert.equal(csvCell(undefined), "");
  assert.equal(csvCell(null), "");
});

test("queueToJson round-trips the queue array", () => {
  const q = [entry()];
  assert.deepEqual(JSON.parse(queueToJson(q)), q);
});

test("queueToCsv derives columns from the union of state/question keys actually seen", () => {
  const q = [
    entry({ id: "q0", state: { message: "hi" } }),
    entry({
      id: "q1",
      state: { other: "field" },
      questions: { churn: { type: "noul", instructions: "Will churn?" } },
      outcomes: [{ label: "WebGPU f16", ms: 5, result: { answers: { churn: { type: "noul", confidence: 0.5, action: { act_probability: 0.5 }, noul: 0.2 } } } }],
    }),
  ];
  const csv = queueToCsv(q, "");
  const [header, ...rows] = csv.split("\n");
  assert.equal(header, "id,timestamp,engine,message,other,priorityValue,latencyMs,urgency,urgency.confidence,refund,refund.confidence,churn,churn.confidence");
  assert.equal(rows.length, 2);
  // q1's row has no `message` (blank) and no `urgency`/`refund` answers (blank), but does have `churn`.
  const q1Row = rows[1]!.split(",");
  assert.equal(q1Row[3], ""); // message column, blank for q1
  assert.equal(q1Row[4], "field"); // other column
});

test("queueToCsv includes one row per outcome (compare mode fans out)", () => {
  const q = [
    entry({
      outcomes: [
        { label: "WebGPU f16", ms: 10, result: { answers: {} } },
        { label: "CPU reference", ms: 40, result: { answers: {} } },
      ],
    }),
  ];
  const rows = queueToCsv(q, "").split("\n").slice(1);
  assert.equal(rows.length, 2);
});

test("queueToCsv reports an error outcome's cells as blank rather than throwing", () => {
  const q = [entry({ outcomes: [{ label: "WebGPU f16", error: "boom" }] })];
  const rows = queueToCsv(q, "").split("\n").slice(1);
  assert.equal(rows.length, 1);
  assert.ok(!rows[0]!.includes("boom")); // error text isn't a CSV column; latency/answers are just blank
});

test("queueToCsv includes the priority value when a priority question is set", () => {
  const csv = queueToCsv([entry()], "refund");
  const cols = csv.split("\n")[1]!.split(",");
  const header = csv.split("\n")[0]!.split(",");
  assert.equal(cols[header.indexOf("priorityValue")], "0.75");
});
