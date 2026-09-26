import { test } from "node:test";
import assert from "node:assert/strict";
import { mapJevResult, mapOnnxResult, toJevQuestions, NO_ACT_PROBABILITY } from "../src/mapping.ts";

test("mapOnnxResult passes rl_agent.act_probability through as action.act_probability", () => {
  const raw = {
    model: "laya-onnx",
    answers: {
      dept: { type: "choice" as const, choice: "billing", confidence: 0.8, probabilities: { billing: 0.8, sales: 0.2 }, rl_agent: { act_probability: 0.91 } },
      urgency: { type: "score" as const, score: 2.3, confidence: 0.6, legend: { "0": "low", "1": "mid", "2": "high" }, probabilities: { "0": 0.1, "1": 0.2, "2": 0.7 }, rl_agent: { act_probability: 0.4 } },
      churn: { type: "noul" as const, noul: 0.83, rl_agent: { act_probability: 0.6 } },
    },
    usage: { input_tokens: 42, output_tokens: 0 },
  };
  const result = mapOnnxResult(raw);
  assert.equal(result.model, "laya-rl-agent");
  assert.equal(result.usage.input_tokens, 42);
  assert.equal(result.answers.dept!.action.act_probability, 0.91);
  assert.equal(result.answers.dept!.choice, "billing");
  assert.equal(result.answers.urgency!.action.act_probability, 0.4);
  assert.equal(result.answers.urgency!.score, 2.3);
  assert.deepEqual(result.answers.urgency!.legend, { "0": "low", "1": "mid", "2": "high" });
  assert.equal(result.answers.churn!.action.act_probability, 0.6);
  assert.equal(result.answers.churn!.noul, 0.83);
  // laya-core's own noul-confidence formula: max(p, 1-p); ONNX doesn't report one of its own.
  assert.equal(result.answers.churn!.confidence, Math.max(0.83, 1 - 0.83));
});

test("mapJevResult marks act_probability as NO_ACT_PROBABILITY (NaN) -- Jev has no RL-agent act signal", () => {
  const raw = {
    model: "jev-latest",
    answers: {
      billing: { type: "noul" as const, noul: 0.2 },
      dept: { type: "choice" as const, choice: "sales", confidence: 0.7, probabilities: { sales: 0.7, support: 0.3 } },
      severity: { type: "score" as const, score: 1.5, confidence: 0.55, legend: { 0: "low", 1: "high" }, probabilities: { 0: 0.45, 1: 0.55 } },
    },
    usage: { input_tokens: 30, output_tokens: 12 },
  };
  const result = mapJevResult(raw as never);
  assert.ok(Number.isNaN(result.answers.billing!.action.act_probability));
  assert.equal(result.answers.billing!.action.act_probability, NO_ACT_PROBABILITY);
  assert.ok(Number.isNaN(result.answers.dept!.action.act_probability));
  assert.equal(result.answers.dept!.choice, "sales");
  assert.ok(Number.isNaN(result.answers.severity!.action.act_probability));
  assert.equal(result.answers.severity!.score, 1.5);
  // Jev's NoulResponse has no confidence field either -- same synthesized formula as ONNX.
  assert.equal(result.answers.billing!.confidence, Math.max(0.2, 1 - 0.2));
  // usage passes through untouched (unlike laya-core's own literal-0 output_tokens).
  assert.equal(result.usage.output_tokens, 12);
});

test("toJevQuestions converts array-form choice criteria to Jev's required label->description object form", () => {
  const questions = {
    dept: { type: "choice" as const, instructions: "Which department?", criteria: ["billing", "sales"] },
  };
  const converted = toJevQuestions(questions);
  assert.deepEqual(converted.dept, { type: "choice", instructions: "Which department?", criteria: { billing: null, sales: null } });
});

test("toJevQuestions leaves object-form choice criteria and other question types untouched", () => {
  const questions = {
    dept: { type: "choice" as const, instructions: "Which department?", criteria: { billing: "Invoices and payments", sales: null } },
    urgency: { type: "score" as const, instructions: "How urgent?", criteria: ["low", "mid", "high"] },
    isBilling: { type: "noul" as const, instructions: "Is this about billing?" },
  };
  const converted = toJevQuestions(questions);
  assert.deepEqual(converted, questions);
});
