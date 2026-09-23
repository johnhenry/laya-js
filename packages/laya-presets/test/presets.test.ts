/**
 * Oracles: the laya-fixtures email table (Python clean_email_body), the
 * ported laya-mlx tests/test_email.py, and — when a python3 is available and
 * ../laya-mlx is checked out (LAYA_MLX_DIR) — a live differential run of
 * presets.py / email.py (loaded by file path; no MLX needed). Skips otherwise.
 */
// @ts-ignore -- bun types are not installed
const bun = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
import { env, makeTest } from "./harness.ts";
const test = makeTest(bun);
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadJson } from "@johnhenry/laya-fixtures";
import { toInternal } from "@johnhenry/laya-core";
import * as presets from "../src/index.ts";

test("cleanEmailBody matches the Python email table", async () => {
  const table = await loadJson<{ body: string; clean: string }[]>("tables", "email.json");
  assert.ok(table.length >= 6);
  for (const { body, clean } of table) assert.equal(presets.cleanEmailBody(body), clean, JSON.stringify(body));
});

const DISCLAIMER = "This email is confidential and intended solely for the named addressee.";
const CASES: [string, string][] = [
  // the request survives the footer
  [`My account is locked.\n${DISCLAIMER}\nPlease unlock it.`, "My account is locked. Please unlock it."],
  [`My account is locked\n${DISCLAIMER}\nPlease unlock it.`, "Please unlock it."],
  [`My account is locked. ${DISCLAIMER}`, "My account is locked."],
  // a pure footer is still removed
  [`My account is locked.\n\n${DISCLAIMER}`, "My account is locked."],
  ["My account is locked.\n\nThis email and any files transmitted with it are\nconfidential and intended solely for the named addressee.", "My account is locked."],
  ["Please reopen ticket 4411.\n\nIf you have received this message in error, delete it.", "Please reopen ticket 4411."],
  // unrelated cleaning is unchanged
  ["Thanks for the update.\nOn Mon, Sep 20, Bob wrote:\n> original text", "Thanks for the update."],
  ["Hi team,\nCan you confirm the refund?\nRegards,\nAlice", "Hi team,\nCan you confirm the refund?"],
  ["", ""],
];

test("cleanEmailBody: ported test_email.py cases", () => {
  for (const [body, want] of CASES) assert.equal(presets.cleanEmailBody(body), want, JSON.stringify(body));
});

test("emailState keeps the request, adds sender and extra fields", () => {
  assert.equal(presets.emailState("Locked out", `My account is locked. ${DISCLAIMER}`).body, "My account is locked.");
  assert.deepEqual(presets.emailState("  Hi  ", "x\\ny", { sender: "a@b.c", extra: { id: 7, skip: null } }), {
    subject: "Hi",
    body: "x\ny",
    from: "a@b.c",
    id: 7,
  });
  assert.deepEqual(presets.emailState(null, "raw  body", { clean: false }), { subject: "", body: "raw  body" });
});

test("every preset is a valid question set", () => {
  const sets = [presets.triageQuestions(), presets.emailQuestions(), presets.guardQuestions(), presets.moderationQuestions(), presets.routerQuestions()];
  for (const qs of sets) for (const q of Object.values(qs)) toInternal(q);
  assert.deepEqual(Object.keys((presets.emailQuestions({ a: "x" }).category as { criteria: object }).criteria), ["a"]);
  assert.deepEqual(presets.emailQuestions({}), presets.emailQuestions()); // `categories or {...}`
});

// ---------------------------------------------------------------- live Python oracle
const laya = env.LAYA_MLX_DIR ?? fileURLToPath(new URL("../../../../laya-mlx/", import.meta.url));
const python = env.MATH_PLUS_ORACLE_PYTHON ?? "python3";
const ORACLE = String.raw`
import importlib.util, json, sys
def mod(name):
    spec = importlib.util.spec_from_file_location(name, sys.argv[1] + "/laya_mlx/" + name + ".py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
p, e = mod("presets"), mod("email")
bodies = json.loads(sys.stdin.read())
print(json.dumps({
  "presets": {n: getattr(p, n)() for n in ["triage_questions", "email_questions", "guard_questions", "moderation_questions", "router_questions"]},
  "email_questions": e.email_questions(),
  "clean": [e.clean_email_body(b) for b in bodies],
  "clean_short": [e.clean_email_body(b, 17) for b in bodies],
  "state": e.email_state(" S ", bodies[0], "me@x", ticket=5, none=None),
}, ensure_ascii=False))
`;
const BODIES = [
  ...CASES.map(([b]) => b),
  "Hello\r\nSecond line\rThird\\nFourth",
  "Please help.\n\n\n\nBest regards,\nBob\n\nSent from my iPhone",
  "Kind regards\nnot a signature because it is the first line?",
  "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nCheers!\nl",
  "Grüße aus München.   Tabs\t\there.\n\n________\nold",
  "Line\n---------- Forwarded message ----------\nFrom: x",
  "Ünïcödé 🙂 text. It is confidential. Intended for the recipient only. Keep this.",
  "> quoted first\nreal line\n  > indented quote\nend",
  "From: first line is kept\nbody",
  "Thanks, Jürgen\nfoo",
];
let oracle: any = null;
let skip: string | false = false;
if (!existsSync(`${laya}/laya_mlx/email.py`)) skip = `laya-mlx not found at ${laya} (set LAYA_MLX_DIR)`;
else {
  try {
    oracle = JSON.parse(execFileSync(python, ["-c", ORACLE, laya], { input: JSON.stringify(BODIES), encoding: "utf8" }));
  } catch (e) {
    skip = `${python} failed: ${(e as Error).message.split("\n")[0]}`;
  }
}

test("presets and email match live Python (presets.py, email.py)", () => {
  assert.deepEqual(presets.triageQuestions(), oracle.presets.triage_questions);
  assert.deepEqual(presets.emailQuestions(), oracle.presets.email_questions);
  assert.deepEqual(presets.emailQuestions(), oracle.email_questions);
  assert.deepEqual(presets.guardQuestions(), oracle.presets.guard_questions);
  assert.deepEqual(presets.moderationQuestions(), oracle.presets.moderation_questions);
  assert.deepEqual(presets.routerQuestions(), oracle.presets.router_questions);
  BODIES.forEach((b, i) => {
    assert.equal(presets.cleanEmailBody(b), oracle.clean[i], JSON.stringify(b));
    assert.equal(presets.cleanEmailBody(b, 17), oracle.clean_short[i], JSON.stringify(b));
  });
  assert.deepEqual(presets.emailState(" S ", BODIES[0], { sender: "me@x", extra: { ticket: 5, none: null } }), oracle.state);
}, { skip });
