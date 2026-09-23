import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed
const test = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import assert from "node:assert/strict";
import { loadJson } from "@johnhenry/laya-fixtures";
import {
  analyse,
  detectScript,
  guessLatinLanguage,
  isEnglish,
  latinProfile,
  scriptProfile,
  stateText,
} from "@johnhenry/langdetect-lite";

interface Row {
  state: unknown;
  state_text: string;
  detect_script: string;
  script_profile: Record<string, number>;
  latin_profile: unknown;
  guess_latin_language: string | null;
  analyse: unknown;
  is_english: boolean;
}
const rows = await loadJson<Row[]>("tables", "lang.json");

test("every lang.json field matches Python exactly", () => {
  assert.ok(rows.length >= 30);
  for (const r of rows) {
    const text = stateText(r.state);
    const label = JSON.stringify(r.state);
    assert.equal(text, r.state_text, label);
    assert.equal(detectScript(text), r.detect_script, label);
    assert.deepEqual(scriptProfile(text), r.script_profile, label);
    // key order too (Python dict order is part of the output)
    assert.deepEqual(Object.keys(scriptProfile(text)), Object.keys(r.script_profile), label);
    assert.deepEqual(latinProfile(text), r.latin_profile, label);
    assert.equal(guessLatinLanguage(text), r.guess_latin_language, label);
    assert.deepEqual(analyse(r.state), r.analyse, label);
    assert.equal(isEnglish(r.state), r.is_english, label);
  }
});

// ---- ported from laya-mlx tests/test_router.py

test("detect_script", () => {
  const cases: Array<[string, string]> = [
    ["The customer was charged twice and wants a refund.", "latin"],
    ["Հայերեն", "armenian"],
    ["ՀԱՅԵՐԵՆ", "armenian"],
    ["։֊", "unknown"],
    ["Le client a été facturé deux fois et demande un remboursement.", "latin"],
    ["ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।", "devanagari"],
    ["お客様は二重に請求されたため返金を希望しています。", "kana"],
    ["客户被重复扣款要求退款", "han"],
    ["고객이 두 번 청구되어 환불을 원합니다", "hangul"],
    ["تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال", "arabic"],
    ["С клиента дважды сняли деньги и он хочет возврат", "cyrillic"],
    ["Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων", "greek"],
    ["הלקוח חויב פעמיים ורוצה החזר כספי", "hebrew"],
    ["", "unknown"],
    ["12345 6789", "unknown"],
  ];
  for (const [text, want] of cases) assert.equal(detectScript(text), want, text);
});

test("is_english", () => {
  const cases: Array<[string, boolean]> = [
    ["Please refund the duplicate charge on invoice 4411 today.", true],
    ["Հայերեն", false],
    ["refund me", true],
    ["ग्राहक से दो बार शुल्क लिया गया", false],
    ["お客様は二重に請求されました", false],
    [
      "Le client a été facturé deux fois et il demande un remboursement pour la " +
        "facture qui a été payée le mois dernier avec la carte de crédit",
      false,
    ],
    [
      "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die " +
        "Rechnung die nicht korrekt ist und auch nicht bezahlt wurde",
      false,
    ],
    ["Gătește-mi o rețetă de sarmale de post pentru mâine.", false],
    ["Am fost taxat de două ori pentru factura din luna martie și vreau banii", false],
    ["Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę", false],
    ["Zákazníkovi byla částka účtována dvakrát a žádá o vrácení peněz", false],
    ["Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım", false],
    ["Khách hàng đã bị thu phí hai lần và muốn được hoàn tiền ngay", false],
    [
      "We visited a cafe in Zurich and the naive assumption about the " +
        "invoice was wrong, so please refund the duplicate charge",
      true,
    ],
  ];
  for (const [text, want] of cases) assert.equal(isEnglish(text), want, text);
});

test("undecided latin is not dressed up as a detection", () => {
  const det = analyse("Müşteriden iki kez ücret alındı ve para iadesi istiyor");
  assert.equal(det.language_undecided, true);
  assert.equal(det.language, null);
  assert.equal(analyse("Please refund the duplicate charge on the invoice").language_undecided, false);
  assert.ok(analyse("Gătește-mi o rețetă de sarmale").diacritic_rate > 0.02);
  assert.equal(analyse("Please refund the duplicate charge today").diacritic_rate, 0.0);
});

test("analyse reports the same keys from every branch", () => {
  const keys = ["diacritic_rate", "is_english", "language", "language_undecided", "non_latin_fraction", "script", "script_profile"];
  for (const text of ["Please refund the duplicate charge", "ग्राहक से दो बार", "Gătește-mi o rețetă de sarmale", "12345 ???"]) {
    assert.deepEqual(Object.keys(analyse(text)).sort(), keys);
  }
});

test("zero stopword tie invents no language; known Romanian gap stays visible", () => {
  assert.equal(guessLatinLanguage("Cât e ora acum la Tokyo"), null);
  assert.equal(isEnglish("Care este ora in Tokyo?"), true);
});

test("guess_latin_language", () => {
  const cases: Array<[string, string | null]> = [
    ["The customer was charged twice and wants a refund for this invoice", "en"],
    ["Le client a ete facture deux fois et il demande un remboursement pour la facture", "fr"],
    ["Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung", "de"],
    ["El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura", "es"],
    ["refund", null],
  ];
  for (const [text, want] of cases) assert.equal(guessLatinLanguage(text), want, text);
});

test("state_text flattens and ignores keys", () => {
  assert.ok(stateText({ body: "charged twice", n: 3 }).includes("charged twice"));
  assert.ok(stateText({ a: { b: ["deep"] } }).includes("deep"));
  assert.ok(stateText(["x", { y: "z" }]).includes("x"));
  assert.equal(stateText(null), "");
  assert.equal(analyse({ subject: "नमस्ते", body: "ग्राहक से दो बार शुल्क लिया गया" }).is_english, false);
  // truncation counts code points, not UTF-16 units
  assert.equal(stateText("🙂".repeat(5), 3), "🙂🙂🙂");
});

test("script_profile armenian", () => {
  assert.deepEqual(analyse("Հայերեն").script_profile, { armenian: 1.0 });
  assert.equal(analyse("Հայերեն abc").non_latin_fraction, 0.7);
});
