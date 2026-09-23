# @johnhenry/langdetect-lite

Script and Latin-language detection with no model and no data files: a faithful port of
laya-mlx `laya_mlx/lang.py` (from [Laya](https://github.com/NandhaKishorM/laya), Apache-2.0).
It answers one routing question: *can an English-only checkpoint read this text?*

```js
import { analyse, isEnglish, detectScript, guessLatinLanguage } from "@johnhenry/langdetect-lite";

detectScript("客户被重复扣款要求退款");        // "han"
isEnglish("Please refund invoice 4411");         // true
isEnglish("Gătește-mi o rețetă de sarmale");     // false (Romanian diacritics)
guessLatinLanguage("Le client a ete facture deux fois et il demande un remboursement pour la facture"); // "fr"
analyse({ subject: "Refund", body: "Der Kunde wurde zweimal belastet und nicht erstattet" });
// { script: "latin", script_profile: { latin: 1 }, language: "de", is_english: false,
//   language_undecided: false, diacritic_rate: 0, non_latin_fraction: 0 }
```

## API

All functions accept a *state*: a string, or any JSON-like value whose string leaves
(object values and array items, up to depth 6) are joined with spaces. Keys are ignored.

| Export | Python (`laya_mlx.lang`) |
|---|---|
| `stateText(state, maxChars = 4000)` | `state_text` |
| `detectScript(text)` → `"latin" \| "han" \| ... \| "unknown"` | `detect_script` |
| `scriptProfile(text)` → `{ script: fraction }` | `script_profile` |
| `latinProfile(text)` → `{ language, english_hits, diacritic_rate, looks_non_english }` | `latin_profile` |
| `guessLatinLanguage(text)` → `"en" \| "fr" \| "de" \| "es" \| "pt" \| "it" \| "nl" \| "ro" \| null` | `guess_latin_language` |
| `analyse(state)` → `Analysis` | `analyse` |
| `isEnglish(state)` | `is_english` |
| `SCRIPT_RANGES`, `STOPWORDS`, `NON_EN_DIACRITICS`, `NON_EN_DIACRITIC_RATE` | module constants |

snake_case aliases (`state_text`, `detect_script`, ...) are exported too. Result objects
keep Python's snake_case keys so they compare equal to the Python output; every field of
the fixture table (`lang.json`, 35 states) matches exactly, floats included (rounding uses
`pyRound` from `@johnhenry/pyjson`, the only dependency).

Python semantics are reproduced where JavaScript differs: iteration and truncation by code
point (not UTF-16 unit), `str.isalpha()` as Unicode `\p{L}`, and `re` `[^\W\d_]+` as
`[\p{L}\p{Nl}\p{No}]+` (letters plus non-decimal numerics; combining marks split words,
as in Python).

## Limitations

- The Latin-language guess is a stopword/diacritic heuristic for 8 languages. Short input
  usually returns `null` (undecided) on purpose; Romanian without diacritics that contains an
  English function word reads as English. Pass an explicit language when you know it.
- Unicode tables come from the JavaScript engine, not from Python's `unicodedata`; the two
  can disagree on characters added in different Unicode versions.
- A state object's leaf order follows JavaScript key order (integer-like keys first), which
  only matters for the 4000-character truncation.
