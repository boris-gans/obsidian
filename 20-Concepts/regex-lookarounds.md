---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 06 - Regular Expressions-1.pdf
created: 2026-05-02
---

# Regex lookarounds

**Zero-width assertions** that test the immediate context of a position without **actually consuming any characters**. They match positions in the string, not characters — they assert that something is (or is not) there, while keeping the captured substring clean and unchanged.

Lookarounds are **NOT on the formula sheet** — memorization required. Heavily targeted on the exam (Quiz II Q4, II.M2 Q4, II.M2 Q12, II.M3 Q11).

## The four lookarounds

| Construct | Name | Meaning | Example |
|---|---|---|---|
| `(?=X)` | Positive **lookahead** | followed by X | `\d(?=%)` matches `5` in `5%` (without consuming the `%`) |
| `(?!X)` | Negative lookahead | NOT followed by X | `a(?!n)` matches `a` in `apple` but not in `and` |
| `(?<=X)` | Positive **lookbehind** | preceded by X | `(?<=\$)\d+` matches `120` in `$120` (without consuming the `$`) |
| `(?<!X)` | Negative lookbehind | NOT preceded by X | `(?<!-)cat` matches `cat` but not `-cat` |

## Why "zero-width"

> They do not consume characters in the matched string.

They assert context **at a position** without advancing the matching cursor. They can succeed or fail (Quiz II.M2 Q12); they are not "always matching empty" — that's a separate idea.

## Worked exam shapes

**"Match the word `data` only when not immediately preceded by a letter":**
```
(?<![a-zA-Z])data
```
The negative lookbehind asserts the previous character is not a letter, **without consuming it**. `\$data` includes the `$`; `[a-zA-Z]data` requires a preceding letter (the opposite); `^data$` is too restrictive (anchors the entire string).

(Quiz II Q4 answer: `(?<![a-zA-Z])data`.)

**"Match digits only when preceded by `$`, without including it":**
```
(?<=\$)\d+
```
Positive lookbehind. `\$\d+` would include the `$` in the match; `\d+(?=\$)` requires the `$` *after* the digits (lookahead, not lookbehind).

(Quiz II.M2 Q4 answer: `(?<=\$)\d+`.)

**"Digits not preceded by another digit":**
```
(?<!\d)\d+
```
(Quiz II.M3 Q11 answer.)

## Why this matters

In NLP preprocessing, you frequently need to **extract a substring whose context matters** but whose context **shouldn't pollute the extracted value** — currency amounts (`$120` → just `120`), percentages (`5%` → just `5`), prefixed identifiers, and so on. Lookarounds give you precise extraction without altering surrounding text.
