---
tags: [flashcards, NLP]
---

# Lecture 06 — Regular Expressions (flashcards)

What is a regular expression formally?
?
A **formal language for specifying patterns in text** — an algebraic notation that characterizes sets of strings using a finite set of symbols and operators (Kleene, automata theory).

What kind of questions do regular expressions answer about text?
?
**Structural questions** — whether a string matches a format, where a pattern appears, and which substrings satisfy that pattern. Not meaning.

What can regular expressions NOT do?
?
**Interpret meaning, resolve ambiguity, or capture long-range dependencies.** *Bank* (river vs financial) is identical to a regex.

Where in the NLP pipeline do regex appear?
?
**Early** — data cleaning, preprocessing, feature preparation, before any classifier or neural model.

Write a regex that matches words ending in `ing`.
?
`\b\w+ing\b` (mock Q4). The trailing `\b` prevents matching mid-word like `singer`.

What does `\d{4}` match?
?
Exactly four digits — e.g. a year like `2025`.

What's the meaning of `colou?r`?
?
`color` or `colour` — the `?` makes the previous `u` optional (zero or one).

What does `(add|ing|s)+` match?
?
Strings like `addings, sing, adding` — one or more of `add`, `ing`, or `s`. The parentheses scope the alternation; `+` then applies one-or-more to the whole group.

What's the difference between `\d` and `\b`?
?
`\d` is a **character class** (matches one digit, consuming a character). `\b` is a **positional/zero-width assertion** (matches a position at a word boundary, no character consumed).

What is a lookaround?
?
A **zero-width assertion** that tests context without consuming characters — matches a position in the string. Examples: `(?=X)`, `(?!X)`, `(?<=X)`, `(?<!X)`.

Write a regex matching the word "data" only when NOT immediately preceded by a letter.
?
`(?<![a-zA-Z])data` — negative lookbehind asserts the previous character is not a letter, without consuming it.

Write a regex matching digits preceded by `$` without including the `$`.
?
`(?<=\$)\d+` — positive lookbehind for `$`, then capture `\d+`.

Why are lookarounds called zero-width assertions?
?
Because **they do not consume characters in the matched string** — they test context at a position without advancing the matching cursor.

What's the difference between greedy and lazy quantifiers?
?
**Greedy** (`*`, `+`, `?`, `{n,m}`) match as much text as possible. **Lazy** (`*?`, `+?`, `??`, `{n,m}?`) match as little as possible.

Apply `<.+>` and `<.+?>` to `<p>Pay €1,249.50 now</p>`. What's the difference?
?
Greedy `<.+>` matches the **entire** string from the first `<` to the last `>` — one big match. Lazy `<.+?>` matches **each tag separately** (`<p>`, `</p>`).

What does the pattern `.*?` match first on the string `"abcabc"`?
?
The **empty string**. The lazy `*?` can match zero characters, and the first match is empty.

What does `\b` mean in regex?
?
**Word boundary** — a zero-width position between a word character and a non-word character. `\bcat\b` matches `cat` but not `cats`.

What's the meaning of `\D`?
?
Any **non-digit** character — the complement of `\d`.

How do uppercase escape classes (`\D`, `\W`, `\S`) relate to lowercase ones (`\d`, `\w`, `\s`)?
?
They are **complements** — `\D` matches anything that is not a digit, `\W` anything not a word character, `\S` anything not whitespace.

What's the role of regex in modern NLP pipelines that use transformers?
?
They **protect the pipeline** at its boundaries — clean text, extract fields, normalize formatting, filter malformed inputs. Many failures attributed to models are failures of preprocessing.
