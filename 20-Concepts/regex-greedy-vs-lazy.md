---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 06 - Regular Expressions-1.pdf
created: 2026-05-02
---

# Regex greedy vs lazy quantifiers

By default, regex quantifiers (`*`, `+`, `?`, `{n,m}`) are **greedy** — they match as much text as possible. Adding `?` after a quantifier makes it **lazy** — match as little as possible.

**Lazy quantifiers are NOT on the formula sheet.** Quiz II Q13, Q19 target this.

## Greedy vs lazy table

| Greedy | Lazy | Behaviour |
|---|---|---|
| `*` | `*?` | zero or more (max vs min) |
| `+` | `+?` | one or more (max vs min) |
| `?` | `??` | zero or one (one vs zero) |
| `{n,m}` | `{n,m}?` | between n and m (max vs min) |

## The HTML-tag trap (Quiz II Q13)

Apply each pattern to: `<p>Pay €1,249.50 now</p>`

- `<.+>` (greedy) matches `<p>Pay €1,249.50 now</p>` — the entire string from the first `<` to the **last** `>`.
- `<.+?>` (lazy) matches `<p>` first, then `</p>` — each tag separately, **minimal matching**.

Quiz II Q13: "What does `<.+?>` enforce when applied to HTML-like text?" → **Minimal matching**.

The same idea is `<[^>]+>` (greedy but constrained to not cross `>`), which also produces individual tags (Quiz II.M2 Q13).

## The empty-match trap (Quiz II Q19)

What does `.*?` match on `"abcabc"`?

- `.*` would match the whole string (greedy)
- `.*?` matches the **empty string** as its first match (lazy can match zero characters)

Quiz II Q19 answer: **Empty string** — the lazy `*?` can match zero characters, and the first match is empty.

## Why this matters in practice

When delimiters can repeat (HTML tags, quoted strings, code blocks), greedy matching surprises. Lazy quantifiers fix this — they tell the engine "stop as soon as you can". For NLP preprocessing tasks like stripping HTML or extracting fields with delimiters, lazy quantifiers (or constrained character classes like `[^>]+`) are the right tool.
