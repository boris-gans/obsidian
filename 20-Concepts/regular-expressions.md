---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 06 - Regular Expressions-1.pdf
created: 2026-05-02
---

# Regular expressions

A **formal language for specifying patterns in text**: algebraic notation that characterizes a set of strings using a finite set of symbols and operators. Originally introduced by Stephen Kleene in automata theory; in NLP they appear during cleaning, preprocessing, and feature preparation.

## What regex see

Regular expressions treat text as a **sequence of characters** — letters, digits, punctuation, whitespace — without access to words, syntax, or meaning. They combine simple mechanisms (literals, wildcards, character classes, quantifiers, anchors, groups) into expressive patterns.

## Basic operators (on the formula sheet)

| Operator | Behaviour | Example |
|---|---|---|
| `.` | Wildcard, any character | `c.t` matches `cut, cat, cot` |
| `^abc` | Start of string | `^abc` matches `abc123`, not `xabc123` |
| `abc$` | End of string | `abc$` matches `123abc`, not `123abcc` |
| `[abc]` | Any in set | matches `a, b, or c` |
| `[A-Z0-9]` | Any in range | matches `k` or `7` |
| `ed\|ing\|s` | Disjunction | matches `ed`, `ing`, or `s` |
| `*` | Zero or more (Kleene closure) | `ab*` → `a, ab, abb, abbb` |
| `+` | One or more | `ab+` → `ab, abb`, not `a` |
| `?` | Zero or one | `colou?r` matches `color` and `colour` |
| `{n}` | Exactly n | `\d{4}` matches `2025` |
| `{n,}` | At least n | `a{2,}` matches `aa, aaa, aaaa` |
| `{,n}` | At most n | `a{,3}` matches `a, aa, aaa` |
| `{m,n}` | Between m and n | `a{2,4}` matches `aa, aaa, aaaa` |
| `(...)` | Group (scope of operators) | `(add\|ing\|s)+` → `addings, sing, adding` |

## Special character classes & escapes

| Class | Meaning | Example |
|---|---|---|
| `\d` | Any digit (0-9) | `\d+` → `123` |
| `\D` | Any non-digit | `\D+` → `abc` |
| `\w` | Word character (letters, digits, underscore) | `\w+` → `user_42` |
| `\W` | Any non-word | `@#$` |
| `\s` | Whitespace | spaces, tabs, newlines |
| `\S` | Non-whitespace | `text` |
| `\b` | Word boundary (**position**, not character) | `\bcat\b` matches `cat` but not `cats` |
| `\B` | Non-word boundary | `\Bcat` matches `scat` |
| `\A` | Start of string (position) | `\Aabc` matches `abc...` |
| `\Z` | End of string (position) | `abc\Z` matches `...abc` |

`\d`, `\w`, `\s` are character classes (consume a character). `\b`, `\A`, `\Z` are **positional** — they match positions in the string but consume nothing.

## Exam-ready idioms

- "Words ending in *ing*" → `\b\w+ing\b` (mock Q4 — note the trailing `\b` to prevent matching `bring`-mid-word)
- 4-digit year → `\d{4}`
- Email-like: `[\w.+-]+@[\w-]+\.[\w.-]+`
- Date `dd/mm/yyyy` → `\d{2}/\d{2}/\d{4}`

## Pipeline place

Regex are used for **data cleaning, preprocessing, and feature preparation** before any classifier or neural model is applied. They **protect the pipeline** at its boundaries — failures attributed to models are often failures of preprocessing.

> Regular expressions define what can be solved without uncertainty. Classifiers and transformers handle what remains.

## What regex don't do

They **cannot interpret meaning, resolve ambiguity, or capture long-range dependencies**. *Bank* in "the bank approved the loan" and "the river bank overflowed" is identical to a regex. When variation itself becomes meaningful, learning from data takes over.

See also [[regex-lookarounds]] and [[regex-greedy-vs-lazy]] for two off-sheet constructs heavily tested on the exam.
