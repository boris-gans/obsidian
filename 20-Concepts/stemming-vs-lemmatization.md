---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 03 - Fundamental Concepts.pdf
created: 2026-05-02
---

# Stemming vs lemmatization

Both are preprocessing techniques that reduce **morphological variation** by mapping different word forms to a common representation, increasing **statistical robustness** at the cost of linguistic precision.

> Q5 (mock): "Which preprocessing step reduces vocabulary size by merging different word forms such as *running* and *runs*?" → **Stemming or lemmatization**.

## Stemming

A **rule-based truncation** algorithm that maps related forms to the same string, often a non-word. Example:

```
connect, connected, connecting, connection  →  connect
```

Or sometimes to a stem that isn't itself a word: `citi` for `city`. Stemming prioritizes **statistical simplicity** — fast, language-light.

## Lemmatization

Maps each form to its **dictionary base form (lemma)**, which depends on the **grammatical role**. *Better* → *good*; *was* → *be*. Lemmatization preserves more linguistic structure but is more expensive: it needs a lexicon and POS information.

## Trade-off

| | Stemming | Lemmatization |
|---|---|---|
| Output | Often non-word stems | Real dictionary lemmas |
| Cost | Cheap, rule-based | Needs lexicon + POS |
| Linguistic precision | Low | High |
| Statistical robustness | High | High |

Both reduce vocabulary size and merge surface variants — the choice depends on whether you want statistical simplicity (stemming) or linguistic structure (lemmatization).
