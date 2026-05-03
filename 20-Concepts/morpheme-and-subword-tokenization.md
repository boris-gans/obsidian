---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 03 - Fundamental Concepts.pdf
created: 2026-05-02
---

# Morpheme and subword tokenization

Choosing the unit of [[tokenization]] is a **modelling decision**: it trades off linguistic structure, statistical robustness, and computational simplicity.

## The three options

**Words** are the most intuitive unit but present **practical difficulties**: large vocabularies due to inflections / derivations / compounds, and the **out-of-vocabulary** problem on rare or unseen words.

**Morphemes** — the smallest units of language carrying meaning or grammatical function. *unhappiness* → `un-` + `happy` + `-ness`. Morpheme-level representation **reduces data sparsity** and **captures shared structure** across related word forms (e.g. `connect`, `connecting`, `connection` share `connect`). The cost: requires linguistic analysis that may be complex or language-specific.

**Subword units** (e.g. BPE, WordPiece) — a **computational compromise**. Instead of relying on full words or linguistically defined morphemes, segment words into **frequently occurring fragments**. Models can handle rare or unseen words while avoiding explicit linguistic rules. This is what Hugging Face tokenizers do.

## Example: "unhappiness"

| Scheme | Result |
|---|---|
| Word | `[the, unhappiness, of, players]` |
| Morpheme | `[the, un-, happy, -ness, of, play, -er, -s]` |
| Subword | `[the, un, happi, ness, of, play, ers]` |

## Why it matters downstream

[[n-gram-model|N-gram models]] are **agnostic to what tokens represent** — they apply the same counting and prediction mechanism to whatever tokens you give them. Changing the tokenization changes the objects being counted, and therefore the probabilities the model learns. The choice cascades through the whole pipeline.

## Exam framing

[[stemming-vs-lemmatization|Stemming or lemmatization]] reduces vocabulary by merging different word forms (Quiz I Q5: "running" and "runs" → "run"). Subword tokenization addresses the same OOV / sparsity problem but without explicit rules.
