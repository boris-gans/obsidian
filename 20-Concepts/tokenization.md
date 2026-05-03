---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 03 - Fundamental Concepts.pdf
created: 2026-05-02
---

# Tokenization

The first stage in most NLP pipelines: **segmenting raw text into smaller units (tokens)** — the basic units assumed by downstream NLP algorithms. A token may be a full word, a subword, a morpheme, or even a single character — see [[morpheme-and-subword-tokenization]].

> Q1 (mock): tokenization's main purpose is to **split text into smaller units such as words or subwords**.

## Why it's not a trivial step

Tokens are **not given by nature**. The choice of unit defines the **vocabulary** of the model and shapes the statistical structure that will be learned downstream. Tokenization therefore introduces representational assumptions before any learning happens.

Different schemes yield different views of language:

| Scheme | Example for "unhappiness" | Trade-off |
|---|---|---|
| Word | `unhappiness` | Largest vocabulary; OOV problems |
| Morpheme | `un-`, `happy`, `-ness` | Captures shared meaning; needs linguistic analysis |
| Subword (BPE / WordPiece) | `un`, `happi`, `ness` | Compromise; handles rare/unseen words |

## Place in the pipeline

[[nlp-pipeline]] step 2. **Tokenization must occur before counting word frequencies** (Quiz I Q43): you can't tabulate what you haven't segmented.

## Output

`word_tokenize('NLP is fun')` returns a **list of tokens** (Quiz I Q41) — e.g. `['NLP', 'is', 'fun']`. Note the common bug `tokens = nltk.word_tokenize` (not called) — assigns the function, doesn't tokenize anything (Quiz I Q42).

## Counting tokens

Recurring exam shape:

- "Language models learn language" → 4 tokens (Quiz I Q25)
- "Models model models" → 2 distinct tokens (Quiz I Q26)

## Related preprocessing decisions

- [[stemming-vs-lemmatization]] — reduce different word forms to a common base
- [[stop-words]] — drop frequent low-content tokens
- Lowercasing, punctuation stripping
