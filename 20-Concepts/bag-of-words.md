---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 04 - Adnvaced Concepts.pdf
  - course: NLP
    file: 03_BoW_modelling.ipynb
created: 2026-05-02
---

# Bag-of-Words (BoW)

A document representation where text is treated as an **unordered collection of words** — only word identity and frequency are retained. Grammar, word order, and syntax are discarded.

## Definition

For vocabulary $V = \{w_1, \ldots, w_{|V|}\}$, document $d$ becomes:
$$\mathbf{x}_d = (c(w_1, d), \ldots, c(w_{|V|}, d))$$

where $c(w, d)$ = number of occurrences of $w$ in $d$. Binary BoW uses $\mathbb{1}(w \in d)$ instead of count.

## The defining assumption

> Word order is irrelevant.

Two sentences with the same word counts but different orders are **treated as identical** by BoW (Quiz I Q27). Example:

- "The kid ate the cookie" → `[2,1,1,1]` over `[the, kid, ate, cookie]`
- "The cookie ate the kid" → `[2,1,1,1]` — same vector, different meaning.

## Exam-ready facts

| | Fact |
|---|---|
| What's true | Documents are word-count vectors; produces sparse vectors |
| What's true | BoW **ignores word order and only considers if/how often the word appears** |
| What's **false** | "BoW captures semantic similarity between words" — mock Q2 false-statement target |
| Why problematic | **Ignores semantic relationships** between words (Quiz I Q29) |
| Sequence-model advantage | RNNs / Transformers **capture word order and context** in ways BoW cannot (mock Q24) |

## Trade-off

The representation is **deliberately crude**. Its purpose is not to model language faithfully, but to make **large-scale statistical analysis** possible — fast, simple, compatible with linear models. It loses meaning to gain tractability.

## Pipeline place

BoW sits between [[tokenization]] and [[term-document-matrix]] / [[tf-idf]]: tokenize → count → reweight by TF-IDF (or normalize) → use as feature vector for [[naive-bayes]] / [[logistic-regression]] / similarity ranking.

## Where BoW falls short

- Word order matters for meaning — "dog bites man" ≠ "man bites dog"
- Same word, different sense (polysemy) collapsed
- Synonyms not connected — see [[vocabulary-mismatch]] and [[latent-semantic-analysis]]

## Canonical code skeleton — Counter (mock Q29 / Code 1)

The exam tests this exact idiom (predict-output on a Python snippet):

```python
from collections import Counter

tokens = [["nlp","is","fun"], ["nlp","is","powerful"], ["learning","nlp","is","fun"]]
vocab = sorted({t for sent in tokens for t in sent})

tf_matrix = []
for sent in tokens:
    counts = Counter(sent)
    tf_matrix.append([counts[w] for w in vocab])

print(vocab)         # ['fun','is','learning','nlp','powerful']
print(tf_matrix)     # [[1,1,0,1,0], [0,1,0,1,1], [1,1,1,1,0]]
```

**Critical insights:**
- `Counter[w]` returns **0 for missing keys** — no `KeyError`
- The set comprehension `{t for sent in tokens for t in sent}` flattens nested lists
- `sorted()` on a set returns a **list in alphabetical order** — vocab is reproducible
- Each row of `tf_matrix` is a document's count vector over the vocab; same length per row

The toy `tokens` list above is illustrative — the notebook itself runs on Moby Dick text (`gutenberg.raw('melville-moby_dick.txt')`). The **shape** of the idiom matches: see [BoW with Counter — sorted vocab + per-document Counter (cells 26, 31)](30-Sources/NLP/notebooks/03_BoW_modelling.ipynb).
