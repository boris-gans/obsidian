---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 04 - Adnvaced Concepts.pdf
created: 2026-05-02
---

# One-hot encoding

The first numerical representation of words: each word becomes a vector with a single 1 at its position in the vocabulary and 0s elsewhere. For vocabulary size $V$, every word lives on a coordinate axis of $\mathbb{R}^V$.

## Example

Sentence: "I think it will rain". Vocabulary: `[I, think, it, will, rain]`.

| Word | Vector |
|---|---|
| I | `[1,0,0,0,0]` |
| think | `[0,1,0,0,0]` |
| it | `[0,0,1,0,0]` |
| will | `[0,0,0,1,0]` |
| rain | `[0,0,0,0,1]` |

Sentence as matrix: `[[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]` — a $p \times q$ matrix where $p$ = number of words and $q$ = vocabulary size.

## Advantages

- One-hot vectors form an **orthonormal basis** of $\mathbb{R}^V$ — features are independent.
- **Compatible with linear and probabilistic models**: simple geometry, convex optimization, interpretable parameters.
- Preserves the **symbolic nature** of the data — a word is a discrete symbol, not a point in a semantic continuum.

## Disadvantages

- Does not scale to large vocabularies — extremely sparse, high-dimensional vectors.
- **Carries no information about similarity or meaning** — `cosine(king, queen) = 0` just like `cosine(king, banana) = 0`. Mock Q23: "**One-hot vectors capture similarity between words**" is **false**.
- Forces the model to learn structure from scratch.

## Why it persists despite being "obsolete"

Modern systems still start from one-hot tokens — but they pass them through an **embedding layer** that learns a transformation into a dense space ([[word-embeddings]]). The embedding lookup `x_w = E[w]` is just a one-hot vector multiplied by the embedding matrix. So one-hot is hidden, not gone.
