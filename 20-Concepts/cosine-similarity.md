---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 04 - Adnvaced Concepts.pdf
created: 2026-05-02
---

# Cosine similarity

A geometric measure of similarity between two vectors based on the **angle** between them, not the magnitude.

## Definition

$$\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|\,\|\mathbf{y}\|} = \frac{\sum_i x_i y_i}{\sqrt{\sum_i x_i^2}\,\sqrt{\sum_i y_i^2}}$$

where the dot product $\mathbf{x}\cdot\mathbf{y} = \sum_i x_i y_i$ and the Euclidean norm $\|\mathbf{x}\| = \sqrt{\sum_i x_i^2}$ are both on the formula sheet.

Output: $\cos \in [-1, 1]$ (or $[0,1]$ for non-negative vectors like BoW / TF-IDF).

## What cosine measures

> The angle between vectors in vector space.

Mock Q12: "Cosine similarity between two vectors measures…" → **the angle between them in vector space**.

## Why it's preferred over Euclidean distance for documents

Cosine **focuses on direction rather than magnitude**. Two documents talking about the same topic — one short (small vector magnitudes), one long (large magnitudes) — should still be considered similar. Cosine **normalizes by vector magnitude**, removing **document-length effects** (Quiz II Q3, II.M2 Q3).

For pre-normalized unit vectors ($\|\mathbf{x}\| = 1$), cosine reduces to the dot product and gives the same ranking as Euclidean distance.

## Worked exam shapes

**$A = (2,1)$, $B = (1,3)$**:
$$\cos(A,B) = \frac{2\cdot1 + 1\cdot3}{\sqrt{5}\cdot\sqrt{10}} = \frac{5}{\sqrt{50}} \approx 0.71$$

(Quiz III Q1, Model A.)

**$A = (3,1)$, $B = (1,2)$**:
$$\cos(A,B) = \frac{3 + 2}{\sqrt{10}\sqrt{5}} = \frac{5}{\sqrt{50}} \approx 0.71$$

(Quiz III Q1, Model B — answer was rounded to 0.71, option (c).)

## In the broader family of similarity coefficients

| Coefficient | Formula | When useful |
|---|---|---|
| Dice | $\dfrac{2|X \cap Y|}{|X|+|Y|}$ | Set-theoretic overlap |
| Jaccard | $\dfrac{|X \cap Y|}{|X \cup Y|}$ | Set-theoretic, normalized differently |
| Overlap | $\dfrac{|X \cap Y|}{\min(|X|,|Y|)}$ | Asymmetric containment |
| **Cosine** | $\dfrac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$ | Continuous direction in vector space |

Cosine is the one used throughout the course because it's the natural similarity in continuous vector spaces and is what [[information-retrieval-ranking]], [[tf-idf]] retrieval, and [[word-embeddings]] all use.
