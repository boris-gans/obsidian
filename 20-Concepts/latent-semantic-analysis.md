---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 04 - Adnvaced Concepts.pdf
created: 2026-05-02
---

# Latent Semantic Analysis (LSA)

A technique that addresses [[tf-idf|TF-IDF]]'s reliance on **direct word overlap** by introducing **latent semantic factors** through matrix factorization.

## Core idea

Two documents may discuss the same topic with different vocabularies (*car* vs *automobile*). TF-IDF underestimates their similarity because they share no words. LSA assumes that observed word usage is driven by a **smaller number of latent semantic factors** — recover those factors and similarity emerges.

## Method: SVD on the term-document matrix

Apply Singular Value Decomposition to the [[term-document-matrix]] $X$:
$$X = U\Sigma V^T$$

| Quantity | What it captures |
|---|---|
| $U$ | term-to-latent-topic matrix |
| $\Sigma$ | diagonal of singular values (importance) |
| $V^T$ | document-to-latent-topic matrix |

Keep the top $k$ singular values (largest variance directions) — **dimensionality reduction**:
$$X_k = U_k \Sigma_k V_k^T$$

This yields a low-rank semantic approximation of $X$.

## What the singular values mean

- Diagonal entries of $\Sigma$ = **singular values describing variance** (Quiz III Q4 Model A)
- The largest singular values explain the largest variance (Q4 Model B: 5 > 3 > 1)
- LSA dimensionality reduction = **selecting the largest singular values** (Q3 Model B)

## What LSA gives you

- Documents and terms embedded **together** in a shared lower-dimensional space
- Semantic similarity = **alignment along latent dimensions**, not exact term match
- Related documents and terms can be connected **even when they never appear together explicitly** — the *car* / *automobile* problem is solved

## Exam framing

| Question | Answer |
|---|---|
| What technique factorizes the term-document matrix? | **SVD** (Quiz III Q3) |
| What is the main idea behind LSA? | **Matrix factorization to reduce dimensionality** (mock Q7) |

## Conceptual significance

LSA marks a transition: **from weighted counting to geometry**, where meaning is inferred from position within a structured space rather than from individual features. It anticipates dense [[word-embeddings]] and modern transformer representations — both inherit LSA's idea that semantics emerges from latent structure across a corpus.

## Limits

LSA captures **global** patterns of co-occurrence but is insensitive to **local context** within sentences. That's the problem [[hyperspace-analogue-to-language|HAL]] addresses — the complementary local approach.
