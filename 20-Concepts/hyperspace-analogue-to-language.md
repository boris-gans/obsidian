---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 04 - Adnvaced Concepts.pdf
created: 2026-05-02
---

# Hyperspace Analogue to Language (HAL)

A vector-space model that addresses a **different** limitation of [[tf-idf|TF-IDF]] than [[latent-semantic-analysis|LSA]]: TF-IDF treats documents as **unordered bags**, ignoring how words interact in **running text**.

## Method: sliding-window co-occurrence

HAL constructs each word's vector from **co-occurrence within a sliding window** of running text. Each entry of a word's vector reflects how often other words appear near it, often weighted by **inverse distance** (closer co-occurrences count more).

This captures **fine-grained semantic associations** that document-level statistics miss — including **directional and asymmetric** relationships.

## Local vs global

HAL and LSA are complementary strategies for enriching BoW:

| Aspect | LSA | HAL |
|---|---|---|
| Scope | **Global** patterns across documents | **Local** context within a sliding window |
| Mathematical tool | SVD on term-doc matrix | Co-occurrence counts in window |
| Captures | Latent topical / semantic factors | Fine-grained directional associations |
| Strength | Solves vocabulary mismatch | Captures collocational meaning |

Together they illustrate the two complementary ways to enrich [[bag-of-words|BoW]]: **one global and latent, the other local and contextual**.

## Why it foreshadows modern embeddings

The same intuition — meaning from local co-occurrence — drives [[word2vec|Word2Vec]]'s skip-gram and CBOW, which predict context within a window rather than counting it directly. HAL is conceptually the bridge from classical co-occurrence statistics to modern predictive embeddings.
