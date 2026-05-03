---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 04 - Adnvaced Concepts.pdf
created: 2026-05-02
---

# Distributional hypothesis

The foundational assumption behind almost every statistical and neural approach to semantics in NLP:

> Words that occur in similar contexts tend to have similar meanings.

Rather than defining meaning explicitly (dictionaries, ontologies), vector-space models **infer semantic relationships from patterns of co-occurrence** across large corpora.

## Operationalization

If each word $w_i$ is represented by a vector
$$\vec{w}_i = (X_{i1}, X_{i2}, \ldots, X_{in})$$
formed from its [[term-document-matrix|term-document matrix]] row (or by its sliding-window co-occurrence row in [[hyperspace-analogue-to-language|HAL]]), then **words with similar usage patterns will have similar vectors**.

This is what makes:

- [[cosine-similarity]] a sensible measure of semantic similarity in vector space
- [[latent-semantic-analysis|LSA]] able to detect that *car* and *automobile* are related (they appear in similar contexts)
- [[word2vec|Word2Vec]] able to learn meaningful embeddings by predicting context (Quiz III Q9.B: "two words appearing in similar contexts will tend to have **similar vectors in embedding space**")

## Geometric view

Once language is embedded in a vector space, **geometry becomes the language of semantics**:

- **Similarity** is expressed through distance or angle between vectors, rather than exact matches
- **Semantic relations become graded, continuous, and quantitative** instead of all-or-nothing
- The space supports operations like analogies (king − man + woman ≈ queen)

## Why it matters

The distributional hypothesis is what justifies the entire vector-space program — it's the implicit promise that statistical regularities in usage carry semantic information. It is the **same idea** behind [[latent-semantic-analysis|LSA]], [[hyperspace-analogue-to-language|HAL]], [[word2vec|Word2Vec]], and contextual transformer embeddings, just operationalized differently.
