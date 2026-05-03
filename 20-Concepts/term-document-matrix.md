---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 04 - Adnvaced Concepts.pdf
created: 2026-05-02
---

# Term-document matrix

A corpus encoded as a single algebraic structure. Once each document has a [[bag-of-words|BoW]] representation, stacking them yields a matrix $X$ where:

$$X_{ij} = c(w_i, d_j) = \text{number of times word } i \text{ appears in document } j$$

Convention used in slides: **rows = words, columns = documents**. (The transpose is the *document-term matrix*; same content, different orientation.)

## Two views of the matrix

- **Column vectors** = the [[bag-of-words|BoW]] encoding of each document
- **Row vectors** = the **word vector** — encodes how that word distributes across the corpus

## Feeds into

- [[tf-idf]] — reweight each cell by document and corpus statistics
- [[latent-semantic-analysis]] — apply SVD to find latent semantic structure
- [[information-retrieval-ranking|Vector-space retrieval]] — rank documents against a query vector by [[cosine-similarity]]

The matrix is the base structure that almost every classical IR / topic-modelling / vector-space NLP technique builds on.
