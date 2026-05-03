---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 07 - Information Retrieval.pdf
created: 2026-05-02
---

# Information retrieval (ranking)

The problem of **selecting relevant information from a large collection of documents**. Given a user query and a document corpus, the goal is to **rank documents** by how relevant they are to the query.

> The output of an Information Retrieval system is **not a single answer, but an ordered list of documents**.

IR is **fundamentally an ordering problem**, where documents are compared and ranked based on measurable criteria. It does not require the system to interpret meaning, intentions, or truth — same lesson as [[eliza|ELIZA]]: effective behaviour without semantic understanding.

## What determines ranking

In classical IR, document ranking is primarily determined by **statistical term weighting** (Quiz II.M2 Q7) — most prominently [[tf-idf|TF-IDF]] combined with [[cosine-similarity|cosine similarity]]. Not syntactic correctness, not semantic equivalence, not document length per se (though length influences indirectly through normalization).

## Why precompute document norms?

Standard implementation:
```python
doc_norms = np.linalg.norm(tfidf_docs, axis=1)
scores = (tfidf_docs @ q) / (doc_norms * np.linalg.norm(q))
```
The `doc_norms` are needed **to compute cosine similarity and reduce document-length bias** (Quiz II.M3 Q16). Cosine divides by norms, removing length effects.

## Failure modes (the limits of classical IR)

| Failure | Cause | Fix direction |
|---|---|---|
| Misses paraphrases | [[vocabulary-mismatch]] (different words, same meaning) | Dense [[word-embeddings]] |
| Lexical overlap, missed intent | **Lack of semantic understanding** (Quiz II Q12) | Semantic / neural retrievers |
| Recall too low | Vocabulary mismatch (Quiz II.M3 Q13) | Query expansion, dense retrieval |

Mock Q22: a "retrieval-based system" **retrieves relevant documents and uses them to condition the final answer** — the modern RAG idea — built on top of classical IR.

## What IR is NOT

- Not Information **Extraction** — IE produces structured output (entities, relations, events) by analysing within documents (slide 116).
- Not semantic understanding — classical IR can rank well even when the model has no idea what the documents are about. Quiz II.M3 Q18: "Retrieval models always capture meaning" is **false**.

## Adaptive notion of relevance

Through [[relevance-feedback]], relevance is **not static**: the query evolves based on user interaction with the ranked output. The Rocchio update equation pushes the query vector toward documents marked relevant and away from those marked non-relevant.
