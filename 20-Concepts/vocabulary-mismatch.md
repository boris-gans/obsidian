---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 07 - Information Retrieval.pdf
created: 2026-05-02
---

# Vocabulary mismatch

The phenomenon where two pieces of text refer to the same concept using **different words**, causing surface-overlap-based methods (like [[tf-idf|TF-IDF]] retrieval) to underestimate their similarity. The classic example: "car" vs "automobile", or "buy" vs "purchase".

## Why it matters for IR

> Vocabulary mismatch is the main factor that limits **recall** in classical information retrieval (Quiz II.M3 Q13).

If a relevant document uses different words than the query, [[tf-idf|TF-IDF]] retrieval will rank it low — there's no surface overlap to score. The user fails to find what they want even though the corpus contains it. This is **lack of semantic understanding** at the system level (Quiz II Q12, II.M2 Q11).

## What it illustrates

Quiz II Q12: "A search engine retrieves documents sharing many rare terms with a query but missing the user's intent. Which limitation is illustrated?" → **Lack of semantic understanding.**

The system can score high on lexical overlap and still fail to satisfy the query. The terms that overlap may not be the ones that capture the user's intent.

## Two ways to address it

| Approach | How |
|---|---|
| **Latent topic models** | [[latent-semantic-analysis|LSA]] / LDA — let documents and terms align in a latent space where *car* and *automobile* end up close |
| **Dense embeddings** | [[word2vec|Word2Vec]], [[word-embeddings|contextual embeddings]] — words with similar meaning have similar vectors by construction |
| **Query expansion** | Add synonyms / related terms to the query; can be driven by [[relevance-feedback|relevance feedback]] |

Modern dense retrieval (e.g. retrieval-based / RAG systems) is essentially the dense-embedding solution operationalized at scale.

## Why it's a recall problem, not a precision problem

A retrieval system that misses paraphrased relevant documents has **low recall** — it fails to find documents that exist. Precision is unaffected (the documents it does return are still on-topic). This makes recall the right metric to evaluate paraphrase / synonym handling.
