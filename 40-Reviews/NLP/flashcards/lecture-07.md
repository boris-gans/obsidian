---
tags: [flashcards, NLP]
---

# Lecture 07 — Information Retrieval (flashcards)

What problem does Information Retrieval address?
?
**Selecting relevant documents from a large collection** — given a query and a corpus, rank documents by relevance. The output is an ordered list, not a single answer.

Is Information Retrieval a task about understanding language?
?
**No.** It is fundamentally an **ordering problem** — comparing and ranking based on measurable criteria. It does not require interpreting meaning or intentions.

What primarily determines document ranking in classical IR?
?
**Statistical term weighting** (e.g. TF-IDF combined with cosine similarity). Not syntax, not semantics, not document length per se.

What's the difference between Information Retrieval and Information Extraction?
?
IR ranks documents by relevance (output: ordered list). IE identifies and structures specific pieces of information within documents (output: entities, relations, events).

What is "relevance"?
?
**Not an intrinsic property of a document** — it depends on the query, corpus, and task. Operational and context-dependent, not semantic in the strong sense.

Why is recall typically the limiting factor in classical IR?
?
Because of **vocabulary mismatch** — relevant documents that use different words from the query are missed. Precision is unaffected; recall suffers.

What does it mean if a search engine retrieves documents sharing many rare terms with a query but missing the user's intent?
?
This illustrates **lack of semantic understanding** — surface-level overlap doesn't capture meaning or intent.

Write the Rocchio relevance-feedback update equation.
?
$\vec{q}_{i+1} = \vec{q}_i + \dfrac{\beta}{R}\sum_j \vec{r}_j - \dfrac{\gamma}{S}\sum_j \vec{s}_j$ — push the query toward relevant docs ($\vec{r}$) and away from non-relevant ($\vec{s}$).

What does the Rocchio update show about relevance?
?
That relevance is **not static** — it is adaptive, depending on user interaction and the residual document collection.

Why must we precompute `doc_norms` for cosine-similarity-based retrieval?
?
**To compute cosine similarity and reduce document-length bias** — cosine divides by vector norms, normalizing for length.

Why does cosine similarity remove document-length effects?
?
Because it **normalizes vector magnitude** — the score depends on direction (term ratios), not on raw counts.

What does cross-entropy measure?
?
**How unexpected observed data are under a probabilistic model** — when the model assigns low probability to observed words, surprise (cross-entropy) is high.

Compute the entropy of a uniform binary distribution (probabilities 0.5/0.5) using natural log.
?
$H = -2 \cdot 0.5 \log 0.5 = \log 2$.

What does training logistic regression by maximum likelihood minimize?
?
**Cross-entropy with observed labels** — equivalently, negative log-likelihood.

A classifier assigns probability 1 to one label. What is the entropy?
?
**0** — certainty implies zero entropy.

Probabilities near 0.5 in a binary classifier indicate what about entropy?
?
**High entropy** — maximum uncertainty in binary classification.

In what sense does TF-IDF anticipate cross-entropy?
?
Both reward **statistically surprising** events — rare-but-observed terms (TF-IDF) and rare-but-observed words (cross-entropy) carry more information than predictable ones.

What's the bridge from "document relevance" to "word relevance"?
?
The same statistical logic that ranks documents against a query can rank candidate next-words against a context — ranking words by their **statistical association with context** is a simple language model.

What are the limits of statistical relevance models?
?
They do not capture **word order, compositional meaning, paraphrase equivalence, or world knowledge**. They operate entirely on **surface-level patterns of usage** — effective behaviour without semantic understanding.

What modern system architecture embodies "retrieval-based" Q&A?
?
A system that **retrieves relevant documents and uses them to condition the final answer** (mock Q22) — the RAG (retrieval-augmented generation) idea.
