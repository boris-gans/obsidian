---
tags: [flashcards, NLP]
---

# Lecture 13 — Word Embeddings (flashcards)

What is a word embedding?
?
A **dense vector** $v_w \in \mathbb{R}^d$ representing a word, where $d \ll V$ (vocabulary size). Vectors are **learned** by gradient descent on a context-prediction task — semantic similarity becomes geometric proximity.

What is the size of the embedding matrix?
?
$V \times d$ — one row per vocabulary word, $d$ columns for the embedding dimension. Total parameters = $V \cdot d$.

If $V = 10{,}000$ and $d = 100$, how many parameters are in the embedding matrix?
?
$10{,}000 \times 100 = 1{,}000{,}000$ parameters.

What's the difference between CBOW and Skip-gram?
?
**CBOW** predicts the **centre word** given surrounding context. **Skip-gram** predicts the **surrounding context words** given the centre word. Same embedding matrix, opposite prediction direction.

Which Word2Vec formulation handles rare words better, and why?
?
**Skip-gram** — each centre word generates one training pair per context position, giving rare words more gradient updates than CBOW (which makes a single prediction per centre).

Write the Skip-gram softmax for $P(c \mid w)$.
?
$P(c \mid w) = \dfrac{\exp(v_c^\top v_w)}{\sum_{w' \in V} \exp(v_{w'}^\top v_w)}$ — softmax over inner products with every vocabulary word.

What's the computational problem with the full Skip-gram softmax?
?
The denominator sums over all $V$ words. For $V = 10^5$ and $\sim 10^9$ training pairs, this is infeasible — motivates **negative sampling**.

State the negative-sampling objective.
?
$\log \sigma(v_c^\top v_w) + \sum_{i=1}^{k} \log \sigma(-v_{n_i}^\top v_w)$ — pull true word–context pairs together via $\sigma(\cdot)$, push $k$ randomly sampled noise pairs apart via $\sigma(-\cdot)$.

What distribution are the noise words sampled from in negative sampling?
?
A **unigram distribution** over the vocabulary.

Negative sampling reframes the prediction task as what kind of classifier?
?
**Logistic regression on word pairs** — binary classification: is this $(w, c)$ a real co-occurrence or noise?

What does the Skip-gram-with-negative-sampling model implicitly factorize?
?
A **shifted PMI matrix**: $v_w^\top v_c \approx \mathrm{PMI}(w, c) - \log k$ where $k$ is the number of negatives. So Word2Vec is in the same family as LSA — same target, different optimization route.

How does Word2Vec's training mechanism differ from LSA's?
?
**LSA** decomposes a precomputed term-document matrix via **SVD** (global, one-shot). **Word2Vec** updates vectors **incrementally via stochastic gradient descent** on each word–context pair (local, online). LSA uses counts; Word2Vec uses prediction.

Why are dense embeddings preferred over one-hot vectors?
?
(1) **Encode semantic similarity** — similar words have similar vectors; (2) **Generalize better** to unseen contexts; (3) **Lower dimensionality** ($d \ll V$). One-hot vectors are orthogonal — every word is equidistant from every other.

What relationship do word embeddings empirically capture beyond similarity?
?
**Linear analogies**: $v_{king} - v_{man} + v_{woman} \approx v_{queen}$ — relations encode as approximately linear translations.

Do the axes of an embedding space have semantic meaning?
?
**No.** Meaning is **distributed across all $d$ dimensions**. Unlike LSA (where axes correspond to variance directions), embedding axes have no direct interpretation.

Why is Word2Vec called a "shallow neural network"?
?
It has **only one hidden layer** — the embedding layer itself. There's no deeper architecture; the trained weight matrix $W \in \mathbb{R}^{V \times d}$ *is* the model.

What problem do static word embeddings still fail to solve?
?
**Polysemy.** "bank" gets a single vector that averages its finance and river senses. Resolving polysemy requires **context-dependent representations** — the conceptual step that motivates **transformers**.

What does the distributional hypothesis claim?
?
Words occurring in similar contexts tend to have similar meanings. Word2Vec operationalizes this by training words appearing in similar contexts to make similar predictions, which pulls their vectors together.

Why does Word2Vec maintain two embedding matrices during training?
?
$W$ for **centre** words and $U$ for **context** words — both $\in \mathbb{R}^{V \times d}$. The softmax dot product $v_c^\top v_w$ uses one row from each. Final embeddings are usually $W$ alone (or $\frac{1}{2}(W+U)$).

What makes embedding similarity "graded" rather than "all-or-nothing"?
?
Vectors live in a **continuous space**, so similarity is measured by **cosine** or **dot product** along a continuum. One-hot vectors are orthogonal — similarity collapses to exact match.

What are five structural failures of bag-of-words / count-based models that motivate embeddings?
?
(1) **Sparsity** — high-dim, mostly zero; (2) **Synonymy** — "car" ⊥ "automobile"; (3) **Polysemy** — "bank" gets one vector; (4) **Order erased** — "dog bites man" = "man bites dog"; (5) **Interaction unmodelled** — "I like" vs "I do not like" treated as additive counts.
