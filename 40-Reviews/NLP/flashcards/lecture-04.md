---
tags: [flashcards, NLP]
---

# Lecture 04 — Vector Space Models (flashcards)

What is a one-hot encoding?
?
A vector with a single 1 at the word's position in the vocabulary and 0s elsewhere — each word is a coordinate axis in $\mathbb{R}^V$.

Why don't one-hot vectors capture semantic similarity?
?
Because they are **orthogonal** — every pair of distinct one-hot vectors has cosine 0. The representation carries no information about similarity or meaning.

What is the bag-of-words assumption?
?
**Word order is irrelevant** — only word identity and frequency are retained. Grammar and syntax are discarded.

Under BoW, what happens to two sentences with the same words but different order?
?
They are **treated as identical** (e.g. "the kid ate the cookie" and "the cookie ate the kid" both → `[2,1,1,1]`).

Which statement is FALSE about Bag-of-Words representations?
?
"They capture semantic similarity between words." BoW does not capture semantic similarity.

What advantage do sequence models (RNNs, Transformers) have over Bag-of-Words?
?
They **capture word order and context** — BoW discards both.

What does the term-document matrix $X_{ij}$ contain?
?
The number of times word $i$ appears in document $j$. Rows = words, columns = documents.

State the distributional hypothesis.
?
**Words that occur in similar contexts tend to have similar meanings** — meaning is inferred from context, not defined explicitly.

What does cosine similarity measure?
?
**The angle between two vectors in vector space** (focuses on direction rather than magnitude).

Why does cosine similarity normalize TF-IDF document vectors?
?
To **remove document-length effects** — short and long documents on the same topic should still be similar.

Compute $\cos(A, B)$ for $A = (2,1)$ and $B = (1,3)$.
?
$\cos(A,B) = \dfrac{2+3}{\sqrt{5}\sqrt{10}} = \dfrac{5}{\sqrt{50}} \approx 0.71$.

Write the TF-IDF formula on the formula sheet.
?
$w_{i,j} = \mathrm{TF}_{i,j} \cdot \log\!\left(\dfrac{N}{n_i}\right)$, where $N$ = number of documents and $n_i$ = number of documents containing word $i$.

What does TF-IDF do conceptually?
?
**Reweights terms** to reduce the influence of common words and **highlight discriminative ones** — assigns higher weight to rare but informative terms.

Why does IDF down-weight very common words?
?
Because they **carry little discriminative information** — they appear in almost every document, so they cannot help distinguish between them.

Compute IDF and TF-IDF of "fun" in document $D_3$ = "learning nlp is fun" given the corpus [D1, D2, D3] from the mock exam.
?
$\mathrm{TF}(\text{fun}, D_3) = 1$, $\mathrm{DF}(\text{fun}) = 2$ (it's in $D_1$ and $D_3$), $\mathrm{IDF}(\text{fun}) = \log(3/2) \approx 0.405$, so $\mathrm{TF\text{-}IDF}(\text{fun}, D_3) = \log(3/2) \approx 0.405$.

What is the main idea of Latent Semantic Analysis?
?
**Use matrix factorization (SVD) to reduce dimensionality** of the term-document matrix into a latent semantic space where related words / documents align.

What technique does LSA use to factorize the term-document matrix?
?
**Singular Value Decomposition (SVD)**: $X = U\Sigma V^T$, then keep the top $k$ singular values.

What do the diagonal entries of $\Sigma$ in LSA correspond to?
?
**Singular values describing variance** — the importance of each latent dimension.

What problem does LSA solve that TF-IDF doesn't?
?
**Vocabulary mismatch** — two documents about the same topic using different vocabularies (e.g. *car* / *automobile*) appear unrelated under TF-IDF but become similar under LSA's latent space.

How does HAL differ from LSA?
?
HAL captures **local context** through **co-occurrence within a sliding window**; LSA captures **global** patterns across entire documents via SVD.

Which view of meaning emerges in vector space models?
?
**Geometric**: similarity is expressed through distance or angle between vectors, semantic relations become **graded, continuous, and quantitative**.

State the four common similarity coefficients.
?
**Dice** $\frac{2|X \cap Y|}{|X|+|Y|}$, **Jaccard** $\frac{|X \cap Y|}{|X \cup Y|}$, **Overlap** $\frac{|X \cap Y|}{\min(|X|,|Y|)}$, **Cosine** $\frac{X \cdot Y}{\|X\|\|Y\|}$.
