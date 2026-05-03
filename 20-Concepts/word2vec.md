---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 13 - Word Embeddings.pdf
  - course: NLP
    file: 09_Word2Vec_Embedding.ipynb
created: 2026-05-02
---

# Word2Vec

**Word2Vec** (Mikolov et al., 2013) is an **unsupervised** technique that learns dense [[word-embeddings]] through a **local prediction task**: rather than decomposing a global co-occurrence matrix (as [[latent-semantic-analysis|LSA]] does), Word2Vec trains a shallow neural model to **predict words from their contexts**, and the representation emerges from the prediction loss ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=12|slide 12]]).

It's a **shallow neural network with one hidden layer** — the hidden layer *is* the embedding. The "deep" part is the corpus, not the architecture.

## Two symmetric formulations ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=12|slide 12]])

| Formulation | Direction | Predict |
|---|---|---|
| **CBOW** (Continuous Bag-of-Words) | context → centre | the centre word given surrounding context |
| **Skip-gram** | centre → context | the context words given the centre word |

Both share the same [[embedding-matrix]] $W \in \mathbb{R}^{V \times d}$. See [[skip-gram-and-cbow]] for the detailed mechanics, training tradeoffs, and exam framing.

## The Skip-gram conditional ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=14|slide 14]])

Probability of context word $c$ given centre word $w$, defined via softmax over inner products:
$$P(c \mid w) = \frac{\exp(v_c^\top v_w)}{\sum_{w' \in V} \exp(v_{w'}^\top v_w)}$$

**Similarity is encoded through inner products.** Words whose vectors have large dot products are more likely to co-occur. *Meaning becomes geometry* ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=14|slide 14]]).

## The objective function ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=15|slide 15]])

Over a corpus $(w_1, \ldots, w_T)$ with context window $\mathcal{C}(w_t)$:
$$\max \prod_{t=1}^{T} \prod_{c \in \mathcal{C}(w_t)} P(c \mid w_t)$$
or equivalently in log form:
$$\sum_{t=1}^{T} \sum_{c \in \mathcal{C}(w_t)} \log P(c \mid w_t)$$

Optimization: **stochastic gradient descent** on the embedding parameters.

## Why the predictive turn matters

Unlike LSA — where structure emerges from **matrix decomposition** of a precomputed term-document matrix — Word2Vec's structure emerges from **optimizing a predictive criterion** under the assumption of **independence of context windows** ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=15|slide 15]]). Each word–context pair contributes a small gradient step that incrementally adjusts vectors, so meaning emerges from **repeated prediction and correction** ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=19|slide 19]]).

## Connection to PMI matrix factorization ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=17|slide 17]])

A surprising result: although Word2Vec is predictive, the Skip-gram-with-negative-sampling model **implicitly factorizes a shifted [[pointwise-mutual-information|PMI]] matrix**. Under regularity assumptions (large corpus, large $d$, negative sampling matching the unigram distribution), the learned dot product satisfies:
$$v_w^\top v_c \approx \mathrm{PMI}(w, c) - \log k$$
where $\mathrm{PMI}(w, c) = \log \frac{P(w,c)}{P(w)P(c)}$ and $k$ is the number of negative samples.

So predictive embeddings and count-based methods are not unrelated — they differ in *how* the factorization is achieved: explicitly via SVD (LSA), implicitly via gradient-based optimization (Word2Vec). The predictive approach **scales better** and produces more flexible representations.

## What Word2Vec captures ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=18|slides 18–19]])

- **Semantic similarity** — words appearing in similar contexts get similar vectors
- **Linear analogies** — relations encode as approximately linear translations: $v_{king} - v_{man} + v_{woman} \approx v_{queen}$
- **Distributed meaning** — no axis is interpretable; meaning lives across all $d$ coordinates

## What Word2Vec misses

- **Polysemy** — "bank" gets a single static vector, a weighted average of its senses ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=19|slide 19]])
- **Context dependence** — representations are fixed after training; they don't adapt to the sentence at inference time

This is the conceptual gap closed by **transformers** (Session 19): contextual embeddings depend dynamically on surrounding tokens.

## Exam framing

| Question | Answer |
|---|---|
| What is Word2Vec? | A shallow neural model that learns dense word embeddings by predicting context (mock Q6) |
| What's the network depth? | **One hidden layer** — the hidden layer is the embedding |
| How are similar contexts → similar vectors enforced? | Two words appearing in similar contexts will be **trained to make similar predictions**, which pulls their vectors together (Quiz III Q9, Q9.B) |
| Why is Word2Vec called "predictive"? | Vectors are **learned by minimizing a prediction loss** over word–context pairs, not derived from precomputed counts (Quiz III Q7) |
| Where does the term "shallow neural network" come from? | One hidden layer between input one-hot and output softmax |

## Related concepts

- [[skip-gram-and-cbow]] — the two formulations and when each is preferred
- [[negative-sampling]] — the efficient training objective used in practice
- [[embedding-matrix]] — the $V \times d$ parameter table
- [[distributional-hypothesis]] — the foundational claim Word2Vec operationalizes
- [[latent-semantic-analysis|LSA]] — the count-based predecessor

## Canonical gensim skeleton

In practice, Word2Vec is trained via gensim — the standard library:

```python
from gensim.models import Word2Vec

# tokens: list of tokenized sentences (list of lists of words)
model = Word2Vec(
    sentences=tokens,
    vector_size=100,   # d = embedding dimension
    window=5,          # context window size
    min_count=2,       # ignore words with fewer than 2 occurrences
    sg=1,              # 1 = Skip-gram; 0 = CBOW
    negative=5,        # k = number of negative samples
    epochs=10,
)

# Lookup
v_king = model.wv["king"]                                  # shape (100,)
sim = model.wv.similarity("king", "queen")                 # cosine similarity
nn = model.wv.most_similar("king", topn=5)                 # nearest neighbours

# Analogy: king − man + woman ≈ queen
result = model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
```

Reference: `[Word2Vec via gensim (cells 6–11)](30-Sources/NLP/notebooks/09_Word2Vec_Embedding.ipynb)`.

The exam-critical kwargs:
- `vector_size` = $d$ (embedding dimension) — **embedding matrix is $V \times d$**
- `sg=1` for Skip-gram, `sg=0` for CBOW
- `negative=5` enables negative sampling with $k=5$
- `model.wv` is the trained word-vector table; everything else is training infrastructure
