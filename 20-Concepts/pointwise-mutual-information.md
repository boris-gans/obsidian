---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 13 - Word Embeddings.pdf
created: 2026-05-03
---

# Pointwise Mutual Information (PMI)

**PMI** measures how much more often two events co-occur than independence would predict. For a (word, context) pair:

$$\mathrm{PMI}(w, c) = \log \frac{P(w, c)}{P(w) \, P(c)}$$

If $w$ and $c$ co-occur exactly as often as independence predicts, $\mathrm{PMI} = 0$. If they co-occur more than expected (associated), $\mathrm{PMI} > 0$. If less than expected (e.g. "the" rarely appears next to itself), $\mathrm{PMI} < 0$.

PMI is **not** directly tested on the exam (the formula sheet doesn't carry it), but it's the mathematical structure that ties [[latent-semantic-analysis|LSA]] and [[word2vec|Word2Vec]] together — without it the "two methods, one factorization" story doesn't make sense.

## Intuition

PMI is the **log of the lift** that observing $w$ gives you over the prior on $c$:

- $\mathrm{PMI}(\text{strong}, \text{coffee}) > 0$ — "coffee" is much more likely after "strong" than overall
- $\mathrm{PMI}(\text{strong}, \text{politics}) \approx 0$ — no special association
- $\mathrm{PMI}(\text{the}, \text{the}) < 0$ — "the" rarely follows itself, so seeing "the" reduces the probability of another "the" immediately after

It's the same shape as **log-odds in [[logistic-regression]]** ($\log p / (1-p)$): both express *associations on a log scale*, where addition corresponds to multiplicative effects.

## Positive PMI (PPMI)

Negative PMI values are unstable for rare pairs (a single co-occurrence event can swing the estimate). The standard fix is **Positive PMI**:

$$\mathrm{PPMI}(w, c) = \max(\mathrm{PMI}(w, c), \, 0)$$

Clip negatives to zero. PPMI matrices are the canonical input to count-based word embeddings (typically followed by SVD for dimensionality reduction — i.e. PPMI + SVD is essentially what classical "vector-space semantics" was doing).

## The shifted-PMI / Word2Vec connection

Levy & Goldberg (2014) showed that **Word2Vec's skip-gram with negative sampling (SGNS) implicitly factorizes a shifted PMI matrix**: under regularity assumptions (large corpus, large embedding dim $d$, negative sampling matched to the unigram distribution), the learned dot product satisfies

$$v_w^\top v_c \approx \mathrm{PMI}(w, c) - \log k$$

where $k$ is the number of negative samples. The "$-\log k$" shift comes from the negative-sampling objective.

This result is *the* bridge between count-based and predict-based embeddings: they're not rival paradigms, they're **two algorithmic paths to the same matrix factorization** — explicit linear algebra (LSA on PPMI) vs. implicit gradient-based optimization (SGNS).

## Where this matters in the course

- The "implicitly factorizes a shifted PMI matrix" line in [[word2vec]] is this concept.
- It's a low-priority *connecting* concept — exam doesn't test it directly, but understanding it removes the LSA-vs-Word2Vec confusion that the cluster-5 arc warns against.

## Related

- [[latent-semantic-analysis]] — LSA-on-PPMI is the count-based embedding canon
- [[word2vec]] — implicitly factorizes a shifted PMI matrix
- [[negative-sampling]] — the $-\log k$ shift comes from this objective
- [[log-odds]] — same "associations on a log scale" template
- [[distributional-hypothesis]] — PMI is the formal way to measure "appears in similar contexts"
