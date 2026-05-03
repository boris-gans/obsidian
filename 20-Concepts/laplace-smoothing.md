---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 09 - Naive Bayes-1.pdf
created: 2026-05-02
---

# Laplace smoothing (add-one smoothing)

A technique to handle **zero-probability** events when estimating probabilities from finite training data — particularly important for [[naive-bayes|Naïve Bayes]] and n-gram language models.

## The problem

In Naïve Bayes, the class score is $P(y) \prod_i P(x_i|y)$. If any single word $x_i$ never appears in training data for class $y$, then $P(x_i|y) = 0$ — and the entire product zeros out, **regardless of how strong the other evidence is**. One unseen word can discard the whole document.

## The fix

Add a small constant (typically 1, hence "add-one") to every count before normalizing:

$$P(x_i | y) = \frac{c(x_i, y) + \alpha}{\sum_j c(x_j, y) + \alpha |V|}$$

where $\alpha > 0$ is the smoothing constant and $|V|$ is the vocabulary size. This guarantees no probability is exactly zero.

## Trade-off

- $\alpha$ small → close to MLE estimates; vulnerable to zeros
- $\alpha$ large → uniform distribution; loses signal from observed frequencies
- $\alpha$ is the **only hyperparameter** of basic Naïve Bayes — tune via cross-validation

## Why this matters

Without smoothing, the model fails on real-world test data the moment any unseen word appears. Smoothing is the **minimum** robustness guarantee. The same idea appears in:

- N-gram language models (where unseen contexts are catastrophic)
- Probability estimates for [[hmm-viterbi|HMM]] transition / emission counts
- Any maximum-likelihood estimate from sparse counts

## Operational note

In code (e.g. scikit-learn's `MultinomialNB(alpha=1.0)`), `alpha` is the smoothing constant. `alpha=0` disables smoothing and reproduces MLE — generally a bad idea on text data because of long-tail vocabulary.
