---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Bias-variance(1).pdf
created: 2026-05-03
---

# Generalization error

The **expected loss on a fresh sample drawn from the data distribution** — the actual quantity supervised learning cares about:

$$
\mathcal{L}_{\text{gen}}(h) = \mathbb{E}_{(x,y) \sim P}\big[\mathcal{L}(h(x), y)\big].
$$

Distinct from the **training error** $\frac{1}{N}\sum_i \mathcal{L}(h(x_i), y_i)$, which is computed on the same dataset the model was fit on. Training is the *proxy*; generalization error is the *goal*. Empirical risk minimization minimizes the proxy and **hopes** that minimizes the goal — when this hope fails, you're overfitting.

## Why training error is not enough

A model with enough capacity can drive training error to zero by memorizing labels — but this tells you nothing about how it will perform on a fresh $(x, y)$. The classic failure modes:

- **k-NN with $k=1$**: training error is exactly zero (each point is its own nearest neighbor) but generalization can be terrible.
- **Decision tree grown to leaves of size 1**: training error zero, generalization noisy.
- **Polynomial regression with degree $\ge N$**: training error zero, predictions wild between training points.

Generalization error is what cross-validation, held-out validation sets, and bias-variance theory are all trying to estimate or control.

## The bias-variance lens

Generalization error decomposes into three terms (under squared loss, see [[bias-variance-decomposition]]):

$$
\mathcal{L}_{\text{gen}} = \text{Variance} + \text{Bias}^2 + \text{Noise}.
$$

This decomposition is what makes generalization error tractable: each term has its own meaning and its own fix.

## Estimating generalization error in practice

You can't compute $\mathbb{E}_{(x,y) \sim P}$ directly — $P$ is unknown. Estimators:

1. **Held-out test set** — a fresh sample from $P$ never seen during training or model selection. Gives a single noisy estimate.
2. **Cross-validation** — average held-out error across $K$ folds. Lower variance than a single holdout.
3. **Bootstrap** — resample with replacement from the training set.
4. **Theoretical bounds** (VC dimension, Rademacher complexity) — give worst-case bounds; not used in this course's calculations but motivate why max-margin classifiers generalize.

## Three properties of validation error vs $N$

[[lecture-11-bias-variance|L11]] states three observations as setup for diagnosing bias/variance from learning curves:

1. **Validation error ≥ training error**. The model is fit to the training set; held-out data is unseen.
2. **Validation error decreases monotonically in $N$**. More data narrows the variance of $h_D$.
3. **Training error never decreases with $N$** — it stays flat or rises. A fixed-capacity model can't perfectly fit progressively more diverse data.

Together they imply: the **gap** between train and val curves shrinks as $N$ grows, and that closing gap is the variance term closing.

## Exam-relevant facts

- $\mathcal{L}_{\text{gen}}(h) = \mathbb{E}_{(x,y) \sim P}[\mathcal{L}(h(x), y)]$ — the goal of supervised learning.
- Decomposes into variance + bias² + noise (under squared loss).
- Training error is a proxy that can be made arbitrarily small without improving generalization.
- Estimated in practice by held-out test set or cross-validation.
- As $N \to \infty$, generalization error → bias² + noise (irreducible floor).

## Related

- [[bias-variance-decomposition]] — the formula that breaks generalization error apart.
- [[expected-predictor]] / [[learning-curve]] — diagnostic tools.
- [[overfitting-underfitting]] — failure modes that generalization error exposes.
- [[cross-validation]] — practical estimator.
- [[lecture-11-bias-variance|SLP L11]] — source.
