---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Bias-variance(1).pdf
created: 2026-05-03
---

# Expected predictor $\bar{h}$

The **average prediction** an algorithm $A$ produces at a point $x$, taken over all possible training sets $D$ drawn from $P$:

$$
\bar{h}(x) = \mathbb{E}_{D \sim P^n}\big[h_D(x)\big], \qquad h_D = A(D).
$$

It exists because the training set $D$ is itself a random variable — drawn i.i.d. from the data distribution $P(X, Y)$. Since $h_D = A(D)$ is a function of $D$, it inherits randomness; $\bar{h}$ is the deterministic average.

## Why we need it

The expected predictor is the **center** that the [[bias-variance-decomposition]] uses to split test error into reducible and irreducible parts:

- **Variance** measures how individual $h_D$'s scatter around $\bar{h}$.
- **Bias** measures how far $\bar{h}$ is from the true expected label $\bar{y}(x) = \mathbb{E}[y \mid x]$.

Without a notion of "the algorithm's average behavior," we couldn't separate "the model class is wrong" (bias) from "this particular training set was unrepresentative" (variance).

## How to estimate it in practice

In practice you don't have access to $P$ — only one finite dataset. The **bootstrap** approximation:

1. Sample $m$ datasets $D_1, \ldots, D_m$ from your training data with replacement.
2. Train $m$ predictors $h_{D_1}, \ldots, h_{D_m}$, one per resample.
3. Estimate $\bar{h}(x) \approx \frac{1}{m}\sum_{j=1}^m h_{D_j}(x)$.

This is exactly the recipe **bagging** (L12) uses — and the reason bagging reduces variance is that the averaged predictor approximates $\bar{h}$, which by definition has zero variance term (no scatter around itself).

## Existence depends on the loss

For squared loss, $\bar{h}(x) = \mathbb{E}_D[h_D(x)]$ is well-defined as a real-valued function. For classification with majority-vote aggregation, the corresponding object is the *most-likely-predicted* label — not literally the mean. The general bias-variance story still holds qualitatively, but the algebra differs.

## Exam-relevant facts

- $\bar{h}(x) = \mathbb{E}_D[h_D(x)]$ — average prediction over training-set randomness.
- $D$ is a random variable; therefore so is $h_D$; the "expected" in $\bar{h}$ is over $D$.
- Estimable in practice by averaging predictions from many bootstrap-trained models.
- The center used by the bias-variance decomposition.
- $\bar{h} \ne \bar{y}$ in general — the difference between them squared is the **bias²**.

## Related

- [[bias-variance-decomposition]] — uses $\bar{h}$ as the splitting point.
- [[bagging]] — empirically approximates $\bar{h}$ to reduce variance.
- [[generalization-error]] — what we decompose.
- [[lecture-11-bias-variance|SLP L11]] — source.
