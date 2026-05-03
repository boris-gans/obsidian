---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 10 - Logistic Regression-1.pdf
created: 2026-05-02
---

# Maximum entropy classifier

A probabilistic model that **makes the weakest possible assumptions while remaining consistent with observed data**. The principle: among all distributions that satisfy a set of expected-value constraints derived from data, choose the one with **maximum entropy** — the least committed.

> Quiz II.M3 Q1: "A maximum-entropy text classifier selects the distribution that **is least committed given observed feature constraints**."

## The principle

Given:

- **Normalization**: $\sum_y p(y|x) = 1$ for every $x$
- **Linear feature-expectation constraints**: for each feature function $f_i(x, y)$,
$$\mathbb{E}_p[f_i] = \mathbb{E}_{\text{data}}[f_i]$$
the model's expected feature value matches the empirical expectation from training data

We seek $p^*(y|x)$ that maximizes entropy
$$H(p) = -\sum_y p(y|x) \log p(y|x)$$
subject to those constraints.

## Why entropy?

Entropy is the canonical measure of **uncertainty**. Maximum entropy = "I know nothing beyond what the data forces me to know". Any other choice would inject extra assumptions that the data didn't justify — see also [[generative-vs-discriminative|discriminative]] models more broadly.

## The solution: exponential family

Solving the constrained optimization (Lagrange multipliers) yields a Boltzmann / softmax form:
$$p(y|x) = \frac{1}{Z(x)} \exp\!\left(\sum_i \lambda_i f_i(x, y)\right)$$

| Symbol | Meaning |
|---|---|
| $\lambda_i$ | Lagrange multiplier — becomes the **learned weight** for feature $i$ |
| $f_i(x, y)$ | Feature function — encodes a known data property |
| $Z(x) = \sum_y \exp(\sum_i \lambda_i f_i(x, y))$ | Normalization (partition function) |

> Quiz II Q5: "Why does entropy maximization under linear feature constraints lead to **exponential models** in NLP classifiers?" → It **follows from the entropy-maximization solution** via Lagrange multipliers.

## Connection to logistic regression

For binary labels $y \in \{0, 1\}$ with a common feature-function choice ($f_i = y x_i$, $f_0 = y$), the maxent solution reduces to:
$$p(y=1|x) = \frac{1}{1 + e^{-(\mathbf{w}\!\cdot\!\mathbf{x} + b)}} = \sigma(\mathbf{w}\!\cdot\!\mathbf{x} + b)$$

The Lagrange multipliers $\lambda_i$ are exactly the LR weights $w_i$, and the bias $b$ is the multiplier for the normalization constraint.

> **Logistic regression IS a maximum-entropy classifier** — same model, two derivations.

## Coefficient meaning

Increasing $|\lambda_i|$ (= $|w_i|$) **strengthens the influence of the word feature** on the score and log-odds (Quiz II Q16). The Lagrange-multiplier perspective explains why: a larger multiplier enforces the constraint more strongly.
