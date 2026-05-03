---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 10 - Logistic Regression-1.pdf
created: 2026-05-02
---

# Log-odds

In [[logistic-regression]], **what is linear in the input features is the log-odds**, *not* the probability itself. This is the single most important interpretive fact about LR (Quiz II Q11).

## Definition

For binary classifier with $p = p(y=1|x)$, the **odds** of the positive class are
$$\text{odds}(y=1|x) = \frac{p}{1-p}$$
and the **log-odds (logit)** are
$$\log \frac{p}{1-p} = \mathbf{w}\!\cdot\!\mathbf{x} + b$$

This linearity is what makes LR a **linear** classifier — but linear in **logit space**, not in probability space.

## Coefficient interpretation through log-odds

Each coefficient adds to the log-odds **additively** when its feature is active. Equivalently, each coefficient multiplies the **odds** by $e^{w_i}$ when the word is present.

| Coefficient | Effect on log-odds | Effect on odds |
|---|---|---|
| $w_i = +1.5$ | adds 1.5 | multiplies odds by $e^{1.5} \approx 4.48$ (Quiz II Q15) |
| $w_i = -2$ | subtracts 2 | multiplies odds by $e^{-2} \approx 0.135$ (Quiz II.M3 Q5) |
| $w_i = 0$ | no effect | odds unchanged (word is irrelevant) |

## Why log-odds (not probability)

- **Probability** is bounded in $[0, 1]$ and the relationship between feature changes and probability changes is **non-linear** (saturating)
- **Log-odds** is unbounded $(-\infty, +\infty)$ and additively decomposable, making it interpretable as "evidence" contributed by each feature
- The non-linearity in LR lives in the **link function** ([[sigmoid]]), which maps linear log-odds to probability

## Compact summary

| Quantity | Linear in features? |
|---|---|
| $p(y=1|x)$ | **No** (sigmoid is non-linear) |
| $\log p(y=1|x)$ | No |
| Prediction entropy | No |
| **Log-odds** $\log \frac{p}{1-p}$ | **Yes** (Quiz II Q11) |
