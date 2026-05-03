---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Loss-functions-regs.pdf
created: 2026-05-03
---

# Regularization

A **penalty term added to the training objective that breaks ties between data-equivalent solutions** by pushing $w$ toward a particular kind of vector — small, sparse, smooth, etc. The general form:

$$
\min_w \; \frac{1}{N}\sum_{i=1}^{N} \ell\big(h_w(x_i), y_i\big) + \lambda \, \Omega(w).
$$

**Loss** says how far each prediction is from a "good" answer. **$\Omega(w)$** says, when multiple $w$ fit the data equally well, *which kind* of $w$ to prefer. **$\lambda$** is the dial that trades the two off.

## Why we need it

Without regularization, an over-parameterized model can drive training error to zero by memorizing noise — high variance, poor generalization. Regularization restricts the *effective* hypothesis space without changing the architecture, by penalizing the parameter vectors that the architecture *could* produce but we don't want it to.

The SVM (L09) was the **first concrete regularizer** the SLP course showed: the soft-margin penalty $C\sum_i \xi_i$ is exactly the data-fit half of a regularized objective, and $\tfrac{1}{2}\|w\|^2$ is the $L_2$-regularization half — written backwards because the SVM tradition flips $\lambda$ to its inverse $1/C$.

## The penalty ↔ constraint duality

For every $\lambda \ge 0$ there exists a $B \ge 0$ such that

$$
\underbrace{\min_w \; \mathcal{L}(w) + \lambda\, \Omega(w)}_{\text{penalty form}} \quad \Longleftrightarrow \quad \underbrace{\min_w \; \mathcal{L}(w) \;\text{s.t.}\; \Omega(w) \le B}_{\text{constraint form}}.
$$

The relationship is **inverse**: large $\lambda$ ↔ small $B$ (tight constraint). Setting $B$ very large makes the constraint non-binding — equivalent to $\lambda$ near zero. The constraint form is the easy way to picture the geometry: feasible $w$'s lie inside a ball whose shape depends on the regularizer.

## Common regularizers

| $\Omega(w)$ | Name | Effect | Convexity |
| --- | --- | --- | --- |
| $\|w\|_2^2 = \sum_j w_j^2$ | $L_2$ / weight decay / Ridge | shrinks all coefficients toward zero | strictly convex |
| $\|w\|_1 = \sum_j |w_j|$ | $L_1$ / Lasso | **sparsifies** — many $w_j$ exactly zero | convex but **not** strictly |
| $\alpha\|w\|_1 + (1-\alpha)\|w\|_2^2$ | Elastic Net | both: shrinkage + sparsity | strictly convex when $\alpha < 1$ |

Implicit regularizers don't add a penalty term but achieve the same effect by limiting some other resource:

- **[[early-stopping]]** — caps the iteration count $M$, which empirically behaves like $L_2$ on convex losses.
- **[[dropout]]** — randomly zeros activations, equivalent to weight averaging.

## Selecting $\lambda$

Train and validation error both depend on $\lambda$:

- $\lambda$ very small → regularizer disabled → train error low, **validation error high** (overfitting).
- $\lambda$ at sweet spot → both errors low and close together.
- $\lambda$ very large → regularizer dominates → both errors high (underfitting).

Validation error is **U-shaped** in $\lambda$ and is the curve cross-validation minimizes. Search $\lambda$ on a **log scale** (e.g., $10^{-4}, 10^{-3}, \ldots, 10^2$) — linear sweeps waste compute.

## Why regularization helps generalization

- **Bias–variance lens** (covered in L11): a regularizer trades a small increase in bias for a large decrease in variance.
- **Bayesian lens** (out of scope in SLP): adding $\lambda \Omega(w)$ is equivalent to MAP estimation under a prior $p(w) \propto \exp(-\lambda \Omega(w))$ — Gaussian prior ↔ $L_2$, Laplace prior ↔ $L_1$.
- **Optimization lens**: adding $\lambda \|w\|_2^2$ makes the loss surface strictly convex (or "more convex"), so optimization becomes well-conditioned.

## Exam-relevant facts

- General form: $\min_w \mathcal{L}(w) + \lambda \Omega(w)$.
- Penalty and constraint forms are equivalent (inverse $\lambda \leftrightarrow B$ relationship).
- $L_1$ → sparsity; $L_2$ → shrinkage (no exact zeros). The §1e trap is "L2 → sparse" — **false**.
- Train error monotonically rises with $\lambda$; validation error is U-shaped; cross-validation picks the sweet spot.
- Implicit regularizers (early stopping, dropout) achieve the same goal without an explicit penalty term.

## Related

- [[l1-regularization]] / [[l2-regularization]] / [[elastic-net]] — the canonical penalty choices.
- [[early-stopping]] — implicit regularizer via iteration count.
- [[dropout]] — implicit regularizer for neural nets.
- [[overfitting-underfitting]] — the failure modes regularization mitigates.
- [[support-vector-machine]] — SVM's $C$ is the inverse of $\lambda$ on the same kind of objective.
- [[lecture-10-loss-functions-regularization|SLP L10]] — source.
