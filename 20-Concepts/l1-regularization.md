---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Loss-functions-regs.pdf
created: 2026-05-03
---

# L1 regularization (Lasso)

Penalty $\Omega(w) = \|w\|_1 = \sum_j |w_j|$ added to a training loss. Its defining property is **sparsity**: the optimum typically has many components $w_j = 0$ exactly, making $L_1$ the standard tool for **feature selection**.

When the underlying loss is squared error ($\sum_i (w^\top x_i - y_i)^2$), the resulting estimator is called **Lasso**:

$$
\min_w \; \frac{1}{n}\sum_i (w^\top x_i - y_i)^2 + \lambda \|w\|_1.
$$

## Why $L_1$ produces sparsity

Picture the **constraint form**: $\min \mathcal{L}(w)$ subject to $\|w\|_1 \le B$. The feasible set $\{w : \sum_j|w_j| \le B\}$ is a **diamond** (in 2D) or **cross-polytope** (higher-D) — a polytope whose **vertices lie on the coordinate axes**.

Loss contours (level sets of $\mathcal{L}$) typically intersect the feasible set first at a **vertex**, because the vertices stick out and the loss decreases smoothly toward its unconstrained minimum. At a vertex, all but one coordinate are zero. Hence $L_1$ optima are sparse.

By contrast, the $L_2$ ball is smooth — no corners — so contours touch its boundary at generic non-axis points, and the optimum has all components small but nonzero.

## Convexity properties

- **Convex**: yes. $\|w\|_1$ is a sum of $|w_j|$, each of which is convex; the sum of convex functions is convex.
- **Strictly convex**: **no**. The Hessian is zero almost everywhere (the function is piecewise linear). Two correlated or duplicated features can share weight along a flat ridge of the optimum — non-unique solutions.
- **Differentiable**: **no** at any $w_j = 0$. The kink is what creates the sparsifying behavior; standard gradient descent struggles with the discontinuity, so practical solvers use **subgradients**, **proximal gradient methods** (ISTA), or **coordinate descent**.

## When to use it

- You suspect only a small subset of features matters and want the model to *find* them automatically.
- Interpretability: a sparse $w$ is easier for a human to read.
- Compressing models for inference (zero weights can be skipped).

When features are highly correlated, **Lasso arbitrarily picks one** of the correlated set and zeroes the others, which can be unstable across resamples. **Elastic Net** (= Lasso + a touch of Ridge) fixes this by restoring strict convexity.

## $L_1$ as a regression loss vs. as a regularizer

Be careful with terminology — $L_1$ shows up in two completely different roles:

| Role | Object | Meaning |
| --- | --- | --- |
| **Loss** | $\sum_i \|y_i - \hat{y}_i\|$ | regression loss; estimates the **median** label |
| **Regularizer** | $\lambda\|w\|_1$ | penalty on the parameters; induces **sparsity** of $w$ |

This concept note is about the regularizer role. The loss role is covered alongside MSE / MAE — same norm, different vector being normed (residuals vs. parameters).

## Exam-relevant facts

- $\Omega(w) = \|w\|_1 = \sum_j|w_j|$.
- Induces **sparsity** — many $w_j = 0$ exactly.
- **Convex but NOT strictly convex** — non-unique solutions when features are correlated.
- **Not differentiable** at $w_j = 0$ — subgradient methods needed.
- Lasso = squared loss + $L_1$ regularizer.
- Trap (§1e): "L2 → sparse" is **false**; sparsity is an $L_1$ property only.

## Related

- [[regularization]] — the umbrella concept.
- [[l2-regularization]] — the shrink-don't-zero alternative.
- [[elastic-net]] — combines $L_1$ + $L_2$ to recover strict convexity.
- [[overfitting-underfitting]] — what regularizers prevent.
- [[lecture-10-loss-functions-regularization|SLP L10]] — source.
