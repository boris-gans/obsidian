---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Loss-functions-regs.pdf
created: 2026-05-03
---

# Elastic Net

A regularizer that mixes $L_1$ and $L_2$ penalties:

$$
\Omega(w) = \alpha \|w\|_1 + (1 - \alpha)\|w\|_2^2, \qquad \alpha \in [0, 1).
$$

When paired with squared-error loss, the corresponding estimator is

$$
\min_w \; \frac{1}{n}\sum_i (w^\top x_i - y_i)^2 + \alpha\|w\|_1 + (1 - \alpha)\|w\|_2^2.
$$

The lecture's slide table calls Elastic Net the "have your cake and eat it" choice: **strictly convex** (i.e., unique solution) **and** sparsity-inducing.

## The defining property: strict convexity (the §1f answer)

A function is **strictly convex** when its Hessian is positive definite — every direction has positive curvature, and the minimum is unique.

| Regularizer | Convex? | Strictly convex? |
| --- | --- | --- |
| $\|w\|_1$ alone (Lasso) | yes | **no** — Hessian is zero a.e.; flat ridges with correlated features |
| $\|w\|_2^2$ alone (Ridge) | yes | **yes** — Hessian = $2I$, positive definite |
| Elastic Net ($\alpha < 1$) | yes | **yes** — inherits strict convexity from the $L_2$ component |

The $L_2$ piece contributes positive curvature in every direction. Adding the (weakly) convex $L_1$ piece can't undo that, so the combined objective is strictly convex whenever $\alpha < 1$. **Hence: Elastic Net has a unique optimum even when features are correlated**, while pure Lasso does not.

## Why this matters: the correlated-features problem

If two features $x_a$ and $x_b$ are identical or highly correlated, pure Lasso has no preference between

- $w_a = c, w_b = 0$,
- $w_a = 0, w_b = c$,
- or any convex combination.

The optimum is a flat ridge, so Lasso arbitrarily picks one of the correlated features and zeros the rest — unstable across resamples. Elastic Net **spreads weight smoothly** across correlated features (the $L_2$ piece breaks the tie) while still inducing overall sparsity (the $L_1$ piece zeros out features unrelated to the target).

## The trade-off knob $\alpha$

| $\alpha$ | Behavior |
| --- | --- |
| $0$ | pure Ridge — strictly convex, no sparsity |
| small (e.g., $0.1$) | mostly Ridge with a sparsifying nudge |
| $\sim 0.5$ | balanced |
| close to $1$ | mostly Lasso with $L_2$ stabilization |
| $1$ | excluded (the slide gives $\alpha \in [0, 1)$) — strict convexity is lost |

In sklearn's API, the parameter is called `l1_ratio` and matches this $\alpha$.

## Disadvantage: still non-differentiable

Like Lasso, Elastic Net inherits non-differentiability at $w_j = 0$ from the $L_1$ component. The slide flags this as the disadvantage. Practical solvers use coordinate descent or proximal-gradient methods.

## Exam-relevant facts (§1f)

- $\Omega(w) = \alpha \|w\|_1 + (1-\alpha)\|w\|_2^2$ with $\alpha \in [0, 1)$.
- **Strictly convex** when $\alpha < 1$ — unique solution **even with correlated features**.
- $L_1$ alone is convex but **not strictly** convex; adding any $L_2$ component restores strict convexity.
- Sparsity-inducing (from the $L_1$ part) AND stable with correlated features (from the $L_2$ part).
- Disadvantage: not differentiable at $w_j = 0$.

## Related

- [[regularization]] — umbrella concept.
- [[l1-regularization]] — the sparsity component.
- [[l2-regularization]] — the strict-convexity component.
- [[lecture-10-loss-functions-regularization|SLP L10]] — source.
