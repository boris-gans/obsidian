---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Loss-functions-regs.pdf
created: 2026-05-03
---

# Huber loss

A regression loss that behaves like **squared loss near zero** and **absolute loss far away**, sharing the strengths of both. Parameterized by a threshold $\delta > 0$:

$$
\ell_{\text{Huber},\delta}(r) = \begin{cases}
\tfrac{1}{2} r^2 & \text{if } |r| < \delta, \\
\delta\bigl(|r| - \delta/2\bigr) & \text{if } |r| \ge \delta,
\end{cases}
\qquad r = h(x_i) - y_i.
$$

The lecture also calls it the **Smooth Absolute Loss** — *"best of both worlds"*: the smooth quadratic shape near the origin gives a well-behaved gradient signal for small residuals, and the linear tail bounds the contribution of outliers.

## Why "best of both worlds"

| Loss | Near $r = 0$ | Far from $r = 0$ | Differentiable? | Outlier-sensitive? |
| --- | --- | --- | --- | --- |
| **Squared** ($r^2$) | quadratic | quadratic — explodes | yes everywhere | **very** — one outlier dominates |
| **Absolute** ($\|r\|$) | linear with kink at 0 | linear | **no** at $r=0$ | low — bounded gradient |
| **Huber** | quadratic, smooth | linear, bounded slope | yes everywhere | low — bounded slope = bounded influence |

So Huber gets:

- The smooth optimization landscape of squared loss for small residuals (no kink at 0).
- The bounded influence of absolute loss for large residuals (one outlier with $|r| = 100$ contributes $\delta(100 - \delta/2)$, not $10\,000$).

## Continuity at the join

The two pieces are designed to match value AND first derivative at $|r| = \delta$:

- Quadratic side at $r = \delta$: $\tfrac{1}{2}\delta^2$. Linear side at $r = \delta$: $\delta(\delta - \delta/2) = \tfrac{1}{2}\delta^2$. ✓
- Quadratic derivative at $r = \delta$: $r = \delta$. Linear derivative at $r = \delta^+$: $\delta$. ✓

So Huber is **$C^1$** (continuously differentiable) but **not $C^2$** — the second derivative jumps from $1$ to $0$ at $|r| = \delta$. The slide deck mentions **Log-Cosh loss**, $\log(\cosh(r))$, as a similar shape that is **twice differentiable everywhere**.

## Choosing $\delta$

$\delta$ is a hyperparameter that says *how big a residual counts as "outlier-sized."*

- Small $\delta$ → most points fall in the linear region → behaves like absolute loss → robust but no quadratic smoothness.
- Large $\delta$ → most points fall in the quadratic region → behaves like squared loss → smooth but outlier-sensitive again.
- Common heuristic: $\delta$ = empirical scale of the residuals (e.g., median absolute deviation).

## When to use it

- Regression on data with **heavy-tailed noise** or outliers (e.g., predicting income, reaction time).
- When you want the smooth gradient signal of squared loss but can't afford its outlier sensitivity.

## Exam-relevant facts

- Quadratic for $|r| < \delta$, linear for $|r| \ge \delta$.
- Continuously differentiable (matches value and slope at $|r| = \delta$).
- Combines smoothness (near 0) with outlier robustness (far from 0).
- Log-Cosh is a **smoother** alternative — twice differentiable everywhere.

## Related

- [[mean-squared-error]] — the squared-loss alternative (smooth but outlier-sensitive).
- [[regularization]] — Huber as loss is orthogonal to the regularizer choice.
- [[lecture-10-loss-functions-regularization|SLP L10]] — source.
