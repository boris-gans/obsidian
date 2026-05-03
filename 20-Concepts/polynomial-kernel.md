---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Kernels(1).pdf
created: 2026-05-03
---

# Polynomial kernel

A [[kernel-trick|kernel function]] of the form
$$
K(x, z) = (x^\top z + c)^p
$$
where $c \ge 0$ is an offset (often 1) and $p \in \mathbb{N}$ is the degree. The corresponding feature map $\phi$ embeds the $d$-dim input space into the space of all **monomials of degree $\le p$** — dimension $\binom{d+p}{p}$.

## The quadratic kernel (mock §6)

For $p = 2$ and $c = 1$:
$$
K(x, z) = (x^\top z + 1)^2.
$$

The associated feature map $\phi(x)$ contains all monomials of $x_1, \ldots, x_d$ up to total degree 2, plus a constant and the linear terms:
$$
\phi(x) = (1,\; \sqrt{2}\, x_1,\; \ldots,\; \sqrt{2}\, x_d,\; x_1^2,\; \ldots,\; x_d^2,\; \sqrt{2}\, x_1 x_2,\; \ldots,\; \sqrt{2}\, x_{d-1} x_d).
$$

(The $\sqrt{2}$ factors arise from the binomial expansion's cross terms.) Dimension: $\binom{d+2}{2} = (d+1)(d+2)/2$.

**Verification by expansion** for $d = 2$:
$$
(x_1 z_1 + x_2 z_2 + 1)^2 = (x_1 z_1)^2 + (x_2 z_2)^2 + 1 + 2 x_1 z_1 x_2 z_2 + 2 x_1 z_1 + 2 x_2 z_2,
$$
which is the inner product of $\phi(x) = (1, \sqrt{2} x_1, \sqrt{2} x_2, x_1^2, x_2^2, \sqrt{2} x_1 x_2)$ with $\phi(z)$ — same dimension, same components.

## Why polynomial kernels matter

- **Computable in $O(d)$**: just one inner product, one addition, one power. The implicit feature space takes $O(d^p)$ to write down.
- **Captures interactions**: degree-2 polynomial kernel handles all pairwise feature interactions; degree-3 captures triples; etc.
- **Tunable non-linearity**: degree $p$ controls how complex the decision boundary in original space can be. $p = 1$ recovers the linear kernel; $p = 10$ allows extremely wiggly boundaries.

## Practical usage

- **Quadratic / cubic** common for SVM classification when feature interactions are suspected.
- **Higher degrees** can overfit — boundary becomes very flexible.
- **Offset $c$** controls the bias toward higher-order vs. lower-order terms. $c = 0$ means only degree-exactly-$p$ monomials (no lower-order terms); $c > 0$ includes them.
- Generally **less popular than RBF** in modern usage — RBF is more flexible (infinite-dim) and has only one bandwidth hyperparameter to tune.

## Mock §6 specifics

The exam restates the slack primal inline:
$$
\arg\min_{w, b, \xi} \tfrac{1}{2}\|w\|^2 + C \sum_i \xi_i \quad \text{s.t.}\quad y_i(\langle w, \phi(x_i)\rangle + b) \ge 1 - \xi_i,\ \xi_i \ge 0,
$$
with $\phi$ defined so $\phi(x) \cdot \phi(x') = (x \cdot x' + 1)^2$.

You're then asked to:
- Sketch the boundary for **very large $C$** (low slack tolerance — boundary nails every training point in $\phi$-space, possibly very curvy in original space).
- Sketch the boundary for **very small $C$** (wide margin in $\phi$-space — smoother boundary in original space, may misclassify some training points).
- **Justify your answer** — partial credit lives on the written reasoning, not just the sketch.

## Exam-relevant facts

- $K(x, z) = (x^\top z + c)^p$.
- **Quadratic** = $(x^\top z + 1)^2$ — mock §6.
- Implicit feature space = monomials of degree $\le p$, dimension $\binom{d+p}{p}$.
- Computed in $O(d)$ regardless of $p$ — that's the kernel-trick payoff.
- Higher $p$ → more flexible boundary → more risk of overfit.

## Related

- [[kernel-trick]] — the general technique.
- [[support-vector-machine]] — the canonical kernel-SVM application.
- [[lecture-15-kernels-i|SLP L15]] — kernel trick first introduced.
- [[lecture-16-kernels-ii|SLP L16]] — Mercer's condition, kernel construction rules, RBF.
