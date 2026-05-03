---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Lec1-knn(1).pdf
created: 2026-05-03
---

# Minkowski Distance

## Definition

A parametric family of distance metrics on $\mathbb{R}^d$ indexed by $p \ge 1$:

$$
\mathrm{dist}_p(\mathbf{x}, \mathbf{z}) = \left( \sum_{r=1}^{d} |x_r - z_r|^p \right)^{1/p}
$$

Special cases:

- **$p = 1$ Manhattan ($L_1$) distance:** $\sum_r |x_r - z_r|$. Sum of axis-aligned coordinate differences.
- **$p = 2$ Euclidean ($L_2$) distance:** $\sqrt{\sum_r (x_r - z_r)^2}$. The geometric straight-line distance.
- **$p \to \infty$ Chebyshev ($L_\infty$):** $\max_r |x_r - z_r|$. Largest single-coordinate gap.

## Why it matters

[[k-nearest-neighbors|kNN]] depends entirely on its distance metric. *"The nearest-neighbor algorithm relies on a distance metric: the better that metric reflects label similarity, the better the results will be"* ([[30-Sources/Statistical-Learning/pdf/SLP-Lec1-knn(1).pdf#page=46|slide 46]]).

The choice of $p$ changes the *geometry* of the kNN decision boundary (e.g., $L_1$ produces axis-aligned diamond-shaped neighbourhoods; $L_2$ gives spheres).

## Skip the square root for ranking

When the goal is just to *rank* points — as in finding nearest neighbours — the monotonic outer transform can be dropped. For Euclidean distance:

$$
\arg\min_i \|\mathbf{x} - \mathbf{x}^{(i)}\|_2 \;\equiv\; \arg\min_i \|\mathbf{x} - \mathbf{x}^{(i)}\|_2^2
$$

Saves a $\sqrt{\cdot}$ per training example ([[30-Sources/Statistical-Learning/pdf/SLP-Lec1-knn(1).pdf#page=50|slide 50]]).

## When *not* to use vanilla Minkowski

- When features are on different scales — first apply [[feature-normalization]].
- When some features are irrelevant — they pollute the sum (slide 88). Use feature weighting or selection.
- When the data has discrete / categorical features — Hamming or Jaccard may be more meaningful.

## Related

- [[k-nearest-neighbors]]
- [[feature-normalization]]
