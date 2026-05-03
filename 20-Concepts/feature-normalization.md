---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Lec1-knn(1).pdf
created: 2026-05-03
---

# Feature Normalization

## Definition

Pre-processing each numerical feature so they are on comparable scales. Two common schemes:

- **Min–max scaling** — linearly map each feature into $[0, 1]$:
  $$\tilde{x}_j = \frac{x_j - \min_j}{\max_j - \min_j}.$$
- **Z-score (standardisation)** — center to zero mean, scale to unit variance:
  $$\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}.$$

> *"If some attributes (coordinates of $x$) have larger ranges, they are treated as more important. Normalize scale: linearly scale each feature into $[0,1]$, or scale each dimension to have 0 mean and variance 1: $(x_j - m)/\sigma$."* ([[30-Sources/Statistical-Learning/pdf/SLP-Lec1-knn(1).pdf#page=86|slide 86]])

## Why it matters

Algorithms that rely on **distances or inner products** weight features implicitly by their scale. If `weight` is in grams (range $\sim 100$) and `acidity` is in pH (range $\sim 5$), the Euclidean distance is dominated by `weight` even when `acidity` is the more informative feature.

Affected algorithms include [[k-nearest-neighbors]], [[k-means]], [[principal-component-analysis]] (depends on covariances), [[support-vector-machine|SVMs]] (depend on inner products and the regularization term $\|w\|^2$), gradient-descent-trained linear / neural models (large-scale features dominate gradients).

Largely **unaffected**: tree-based methods (decision trees, random forest, gradient boosting) — they make axis-aligned splits and are invariant to monotonic per-feature transformations.

## Where to fit the scaler

In a pipeline with cross-validation:

- **Fit on train only.** Compute $\mu_j, \sigma_j$ (or $\min, \max$) on the training fold.
- **Apply the same transform to validation and test.** Otherwise statistics from the val/test set leak into training.

## Beyond min-max and z-score

- **Robust scaling** — use median and IQR instead of mean and std. Less sensitive to outliers.
- **Log transform** — for heavy-tailed or strictly positive features (counts, prices, durations).
- **Per-feature weighting** — explicitly multiply each feature by a chosen weight $w_j$ so the distance becomes $\sqrt{\sum_j w_j (x_j - z_j)^2}$. Equivalent to fitting a (diagonal) Mahalanobis distance.

## Related

- [[k-nearest-neighbors]]
- [[minkowski-distance]]
