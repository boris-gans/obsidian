---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Lec1-knn(1).pdf
created: 2026-05-03
---

# k-Nearest Neighbours

## Definition

A **non-parametric, instance-based** classifier (also useful for regression). At training time it just stores the labelled dataset $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$. At query time, given a test point $\mathbf{x}$:

1. Compute $\mathrm{dist}(\mathbf{x}, \mathbf{x}^{(i)})$ for every training point.
2. Take the $k$ training points with the smallest distances.
3. Predict the label by **majority vote** among those $k$ neighbours (for classification) or by averaging their labels (for regression).

For $k=1$ this reduces to **nearest-neighbour**: $\mathbf{x}^{*} = \arg\min_{\mathbf{x}^{(i)}} \mathrm{dist}(\mathbf{x}^{(i)}, \mathbf{x})$, $\hat{y} = t^{*}$.

## Why it matters

- **Simplest credible learner.** No training cost, no parametric assumption, no optimization. The decision boundary is whatever the data implies.
- **Naturally complex boundaries.** With enough data the 1-NN boundary is the union of perpendicular bisectors → a [[voronoi-diagram]] over training points; can fit arbitrarily intricate shapes.
- **Bayes-rate within a factor of 2.** Cover & Hart 1967: as $n \to \infty$, $\epsilon_{1\text{-NN}} \le 2\,\epsilon_{\text{Bayes}}$.
- **Foundation for the course.** Every later concept — distance metrics, decision boundaries, hyperparameter tuning, train/val/test splits, [[curse-of-dimensionality]], [[overfitting-underfitting]] — first appears here.

## $k$ as a complexity dial

| $k$ | Decision boundary | Train error | Test error |
| --- | --- | --- | --- |
| 1 | Wiggly, picks up every noise point | 0 | High (overfits) |
| moderate | Smooth | Rises with $k$ | Minimum at some $k^*$ |
| $\to n$ | Constant (predicts class prior) | High | High (underfits) |

Choose $k$ by [[cross-validation]], picking the $k$ that minimizes validation error.

## Algorithmic skeleton

```python
import numpy as np

def knn_predict(X_train, y_train, x_query, k=3):
    # Squared Euclidean distance — sqrt is monotonic, can be skipped.
    d2 = np.sum((X_train - x_query) ** 2, axis=1)
    nn_idx = np.argpartition(d2, k)[:k]      # k smallest distances
    nn_labels = y_train[nn_idx]
    # Majority vote (binary or multi-class)
    values, counts = np.unique(nn_labels, return_counts=True)
    return values[np.argmax(counts)]
```

## Costs and limits

- **Test-time complexity:** $O(kdN)$ — must compute $d$-dim distance to all $N$ training points.
- **Storage:** must keep all training data.
- **Curse of dimensionality:** in high $d$, points drawn from a probability distribution become roughly equidistant; "nearest" is no longer meaningfully near. The $k$-NN of a test point in $[0,1]^d$ require a hypercube of edge $\ell \approx (k/n)^{1/d}$, which → 1 fast.
- **Scale-sensitive:** features with larger ranges dominate the distance — see [[feature-normalization]].
- **Class-noise sensitive at $k=1$:** one mislabelled training point causes a misclassification region; voting with $k > 1$ smooths it ([[30-Sources/Statistical-Learning/pdf/SLP-Lec1-knn(1).pdf#page=64|slide 64]]).

## Exam-relevant facts

- **1-NN training error is always 0** ([[30-Sources/Statistical-Learning/pdf/SLP-Lec1-knn(1).pdf#page=75|slide 75]]). Each training point is its own closest neighbour.
- **kNN does not explicitly compute a decision boundary** — it can be inferred from the data and the chosen distance metric ([[30-Sources/Statistical-Learning/pdf/SLP-Lec1-knn(1).pdf#page=51|slide 51]]).
- **$k$ and the distance metric are hyperparameters** — set, not learned ([[30-Sources/Statistical-Learning/pdf/SLP-Lec1-knn(1).pdf#page=73|slide 73]]).

## Related

- [[minkowski-distance]] — the family of distances kNN can use.
- [[voronoi-diagram]] — visualizes the 1-NN decision boundary.
- [[curse-of-dimensionality]] — kNN's biggest theoretical limit.
- [[overfitting-underfitting]] — picking $k$ navigates this.
- [[cross-validation]] — how to choose $k$.
- [[feature-normalization]] — required preprocessing.
- [[hyperparameter]]
