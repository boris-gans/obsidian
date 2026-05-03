---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Lec1-knn(1).pdf
created: 2026-05-03
---

# Curse of Dimensionality

## Definition

The collection of pathologies that arise when working with high-dimensional feature spaces ($d \gg 1$):

1. **Sample sparsity.** To cover a $d$-dimensional grid at fixed resolution requires *exponentially* many points: 4 values per axis → $4^d$ grid points ([[30-Sources/Statistical-Learning/pdf/SLP-Lec1-knn(1).pdf#page=95|slide 95]]).
2. **Distances concentrate.** Points sampled from a distribution become roughly equidistant from each other; "nearest" stops being informative.
3. **Volume concentrates near the boundary.** Most of the volume of a high-dim hypercube is near its surface.
4. **Required training-set size grows exponentially with $d$** to maintain a given density.

## Formalisation for kNN ([[30-Sources/Statistical-Learning/pdf/SLP-Lec1-knn(1).pdf#page=104|slide 104]])

Imagine training data sampled uniformly from $[0,1]^d$. For a test point, let $\ell$ be the edge length of the smallest hypercube containing its $k$ nearest neighbours. Then

$$
\ell^d \approx \frac{k}{n}, \qquad \ell \approx \left(\frac{k}{n}\right)^{1/d}.
$$

For $n = 1000$, $k = 10$:

| $d$ | $\ell$ |
| --- | --- |
| 2 | 0.10 |
| 10 | 0.63 |
| 100 | 0.955 |
| 1000 | 0.9954 |

By $d \approx 100$, the "10 nearest neighbours" of a test point require almost the entire feature space. At that point those neighbours aren't meaningfully *near* — they're just the 10 closest of a uniformly-spread crowd.

To shrink $\ell$ back to $0.1$ at large $d$ would need

$$
n = k \cdot \left(\frac{1}{\ell}\right)^d = k \cdot 10^d
$$

which is exponential in $d$. *"For $d > 100$ we would need far more data points than there are electrons in the universe"* (extra slide).

## Why it matters

- Breaks the **kNN assumption** that "similar points share similar labels": there's no longer a meaningful notion of "similar" in raw feature space.
- Motivates **dimensionality reduction** ([[principal-component-analysis|PCA]] and friends in L18–19) and **kernel methods** (L15–16) which work in *induced* feature spaces with structure.
- Makes brute-force grid-search hyperparameter tuning infeasible at high $d$.

## Mitigations

- Drop irrelevant features (raises effective signal-to-noise per dimension).
- Project to lower dimension via PCA or random projection.
- Use distance metrics learned for the task ("metric learning" — slide 98 mention).
- Replace the inner product with a [[kernel-trick|kernel]] adapted to the data geometry.

## Related

- [[k-nearest-neighbors]] — the canonical victim.
- [[principal-component-analysis]]
- [[kernel-trick]]
