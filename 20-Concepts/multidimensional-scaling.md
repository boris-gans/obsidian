---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-DimRed-II(1).pdf
created: 2026-05-03
---

# Multidimensional Scaling (MDS)

**Classic MDS** embeds a (squared) pairwise-distance matrix $D_s \in \mathbb{R}^{n \times n}$ into Euclidean space — given only the distances between $n$ objects, recover coordinates.

## Algorithm

1. **Double-centre** to get a kernel:
$$
K = -\tfrac{1}{2} C D_s C \quad\text{where}\quad C = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top.
$$
2. **Eigendecompose** $K = U \Lambda U^\top$.
3. **Embed** in $p$ dimensions by taking the top $p$ eigenpairs:
$$
X = U_p \Lambda_p^{1/2} \in \mathbb{R}^{n \times p}.
$$

The reconstructed coordinates are correct **up to a rigid transformation** (rotation, reflection, translation) — distances alone don't fix orientation or absolute position.

## Why double-centring?

Naively setting $K = -\tfrac{1}{2} D_s$ produces a matrix that's not centred — its row/column sums aren't zero, so it isn't a valid Gram matrix of a centred dataset. Multiplying by $C$ on both sides removes the row mean and column mean of $D_s$, yielding the Gram matrix of the centred data. The relation
$$
D_{s,ij} = K_{ii} + K_{jj} - 2 K_{ij}
$$
inverts cleanly only after this centring.

## Use cases

- **Psychological scaling.** Subjects rate similarities between $n$ stimuli; treat $1 - \text{similarity}$ as a distance and embed.
- **Geographic reconstruction.** Pairwise road distances between cities → 2-D map.
- **String distances.** Edit distances between protein sequences → low-dim visualization.
- **A subroutine of [[isomap|ISOMAP]].** Replace Euclidean distances with **geodesic** distances along a manifold, then run MDS — recovers non-linear embeddings (Swiss roll).

## Connection to PCA

Classic MDS on Euclidean distances is **equivalent to PCA**: the embedding it produces matches the [[principal-component-analysis|PCA]] coordinates of the centred data. The two algorithms differ only in *which matrix they diagonalize*: PCA does $X^\top X$ ($d \times d$); MDS does $K = -\tfrac{1}{2} C D_s C$ ($n \times n$). Both have the same right singular vectors of $X$ as outputs.

## Related

- [[position-distance-similarity]] — the Golden Trio that makes MDS possible.
- [[isomap]] — non-linear MDS using geodesic distances.
- [[kernel-pca]] — sibling method that diagonalizes a kernel matrix instead of a distance matrix.
- [[principal-component-analysis]] — equivalent for Euclidean distances.
- [[lecture-19-dim-reduction-ii]].
