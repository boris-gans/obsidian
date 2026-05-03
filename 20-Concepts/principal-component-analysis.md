---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-PCA.pdf
created: 2026-05-03
---

# Principal Component Analysis (PCA)

A **linear, unsupervised** dimensionality-reduction technique. Given centered data $X \in \mathbb{R}^{n \times d}$ (rows = points), find an ordered orthonormal basis $v_1, v_2, \ldots, v_d$ — the **principal components (PCs)** — such that the projection onto $v_1$ has the largest variance of any unit-vector projection, $v_2$ the largest among directions orthogonal to $v_1$, and so on. Truncating to the top $p \ll d$ PCs gives the **MSE-optimal $p$-dimensional linear approximation** of the data.

## Two equivalent objectives

Both reduce to maximizing $w^\top X^\top X\, w$ subject to $\|w\| = 1$:

1. **Min reconstruction error.** Find the $p$-dim subspace minimizing $\sum_i \|x_i - \mathrm{proj}_{\mathrm{subspace}}(x_i)\|^2$.
2. **Max projected variance.** Find $w$ maximizing $\mathrm{Var}(Xw) = \frac{1}{n}\|Xw\|^2$.

Pythagoras makes them the same: $\|x_i\|^2 = \|\text{proj}\|^2 + \|\text{residual}\|^2$, so minimizing residual = maximizing projection.

## The algorithm

Given data matrix $X \in \mathbb{R}^{n \times d}$ with rows $x_i^\top$:

1. **Center.** $\bar{x} = \frac{1}{n}\sum_i x_i$, $\tilde{x}_i = x_i - \bar{x}$. Stack as $\tilde{X}$.
2. **Covariance.** $\Sigma = \frac{1}{n}\tilde{X}^\top \tilde{X} \in \mathbb{R}^{d \times d}$ ([[covariance-matrix]]).
3. **Eigendecompose.** $\Sigma = V \Lambda V^\top$ with eigenvalues sorted $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d \ge 0$. ([[eigendecomposition]])
4. **Read off.** $k$-th PC = $v_k$ (column of $V$); variance explained = $\lambda_k$.
5. **Project.** Centered point $\tilde{x}$ onto top $p$ PCs: $V_p^\top \tilde{x} \in \mathbb{R}^p$.

## Better algorithm via SVD

Computing $\Sigma$ explicitly is prohibitive for large $d$ (genes, pixels). Instead, run SVD on $X$ directly: $X = U \Sigma_{\text{SVD}} V^\top$. Then $X^\top X = V \Sigma_{\text{SVD}}^2 V^\top$, so:

- The **right singular vectors** of $X$ (columns of $V$) **are the PCs**.
- The eigenvalues of $\Sigma_{\text{cov}}$ are $\sigma_i^2 / n$.

Numerical SVD methods compute one singular vector at a time. ([[singular-value-decomposition]])

## Worked 3-point 2-D example (mock §4 shape)

$x_1 = (0,0),\ x_2 = (2,2),\ x_3 = (4,4)$.

- Mean $\bar{x} = (2, 2)$.
- Centered: $(-2,-2), (0,0), (2,2)$.
- Covariance:
$$
\Sigma = \frac{1}{3}\begin{pmatrix} 8 & 8 \\ 8 & 8 \end{pmatrix} = \begin{pmatrix} 8/3 & 8/3 \\ 8/3 & 8/3 \end{pmatrix} \quad (2 \times 2 = d \times d).
$$
- Eigenvalues: $\lambda_1 = 16/3$, $\lambda_2 = 0$ (rank-1 matrix).
- First PC: $v_1 = \tfrac{1}{\sqrt{2}}(1, 1)^\top$ (unit vector along the diagonal).
- Projections: $-2\sqrt{2}, 0, 2\sqrt{2}$ — lossless 1-D summary.

## Choosing $p$ (number of components)

Variance retained at cutoff $p$:
$$
\frac{\sum_{k=1}^{p} \lambda_k}{\sum_{k=1}^{d} \lambda_k}.
$$

Pick $p$ to retain ≥ 90% (or 95%, or 99%); or look for an "elbow" in the scree plot.

## When PCA fails

PCA finds the best **linear** subspace. If the data lies on a non-linear manifold (a curve, a surface like the **Swiss roll**), the best linear projection destroys local geometry — points that were "close along the surface" get squashed onto each other. Maximum-variance directions may not be the most interesting ones. See [[intrinsic-dimension]], [[manifold-learning]], [[kernel-pca]], [[isomap]].

## Memorize-cold (mock §4)

- Centered data first; $\bar{x}$ is in the original space.
- $\Sigma$ is $d \times d$ regardless of $n$.
- First PC is a **unit** eigenvector with the **largest** eigenvalue.
- Pipeline: center → covariance → eigendecompose → project.

## Related

- [[lecture-18-pca]] — the lecture treatment with full derivation.
- [[lecture-19-dim-reduction-ii]] — non-linear extensions.
- [[covariance-matrix]], [[eigendecomposition]], [[singular-value-decomposition]] — prerequisites.
- [[curse-of-dimensionality]] — the L01 problem PCA partially addresses.
- [[kernel-trick]] — PCA depends on outer products $x_i x_i^\top$, so it kernelizes only after dualization → kernel PCA.
- [[k-means]] — the other Phase F (unsupervised) algorithm.
