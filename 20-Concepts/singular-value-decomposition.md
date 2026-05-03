---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-PCA.pdf
created: 2026-05-03
---

# Singular Value Decomposition (SVD)

For **any** real matrix $X \in \mathbb{R}^{n \times d}$ (need not be square), the SVD writes
$$
X = U \Sigma V^\top
$$
with:

- $U \in \mathbb{R}^{n \times n}$ — orthogonal; columns are the **left singular vectors**.
- $\Sigma \in \mathbb{R}^{n \times d}$ — "diagonal", entries $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ are the **singular values**.
- $V \in \mathbb{R}^{d \times d}$ — orthogonal; columns are the **right singular vectors**.

The rank of $X$ is at most $p = \min(n, d)$; in the **thin SVD** we drop zero rows/cols of $\Sigma$ and have $U \in \mathbb{R}^{n \times p}$, $\Sigma \in \mathbb{R}^{p \times p}$, $V \in \mathbb{R}^{d \times p}$.

## Connection to eigendecomposition

$$
X^\top X = V \Sigma^\top U^\top U \Sigma V^\top = V \Sigma^2 V^\top
$$
$$
X X^\top = U \Sigma V^\top V \Sigma^\top U^\top = U \Sigma^2 U^\top.
$$

So:

- **Right singular vectors of $X$** = eigenvectors of $X^\top X$, with eigenvalues $\sigma_i^2$.
- **Left singular vectors of $X$** = eigenvectors of $X X^\top$, same eigenvalues.

This is why SVD is the practical algorithm for [[principal-component-analysis|PCA]] — it produces the principal components (right singular vectors of the centered data) without ever forming the $d \times d$ matrix $X^\top X$ explicitly. Critical when $d$ is huge (genes, pixels).

## Use in PCA

For PCA on centered data $\tilde{X}$:

1. Run SVD: $\tilde{X} = U \Sigma V^\top$.
2. Columns of $V$ are the **principal components**.
3. Variance explained by the $k$-th PC is $\sigma_k^2 / n$.
4. The $k$-th PC's projection of all points is the $k$-th column of $U \Sigma$.

## Other uses (not in this lecture, but worth knowing)

- **Low-rank approximation** (Eckart-Young): the best rank-$p$ approximation of $X$ in Frobenius norm is $X_p = U_p \Sigma_p V_p^\top$ — keep the top $p$ singular triples. PCA's $p$-component reconstruction is this.
- **Pseudoinverse** $X^+ = V \Sigma^+ U^\top$ where $\Sigma^+$ inverts non-zero singular values.

## Related

- [[principal-component-analysis]] — the canonical use case.
- [[eigendecomposition]] — the special case for square symmetric matrices.
- [[covariance-matrix]] — the matrix SVD lets us avoid forming.
