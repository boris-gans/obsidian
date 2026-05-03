---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-PCA.pdf
created: 2026-05-03
---

# Eigendecomposition

For a square matrix $A \in \mathbb{R}^{n \times n}$, an **eigenvector** $v$ and **eigenvalue** $\lambda$ satisfy
$$
A v = \lambda v.
$$
The action of $A$ on $v$ is just rescaling — no rotation. The pair $(\lambda, v)$ is called an *eigenpair*.

## Diagonalization

Stack $n$ linearly-independent eigenvectors as columns of $V$, and put the eigenvalues on the diagonal of $\Lambda$:
$$
A V = V \Lambda \quad\Longleftrightarrow\quad A = V \Lambda V^{-1}.
$$

This expresses $A$ in the **eigenbasis** — a coordinate system in which $A$ acts as pure scaling along each axis.

## Real, symmetric special case (the one PCA uses)

If $A$ is **real and symmetric** ($A = A^\top$):

1. All eigenvalues are real.
2. Eigenvectors for distinct eigenvalues are orthogonal; can be chosen orthonormal even when eigenvalues repeat.
3. So $V$ is **orthogonal**: $V^{-1} = V^\top$, and
$$
A = V \Lambda V^\top.
$$

The eigenvectors form an **orthonormal basis** in which to best describe $A$'s linear transformation.

## Sum-of-rank-1-matrices form

Equivalent expansion:
$$
A = \sum_{k=1}^{n} \lambda_k\, v_k v_k^\top.
$$

Each $v_k v_k^\top$ is an outer product — a rank-1 matrix. $A$ is decomposed as a weighted sum of these, with weights = eigenvalues. Useful for proving facts about quadratic forms (e.g. PCA's variance maximization):
$$
w^\top A w = \sum_k \lambda_k\, (v_k^\top w)^2.
$$

If $\lambda_1 \ge \cdots \ge \lambda_n$ and $\|w\| = 1$, this is maximized at $w = v_1$ with value $\lambda_1$.

## Use in PCA ([[principal-component-analysis]])

Apply to the [[covariance-matrix]] $\Sigma = \frac{1}{n} X^\top X$ (real, symmetric, PSD). The eigenvectors are the principal components; the eigenvalues are the variances along each PC.

## When to skip eigendecomposition

For large $d$, computing $X^\top X$ is prohibitive. Use [[singular-value-decomposition|SVD]] on $X$ directly: the right singular vectors of $X$ are the eigenvectors of $X^\top X$, with eigenvalues $\sigma_i^2$.

## Related

- [[principal-component-analysis]] — the canonical PCA pipeline diagonalizes $\Sigma$.
- [[covariance-matrix]] — the matrix being diagonalized.
- [[singular-value-decomposition]] — the practical alternative for large $d$.
