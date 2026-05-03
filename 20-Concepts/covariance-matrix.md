---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-PCA.pdf
created: 2026-05-03
---

# Covariance matrix

For a centered data matrix $X \in \mathbb{R}^{n \times d}$ (rows = observations, columns = features, column means subtracted), the **sample covariance matrix** is:

$$
\Sigma = \frac{1}{n} X^\top X \in \mathbb{R}^{d \times d}.
$$

It captures *how features vary together, on average*.

## Entries

Writing $x_i^{(k)}$ for feature $i$ of the $k$-th observation (after centering, $\bar{x}_i = 0$):

- **Diagonal** — variance of each feature:
$$
\Sigma_{ii} = \frac{1}{n}\sum_{k=1}^{n} (x_i^{(k)})^2 = \mathrm{Var}(\text{feature } i).
$$
- **Off-diagonal** — covariance between features $i$ and $j$:
$$
\Sigma_{ij} = \frac{1}{n}\sum_{k=1}^{n} x_i^{(k)} x_j^{(k)} = \mathrm{Cov}(\text{feature } i, \text{feature } j).
$$

## Properties

- **Symmetric** ($\Sigma_{ij} = \Sigma_{ji}$) and **positive semi-definite** ($w^\top \Sigma w = \frac{1}{n}\|Xw\|^2 \ge 0$).
- Real, symmetric ⇒ eigendecomposable with **orthonormal** eigenvectors and real, non-negative eigenvalues.
- **Shape is always $d \times d$** — independent of $n$. (Common exam-pressure mistake: writing $n \times n$.)
- Diagonalizing $\Sigma$ rotates into a basis where features are uncorrelated; the eigenvectors are the [[principal-component-analysis|principal components]].

## Convention note: $1/n$ vs $1/(n-1)$

Dyballa's slides use $\frac{1}{n}$. Statistical inference often uses Bessel's correction $\frac{1}{n-1}$ for an unbiased estimator. On the SLP exam, use $\frac{1}{n}$ unless the question states otherwise.

## Related

- [[principal-component-analysis]] — the canonical use case.
- [[eigendecomposition]] — how to extract the PCs.
- [[lecture-18-pca]] — the lecture treatment.
