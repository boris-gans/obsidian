---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-DimRed-II(1).pdf
created: 2026-05-03
---

# Kernel PCA

The kernelized version of [[principal-component-analysis|PCA]]: do PCA in an implicit, possibly infinite-dimensional, feature space $\phi(x)$ without ever materializing $\phi$. The trick is to work with the **kernel matrix** $K_{ij} = k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$ instead of the data matrix.

## Derivation (kernelizing PCA)

Standard PCA solves $\arg\max_u u^\top X^\top X u$ subject to $\|u\| = 1$ (working in the centred feature space $\Phi$, where each row is $\phi(x_i)$, the objective becomes $u^\top \Phi^\top \Phi u$).

Express $u$ in the span of the training examples (the "representer trick"): $u = \Phi^\top \alpha$ for some $\alpha \in \mathbb{R}^n$. Substitute:
$$
\tfrac{1}{n} u^\top \Phi^\top \Phi u = \tfrac{1}{n} \alpha^\top \Phi \Phi^\top \Phi \Phi^\top \alpha = \tfrac{1}{n} \alpha^\top K^2 \alpha.
$$

So the kernelized objective is $\arg\max_\alpha \alpha^\top K^2 \alpha$.

## The eigenstructure

$K^2$ has the **same eigenvectors** as $K$, with eigenvalues squared:

- Eigenvectors of kernel PCA = eigenvectors of $K$.
- Variance explained by the $k$-th component:
$$
\lambda_{\text{PCA}} = \tfrac{1}{n}\, \lambda_K^2.
$$

So the algorithm is just: **eigendecompose $K$**, take the top $p$ eigenvectors, those are the kernel-PCA coordinates of the data in implicit feature space.

## Algorithm

> 1. Build the kernel matrix $K_{ij} = k(x_i, x_j)$ ($n \times n$).
> 2. **Centre** $K$ in feature space: $K_c = C K C$ with $C = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$.
> 3. Eigendecompose $K_c = U \Lambda U^\top$.
> 4. The $i$-th data point's kernel-PCA coordinates (top $p$): the $i$-th entries of the columns $U_1, \ldots, U_p$, scaled by $\sqrt{\lambda_k}$.

## Standard PCA vs. kernel PCA

| | Standard PCA | Kernel PCA |
| --- | --- | --- |
| Diagonalize | $X^\top X$ ($d \times d$) | $K$ ($n \times n$) |
| Captures | Linear directions in input space | Non-linear directions via $\phi$ |
| Cost-favourable when | $d \ll n$ | $n \ll d$ or $d = \infty$ (RBF) |
| Recovers Swiss roll? | No | Yes, with the right kernel (e.g., RBF + manifold-learning variants) |

## Why "K^2" and not "K"?

A common point of confusion: standard PCA diagonalizes $X^\top X$. Why does kernel PCA effectively diagonalize $K^2$ but get the same eigenvectors as $K$? Because $K = X X^\top$ and $X^\top X$ are related — they share the same non-zero eigenvalues, and the right singular vectors of $X$ are eigenvectors of $X^\top X$ while the left singular vectors are eigenvectors of $X X^\top = K$. The substitution $u = X^\top \alpha$ shifts the variable from input space to coefficient space, where the objective becomes $\alpha^\top K^2 \alpha$ but $K$'s eigenvectors carry the same direction information.

## Related

- [[principal-component-analysis]] — the linear ancestor.
- [[kernel-trick]] — PCA depends on outer products $x x^\top$, so it kernelizes only after the dualization $u = X^\top \alpha$.
- [[multidimensional-scaling]] — sibling method that diagonalizes a kernel-from-distance matrix.
- [[isomap]] — uses kernel-PCA / MDS as a subroutine.
- [[lecture-19-dim-reduction-ii]].
