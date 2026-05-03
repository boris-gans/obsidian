---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Kernels(1).pdf
created: 2026-05-03
---

# Kernel trick

A computational technique for running a linear algorithm in a high-dimensional **feature space** $\phi(x)$ without ever materializing $\phi(x)$ explicitly. Whenever the algorithm depends on its data **only through inner products** $\langle x_i, x_j \rangle$, you can substitute a **kernel function** $K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$ and get the algorithm running in $\phi$-space at essentially the cost of an $n \times n$ matrix.

> *"We can take our data, map it onto an exponentially high-dimensional space, and run our algorithm in this high-dimensional space — except we never once compute any inner products in that space!"* — Dyballa, L15.

## Why it works (the inner-product gateway)

Many linear algorithms have the property: the optimal weight vector is a **linear combination of training points**, $w^* = \sum_i \alpha_i x_i$ (or $\alpha_i y_i x_i$ for SVMs). This is provable by:

- **Induction on GD iterations** for any convex loss + GD training (L15's deck-side derivation).
- **KKT stationarity** for SVMs (the canonical Lagrangian-based derivation).
- **Representer theorem** for L2-regularized losses in a reproducing kernel Hilbert space (the general result).

Once $w^* = \sum_i \alpha_i x_i$ holds, both training and prediction can be rewritten purely in terms of inner products:
- Training updates: $\gamma_i = \sum_j \alpha_j \langle x_j, x_i \rangle - y_i$ (squared loss case).
- Prediction: $h(x_{\text{new}}) = \sum_i \alpha_i \langle x_i, x_{\text{new}} \rangle$.

Substitute $\langle x, z \rangle \to K(x, z) = \langle \phi(x), \phi(z) \rangle$ everywhere, and you've kernelized.

## The motivating example: all-subsets products

Define $\phi(x) = (1, x_1, x_2, \ldots, x_d, x_1 x_2, x_1 x_3, \ldots, x_1 x_2 \cdots x_d) \in \mathbb{R}^{2^d}$ — every product of every subset of features (including the empty product $1$).

The corresponding kernel:
$$
K(x, z) = \prod_{k=1}^{d} (1 + x_k z_k).
$$

**$O(d)$ to compute.** The expanded inner product has $2^d$ terms — for $d = 1000$, that's more than there are atoms in the universe — but the product form needs only $d$ multiplications.

That's the trick: a closed-form formula in the original space gives you the inner product in an exponentially high-dimensional feature space.

## Common kernels

| Kernel | $K(x, z)$ | Implicit feature space |
| --- | --- | --- |
| Linear | $x^\top z$ | original — no transformation |
| **Polynomial degree $p$** | $(x^\top z + c)^p$ | all monomials of degree $\le p$ — $\binom{d+p}{p}$-dim |
| **Quadratic** (mock §6) | $(x^\top z + 1)^2$ | monomials of degree $\le 2$ |
| **Gaussian / RBF** | $\exp(-\|x - z\|^2 / 2\sigma^2)$ | **infinite-dimensional** |
| All-subsets-products | $\prod_{k=1}^d (1 + x_k z_k)$ | $\mathbb{R}^{2^d}$ |

See [[polynomial-kernel]] for the polynomial / quadratic case in detail. L16 covers Gaussian / RBF and the construction rules for valid kernels (Mercer's condition).

## What "depends only on inner products" actually means

For an algorithm to be kernelizable, **every** appearance of a training point $x_i$ must be inside an inner product with another point. Examples:

| Algorithm | Inner-product form? | Kernelizable? |
| --- | --- | --- |
| **SVM** (dual form) | $\sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle$ | ✓ — kernel SVM |
| **Ridge regression** (dual) | $\hat{y} = \sum_i \alpha_i \langle x_i, x_{\text{new}} \rangle$ | ✓ — kernel ridge regression |
| **PCA** | depends on $\Sigma = \tfrac{1}{n}\sum_i x_i x_i^\top$ — **outer products** | only after dualization → kernel PCA |
| **Perceptron** | dual-form update uses $\langle x_i, x_j \rangle$ | ✓ — kernel perceptron |
| **k-NN** | distance $\|x - z\|^2 = \langle x, x \rangle - 2\langle x, z \rangle + \langle z, z \rangle$ | ✓ — kernel k-NN |
| Decision tree | uses individual feature values $x_{i, k}$ | ✗ |
| Random forest | same | ✗ |
| Naive Bayes | uses feature distributions per class | ✗ |

Tree-based methods aren't kernelizable because each split looks at one feature axis at a time — they don't factorize through inner products.

## Cost trade-off

Kernel methods cost $O(n^2)$ memory for the Gram matrix $K_{ij} = K(x_i, x_j)$ and $O(n^2)$ or $O(n^3)$ training time depending on the algorithm. They pay off when:

- Feature dimension $d$ is **huge or infinite** (text, image patches, RBF).
- Training set size $n$ is **moderate** (dozens to a few thousand).
- The non-linearity in $\phi$-space is essential to the problem.

For $n \gg d$ with linear-friendly data, primal methods (just train a linear model in the original space) are cheaper.

## Exam-relevant facts

- The kernel trick replaces every $\langle x_i, x_j \rangle$ with $K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$.
- **Quadratic kernel** $(x^\top z + 1)^2$ lives in the degree-≤2 monomial space — used on mock §6.
- **Gaussian kernel** lives in **infinite-dimensional** space.
- Algorithms must depend on data **only through inner products** to be kernelizable.
- The closed-form $\prod_{k=1}^d (1 + x_k z_k)$ takes $O(d)$ time but corresponds to a $2^d$-dim embedding.
- For SVMs specifically: $w^* = \sum_i \alpha_i y_i x_i$ (KKT), so prediction is $h(x) = \mathrm{sign}(\sum_i \alpha_i y_i K(x_i, x) + b)$ — only support vectors ($\alpha_i > 0$) contribute.

## Related

- [[support-vector-machine]] — the canonical kernel-trick consumer.
- [[polynomial-kernel]] — the quadratic-kernel example used on mock §6.
- [[support-vector]] — the points with $\alpha_i > 0$ that survive kernelization.
- [[lecture-15-kernels-i|SLP L15]] — the GD-based representer derivation + kernel trick.
- [[lecture-09-linear-svms|SLP L09]] — the SVM primal that the kernel trick lifts to feature space.
