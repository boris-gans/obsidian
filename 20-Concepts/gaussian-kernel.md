---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Kernels-II(1).pdf
created: 2026-05-03
---

# Gaussian / RBF kernel

The most popular [[kernel-trick|kernel function]] in machine learning. Often called the **RBF (Radial Basis Function) kernel**.

$$
k(x, z) = \exp\!\Big(-\frac{\|x - z\|^2}{\sigma^2}\Big).
$$

(The L16 deck uses $\sigma^2$ in the denominator. Other texts use $1/(2\sigma^2)$ or a precision parameter $\gamma = 1/(2\sigma^2)$ — same Gaussian shape, different convention.)

## Why it's called "Gaussian"

The kernel evaluates a Gaussian bell curve centred at one input, evaluated at the other. Two points are "similar" (high $k$) when close, near-zero when far apart.

## Why it's called "Radial Basis Function"

The kernel value depends **only** on the radial distance $\|x - z\|$, not on direction. Translation-invariant in feature space.

## The implicit feature space is infinite-dimensional

The Taylor expansion of $\exp$ shows the RBF kernel decomposes as an inner product in an infinite-dimensional space:
$$
\exp\!\Big(-\tfrac{\|x-z\|^2}{\sigma^2}\Big) = \exp\!\Big(-\tfrac{\|x\|^2 + \|z\|^2}{\sigma^2}\Big) \cdot \exp\!\Big(\tfrac{2 x^\top z}{\sigma^2}\Big),
$$
and the second factor expands as $\sum_{k=0}^{\infty} (2x^\top z / \sigma^2)^k / k!$ — an infinite polynomial expansion. Each term contributes its own monomial-product features. The resulting $\phi$ lives in an infinite-dimensional Hilbert space.

> *"It cannot even be computed."* — the deck. We can compute $k(x, z)$ in $O(d)$ time (one squared norm and an exponential), but $\phi(x)$ itself is uncomputable.

## Universal approximator property

> *"The most popular kernel: a universal approximator! Can make almost any dataset linearly separable (provided no two identical points have different labels)."*

For any finite training set with no contradicting labels, an SVM with an RBF kernel can in principle fit the data **exactly** — the implicit feature space is rich enough to separate any finite point cloud. (Practical regularization via $C$ keeps the model from overfitting.)

## The bandwidth $\sigma$ — the only knob

The bandwidth parameter $\sigma$ controls how quickly similarity decays with distance:

| $\sigma$ | Behavior | Risk |
| --- | --- | --- |
| **Very small** | Narrow Gaussians; only nearby points have non-trivial similarity | Each support vector influences a tiny region → very flexible / jagged decision boundary → **overfit** |
| **Medium** | Moderate falloff | Sweet spot |
| **Very large** | Wide Gaussians; all pairs have similar kernel value | Smooth decision boundary, possibly too smooth → **underfit** |

This is **the** RBF tuning loop: cross-validate $\sigma$ alongside the SVM regularizer $C$.

**Median heuristic.** A reasonable starting value: set $\sigma$ near the **median pairwise distance** in the training data. Then refine via cross-validation.

## Why RBF is usually the default

- **Most flexible** of the standard kernels — infinite-dimensional feature space.
- **Translation-invariant** — no preferred origin or direction.
- **Only one hyperparameter** ($\sigma$, plus the SVM $C$) — easy to tune.
- **Almost-universal** — fits almost anything with the right bandwidth.

When you don't know what kernel to start with on tabular non-text data, RBF is the safe first choice.

## When to NOT use RBF

- **High-dim sparse text** (BoW, TF-IDF features) — linear SVM is usually equivalent and faster.
- **Very large $n$** (more than ~10k points) — RBF SVMs cost $O(n^2)$ memory and don't scale gracefully. Modern alternatives: linear models with feature crossing, gradient-boosted trees, or neural networks.
- **Structured non-vector data** — use a domain-specific kernel (string, graph, tree) instead.

## Why it's a valid kernel (Mercer's condition)

Provable via either:

- **Direct factorization** through the Taylor expansion of $\exp$ above.
- **Construction-rule derivation**: $-\|x - z\|^2 / \sigma^2$ is a sum of dot products (rules 1+3); the exponential of this is well-defined by rule 7. ✓

Either way, the Gaussian kernel is always Mercer-valid. See [[mercer-condition]].

## Exam-relevant facts

- $k(x, z) = \exp(-\|x - z\|^2 / \sigma^2)$.
- Implicit feature space is **infinite-dimensional** — cannot be materialized.
- $\sigma$ controls smoothness: small $\sigma$ → flexible / overfit risk; large $\sigma$ → smooth / underfit risk.
- Universal approximator: separates almost any finite labeled dataset (no contradicting labels).
- Default kernel choice for non-text vector data.
- Always Mercer-valid (rules 1, 3, 7 of the well-defined-kernel construction).

## Related

- [[kernel-trick]] — the framework Gaussian kernel plugs into.
- [[polynomial-kernel]] — the alternative finite-degree kernel.
- [[mercer-condition]] — what makes Gaussian valid.
- [[support-vector-machine]] — the canonical RBF-kernel application.
- [[lecture-16-kernels-ii|SLP L16]] — source.
