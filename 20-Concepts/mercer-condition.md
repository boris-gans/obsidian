---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Kernels-II(1).pdf
created: 2026-05-03
---

# Mercer's condition

The condition that a function $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ must satisfy to be a valid **kernel** — i.e., to admit a feature-space representation $k(x, z) = \langle \phi(x), \phi(z)\rangle$ for some $\phi$.

**A function $k$ is a valid kernel iff its Gram matrix on any finite set of inputs is symmetric positive semi-definite (PSD).**

The L16 deck calls this a "well-defined kernel" (no proof given) — Mercer's condition is the standard textbook name.

## Three equivalent characterizations of PSD

For a symmetric matrix $K \in \mathbb{R}^{n \times n}$, the following are equivalent:

1. **Eigenvalue form.** All eigenvalues of $K$ are non-negative.
2. **Factorization form.** $\exists$ a real matrix $P$ such that $K = P^\top P$. (Each column of $P$ is then a $\phi(x_i)$ — the implicit feature vector for training point $i$.)
3. **Quadratic-form definition.** $\forall$ real vector $\mathbf{x}$: $\mathbf{x}^\top K \mathbf{x} \ge 0$.

Form 2 makes the connection to feature spaces visible: the existence of $P$ such that $K_{ij} = \langle P_{:,i}, P_{:,j} \rangle$ is exactly the existence of a $\phi$ such that $k(x_i, x_j) = \langle \phi(x_i), \phi(x_j)\rangle$.

## How to verify Mercer's condition in practice

You almost never check Mercer directly. Two practical routes:

### Route 1: derive your kernel from the construction rules

If you build $k$ by composing already-valid kernels via the [[lecture-16-kernels-ii#The 8 rules for constructing well-defined kernels|8 well-defined-kernel rules]], it's automatically valid. This is by far the most common workflow.

### Route 2: numerical check on the training Gram matrix

Build $K_{ij} = k(x_i, x_j)$ on the training set, compute its eigenvalues. If any are noticeably negative (beyond numerical noise), the kernel isn't valid for that data. Some methods then **project to the nearest PSD matrix** — outside the L16 scope but standard in kernel-method libraries.

## Common kernels and their PSD status

| Kernel | $k(x, z)$ | PSD? |
| --- | --- | --- |
| Linear | $x^\top z$ | Always |
| Polynomial | $(1 + x^\top z)^d$ | Always (rules 1+3+4) |
| RBF / Gaussian | $\exp(-\|x-z\|^2 / \sigma^2)$ | Always |
| Set / count | $e^{|S_1 \cap S_2|}$ | Always (rules 1+7) |
| String $k$-mer | $\sum_u \mathrm{count}_u(s_1) \mathrm{count}_u(s_2)$ | Always (genuine inner product on count vectors) |
| **Sigmoid** | $\tanh(a\, x^\top z + b)$ | **Only for some $(a, b)$** — not always PSD |

The sigmoid "kernel" is widely used (it imitates a single-layer neural net), but its PSD-ness depends on hyperparameters — not Mercer-valid in general.

## Why Mercer's condition matters

Algorithms that depend on the kernel **only** through inner products (SVM dual, kernel ridge regression, kernel PCA) implicitly assume valid kernels. With a non-PSD "kernel":

- The optimization problem may not be convex.
- The implicit feature space $\phi$ doesn't actually exist.
- Solutions can become unstable / unreliable.

PSD validity is the foundation that makes the kernel trick mathematically coherent.

## Exam-relevant facts

- $k$ is a valid kernel iff Gram matrix $K_{ij} = k(x_i, x_j)$ is symmetric PSD.
- 3 equivalent PSD characterizations: non-negative eigenvalues; $K = P^\top P$ for some $P$; $\mathbf{x}^\top K \mathbf{x} \ge 0$.
- The 8 well-defined-kernel rules are the practical way to prove validity.
- Linear, polynomial, RBF, string-count, and set-intersection kernels are all valid; sigmoid kernel isn't always.

## Related

- [[kernel-trick]] — the technique Mercer's condition validates.
- [[polynomial-kernel]] / [[gaussian-kernel]] — important valid examples.
- [[support-vector-machine]] — kernel SVM relies on this validity.
- [[lecture-16-kernels-ii|SLP L16]] — source.
