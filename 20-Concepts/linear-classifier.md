---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-lec2(1).pdf
created: 2026-05-03
---

# Linear classifier

A classifier whose decision rule is a function of the **affine score** $z = w^T x + b$. Includes the [[perceptron]] (sign output), [[logistic-regression]] (sigmoid output), and the linear [[linear-svm|SVM]] (margin-based decision). The *boundary* between classes is a hyperplane $\{x : w^T x + b = 0\}$ — what differs between linear classifiers is **how each one chooses $w$ and $b$** (training criterion) and **what they output for points off the boundary** (hard sign / probability / margin).

## Decision rule and geometry

Two equivalent forms of the boundary:

$$
w \cdot x + b = 0 \qquad \Longleftrightarrow \qquad x_2 = -\tfrac{w_1}{w_2}\,x_1 - \tfrac{b}{w_2} \quad \text{(2-D)}
$$

The second form makes the line equation explicit ([[30-Sources/Statistical-Learning/pdf/SLP-lec2(1).pdf#page=30|slide 30]]). The vector $w$ is **perpendicular to the boundary**; the bias $b$ shifts the boundary along $w$.

The **score** $w \cdot x + b$ is:
- $0$ on the boundary,
- positive on the $+$ side (whose direction is $w$'s direction),
- negative on the $-$ side.

Its magnitude is proportional (up to $\|w\|$) to the perpendicular distance from $x$ to the boundary.

## The "templates" view

Stanford-style framing reused throughout SLP ([[30-Sources/Statistical-Learning/pdf/SLP-lec2(1).pdf#page=72|final slide]]): the rows of a weight matrix $W \in \mathbb{R}^{C \times d}$ are **per-class templates**. The score for class $c$ is

$$
s_c = W_c \cdot x + b_c,
$$

i.e. the dot product of the input with the class template. Classification picks the class with the highest score (binary: sign of $s_+ - s_-$; multi-class: softmax over $s$).

This view explains why "a classifier looks like image X" makes intuitive sense — the trained $W$ matrix literally *is* a stack of class images / class templates that look like the average example of each class.

## Three linear classifiers and their training criteria

| Classifier | Output for $x$ | Training criterion |
| --- | --- | --- |
| [[perceptron]] | $\mathrm{sign}(w^T x + b)$ | mistake-driven update; only converges if linearly separable |
| [[logistic-regression]] | $\sigma(w^T x + b) \in (0, 1)$ | minimize [[cross-entropy]] / negative log-likelihood / [[logistic-loss]] |
| linear [[linear-svm\|SVM]] | $\mathrm{sign}(w^T x + b)$ + margin | maximize $2/\|w\|$ s.t. margin constraints (hinge loss) |

All three share the same boundary geometry; what differs is the loss they minimize, which controls *which* separating hyperplane they pick when many exist.

## Why this matters in the course

L02 builds logistic regression by patching the perceptron's three failure modes (one per issue → one fix). L09–L10 add SVM and soft-margin / regularizer machinery. L15–L16 extend the same template/inner-product structure to non-linear boundaries via [[kernel-trick|kernels]]. The single object — "score = inner product with a learned template" — connects everything from L02 through L16.

## Limit: XOR

A **linear classifier cannot achieve zero training error on XOR** — XOR is not linearly separable. This is the standard example used (mock §2c) to motivate non-linear models: kernels, MLPs, decision trees. Memorize: linear SVM, logistic regression, single-neuron perceptron all fail on XOR; $k$-NN, decision trees, and any non-linear network can succeed.

## Related

- [[perceptron]] — sign-output linear classifier; the L02 starting point.
- [[logistic-regression]] — sigmoid-output linear classifier; the L02 endpoint.
- [[sigmoid]] — what turns a score into a probability.
- [[cross-entropy]] / [[logistic-loss]] — the loss that picks the best linear classifier.
- [[multilayer-perceptron]] — the *non*-linear generalization L03+ builds.
