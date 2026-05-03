---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-SVMs-I.pdf
created: 2026-05-03
---

# Margin (in linear classifiers)

The **distance from the decision boundary to the closest training points**. Larger margins are empirically and theoretically associated with better generalization. The Support Vector Machine is the classifier that *explicitly maximizes* the margin — the entire SVM derivation reduces to "make the margin as wide as possible while still classifying training points correctly (or paying for slack)."

## Geometric definition

For a linear classifier $\hat{y} = \text{sign}(\vec{w}\cdot\vec{x} + b)$, the **signed distance** from a point $\vec{x}$ to the decision boundary $\{\vec{x} : \vec{w}\cdot\vec{x} + b = 0\}$ is

$$
d(\vec{x}) = \frac{\vec{w} \cdot \vec{x} + b}{\|\vec{w}\|}.
$$

For a labeled training point $(\vec{x}_i, y_i)$ with $y_i \in \{+1, -1\}$, the **functional margin** is $y_i(\vec{w}\cdot\vec{x}_i + b)$ and the **geometric margin** is

$$
\gamma_i = \frac{y_i (\vec{w}\cdot\vec{x}_i + b)}{\|\vec{w}\|}.
$$

The **margin of the classifier** is the smallest geometric margin over the training set:

$$
\gamma = \min_i \gamma_i.
$$

## SVM normalization: width = 2/‖w‖

The SVM derivation pins down the scale: it requires $y_i(\vec{w}\cdot\vec{x}_i + b) \ge 1$ for all $i$, so the closest support vectors satisfy this with equality at $y_i(\vec{w}\cdot\vec{x}_i + b) = 1$. The geometric margin to a support vector is then $1/\|\vec{w}\|$, and the **full margin width** (positive gutter to negative gutter) is

$$
\text{width} = \frac{2}{\|\vec{w}\|}.
$$

This is the famous SVM identity. Maximizing width is equivalent to minimizing $\|\vec{w}\|$, hence the SVM primal $\min \tfrac{1}{2}\|\vec{w}\|^2$.

## Functional vs. geometric margin

These are two different scales of the same idea:

- **Functional margin** $y_i(\vec{w}\cdot\vec{x}_i + b)$ — depends on the scale of $\vec{w}$. Doubling $\vec{w}$ doubles the functional margin without changing the geometry.
- **Geometric margin** $y_i(\vec{w}\cdot\vec{x}_i + b)/\|\vec{w}\|$ — invariant to rescaling. This is the actual distance.

The SVM normalization fixes functional margin to $\ge 1$, which then makes geometric margin $\ge 1/\|\vec{w}\|$ — equivalent to maximizing geometric margin.

## Hard-margin vs. soft-margin

| Margin type | Geometric meaning | When |
| --- | --- | --- |
| Hard margin | All training points have geometric margin $\ge 1/\|\vec{w}\|$. No violations. | Linearly separable data only. |
| Soft margin | Some points may violate the margin constraint with cost $C\xi_i$. The SVM trades training error for wider margin. | Standard for real data. |

In soft-margin: a point with $\xi_i = 0$ has margin $\ge 1$. A point with $\xi_i \in (0, 1]$ is on the correct side but inside the margin. A point with $\xi_i > 1$ is misclassified.

## Why max margin generalizes better

Two complementary stories:

**Geometric story.** A wider margin means the boundary is far from any training point — a small perturbation of the training data, or of an unseen test point, doesn't change the prediction. The decision function is **less sensitive** to noise.

**Statistical story (Vapnik).** Structural risk minimization gives generalization bounds in terms of the *margin*, not the dimensionality of the input space. For a linearly separable dataset with margin $\gamma$, the VC-dimension of the max-margin classifier scales as $1/\gamma^2$ — independent of input dimension. **High-dim data with a clean margin generalizes well.** This is why SVMs work for text classification (BoW, $d \sim 10^4$) and image classification (kernel SVMs, $d \sim 10^6$).

## Margin in kernel SVMs (preview L15)

In kernel space, the margin is computed in the kernel-induced feature space, not the input space. Geometric margin to support vector $\vec{x}_i$:

$$
\gamma_i = \frac{y_i \big(\sum_j \alpha_j y_j K(\vec{x}_j, \vec{x}_i) + b\big)}{\sqrt{\sum_{j,k} \alpha_j \alpha_k y_j y_k K(\vec{x}_j, \vec{x}_k)}}.
$$

The kernel trick replaces inner products everywhere — but the geometric interpretation (closest point to boundary) lives in the kernel space, not the input space. Two points close in input space may be far in kernel space, and the margin sees the latter.

## Exam-relevant facts

- SVM margin width $= 2/\|\vec{w}\|$ (memorize).
- Maximize margin = minimize $\|\vec{w}\|$ = minimize $\tfrac{1}{2}\|\vec{w}\|^2$.
- Support vectors are points exactly on the margin gutter (hard margin) or inside the margin (soft margin).
- Geometric margin is the actual distance; functional margin scales with $\|\vec{w}\|$.
- Mock §6: large $C$ → narrow margin (low slack tolerance); small $C$ → wide margin (more slack tolerated).

## Related

- [[support-vector-machine]] — the classifier that maximizes margin.
- [[support-vector]] — the points that touch the gutters.
- [[hinge-loss]] — the loss that penalizes margin violations.
- [[slack-variables]] — the per-example margin-violation budget.
- [[lecture-09-linear-svms|SLP L09]] — source.
