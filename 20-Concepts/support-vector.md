---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-SVMs-I.pdf
created: 2026-05-03
---

# Support vector

A training example whose constraint in the SVM optimization is **active** at the optimum — geometrically, a point sitting **exactly on the margin gutter** $y_i(\vec{w}\cdot\vec{x}_i + b) = 1$ (hard margin) or **inside the margin** (soft margin, $\xi_i > 0$). The decision boundary is **fully determined** by these points; all non-support-vector training examples could be moved freely (as long as they remain outside the margin) without changing the SVM.

## Hard-margin definition

A training point $\vec{x}_i$ with label $y_i$ is a support vector iff

$$
y_i (\vec{w} \cdot \vec{x}_i + b) = 1.
$$

Equivalently, it sits on one of the two gutters. **Every other training point** satisfies the strict inequality $y_i(\vec{w}\cdot\vec{x}_i + b) > 1$ — those points are *outside* the margin and don't constrain the optimization.

## Soft-margin definition

With slack variables, support vectors split into three categories:

| SV type | Constraint | Slack | Position |
| --- | --- | --- | --- |
| **Margin SV** | $y_i(\vec{w}\cdot\vec{x}_i + b) = 1$, $\xi_i = 0$ | $\xi_i = 0$ | exactly on the gutter |
| **Inside-margin SV** | $0 < \xi_i \le 1$, correctly classified | margin violated | between gutter and boundary, but on correct side |
| **Misclassified SV** | $\xi_i > 1$ | crosses the boundary | on the wrong side of the median line |

All three are support vectors — points whose constraints are *not* satisfied with margin $\ge 1$.

## Why support vectors are everything

The Lagrangian dual of the SVM (covered in [[lecture-15-kernels-i|L15]]) gives:

$$
\vec{w}^* = \sum_{i \in \text{SV}} \alpha_i^* y_i \vec{x}_i,
$$

with $\alpha_i^* > 0$ exactly for support vectors and $\alpha_i^* = 0$ for all other training points. The optimal weight vector is a **linear combination of support vectors only**.

Practical consequences:

- **Sparsity.** Most training points have $\alpha_i = 0$ — once trained, only the SV subset is needed for prediction.
- **Insensitivity to non-SV outliers.** Points far from the boundary contribute zero. A far-away outlier doesn't move the SVM. (Logistic regression doesn't have this property — every point's logistic loss is positive.)
- **Identifiability.** The boundary depends *only* on the SVs. Compute the model from any superset of the SVs and you get the same answer.

This is the mechanism behind mock §1c: *"the decision boundary depends only on the support vectors."* **True**, and it's the key fact distinguishing SVM from LR.

## How many support vectors are there?

In high-dimensional or noisy data, almost every point can become a support vector — the margin gets thin and many points end up inside it. In low-dimensional well-separated data, the number of SVs can be as small as $d + 1$ (the minimum needed to define a hyperplane in $d$ dimensions). A common heuristic: when ~half your training data is in the SV set, the model is probably overfitting; reduce $C$ to widen the margin.

## Connection to kernels (preview L15)

Once we move to kernel SVMs, training and prediction depend only on **inner products** between support vectors and other points: $K(\vec{x}_i, \vec{x}_j) = \vec{x}_i \cdot \vec{x}_j$ (or a kernel substitute). The dual formulation makes this explicit — the kernel trick is precisely the substitution

$$
\vec{w} \cdot \vec{x}_{\text{new}} = \sum_{i \in \text{SV}} \alpha_i y_i \, K(\vec{x}_i, \vec{x}_{\text{new}}),
$$

and the SVs are still the only training points that matter at prediction time.

## Exam-relevant facts

- A support vector is a training point with **active** constraint (equality at the gutter, or violated with $\xi_i > 0$).
- **Mock §1c**: the decision boundary depends *only* on the support vectors. **True.**
- Non-SV points contribute nothing to the optimum — they could be moved (within their half-space) without changing the SVM.
- In the dual, $\alpha_i > 0$ ⟺ support vector.
- The number of SVs is *not* fixed — depends on data and $C$.

## Related

- [[support-vector-machine]] — the model.
- [[margin]] — the width that the SVs define.
- [[slack-variables]] — what makes a point a soft-margin SV.
- [[lecture-09-linear-svms|SLP L09]] — source.
