---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Boosting.pdf
created: 2026-05-03
---

# Boosting (general framework)

A **sequential ensemble method** that combines many **weak learners** (high-bias / low-variance predictors that underfit individually) into a strong learner that reduces **bias**. The mirror image of [[bagging]], which averages strong learners in parallel to reduce variance.

The lecture's defining slogan: *"boosting is gradient descent in function space."* Each weak learner takes a small step toward the target in the space of all possible predictors; many such steps compose into a strong final ensemble.

## The general recipe

Initialize $H_0 = 0$. At each iteration $t = 1, \ldots, T$:

1. **Find a weak learner** $h_t \in \mathcal{H}$ — any predictor with accuracy strictly better than 50% (binary classification) or non-orthogonal to the current residual (regression).
2. **Take a small step** in $h_t$'s direction:
$$
H_t(x) = H_{t-1}(x) + \alpha_t\, h_t(x).
$$
3. **Repeat** until convergence or $T$ iterations elapse.

Final ensemble:

$$
H(x) = \sum_{t=1}^{T} \alpha_t\, h_t(x).
$$

The step size $\alpha_t$ is the **shrinkage / learning rate**. Smaller steps require more iterations but yield a more accurate final ensemble (analogous to gradient descent step size).

## Why it works (the geometric argument)

Switch coordinates: let each axis of $\mathbb{R}^N$ correspond to one training point's label. The target is $\vec{y} = (y_1, \ldots, y_N) \in \mathbb{R}^N$. The current ensemble corresponds to $\vec{H} = (H(x_1), \ldots, H(x_N))$.

At each step we move from $\vec{H}_{t-1}$ to $\vec{H}_{t-1} + \alpha_t \vec{h}_t$. The step **gets us closer to $\vec{y}$** as long as $\vec{h}_t$ has angle $< 90°$ with the residual $\vec{y} - \vec{H}_{t-1}$ — equivalently, $\vec{h}_t$ has positive inner product with the residual. Pythagoras then guarantees the new distance² is smaller (provided the step doesn't overshoot).

If by chance $\vec{h}_t$ points the wrong way (angle $> 90°$), **flip its sign** ($\vec{h}_t \to -\vec{h}_t$) — this always switches the direction *unless* the angle is exactly 90° (perfectly orthogonal). The orthogonal case corresponds to a learner that gets exactly 50% accuracy on binary classification, and is when boosting terminates.

## What counts as a "weak learner"

- **Binary classification**: any predictor with accuracy **strictly greater than 50%**. Even 50.1% is enough.
- **Regression**: any predictor whose vector $\vec{h}$ has nonzero (and same-sign-as-needed) inner product with the residual.

The framework is **forgiving** because individual weakness is compensated by iteration count.

## Functional-gradient view

The sequential update can be derived as gradient descent on the loss as a function of the ensemble's predictions. At step $t$:

$$
h_t = \arg\min_{h \in \mathcal{H}} \ell(H_{t-1} + \alpha h) \approx \arg\min_{h \in \mathcal{H}} \left\langle \frac{\partial \ell}{\partial H},\, \alpha h \right\rangle.
$$

Treat the function $h$ as a vector $(h(x_1), \ldots, h(x_N)) \in \mathbb{R}^N$. The next weak learner should point in the direction of the **negative gradient** of $\ell$ w.r.t. the current ensemble. Different choices of $\ell$ give different boosting algorithms:

| Loss $\ell$ | Algorithm | Negative gradient |
| --- | --- | --- |
| **Squared** $\tfrac{1}{2}\sum(H(x_i) - y_i)^2$ | **Gradient boosting** | $y_i - H(x_i)$ — the **residual** |
| **Exponential** $\sum e^{-y_i H(x_i)}$ | **AdaBoost** (L14) | $y_i\, e^{-y_i H(x_i)}$ — re-weighted misclassified examples |
| **Logistic** $\sum \log(1 + e^{-y_i H(x_i)})$ | LogitBoost | similar |
| **Huber, quantile, ...** | various GBDT variants | task-specific |

Gradient boosting and AdaBoost are **sibling algorithms** under this framework — neither subsumes the other. They differ in which loss they target.

## Bias-variance argument

- Each weak learner has **high bias** (underfits by design) and **low variance** (e.g., a stump is barely affected by data perturbations).
- The additive ensemble $H = \sum_t \alpha_t h_t$ **reduces bias** because each $h_t$ corrects what previous $h_1, \ldots, h_{t-1}$ missed (the residuals).
- Variance can **grow with $T$** as the ensemble starts fitting noise in the residuals — boosting can overfit. Iteration count $T$ acts as a regularizer; validation error is **U-shaped** in $T$.

This is the §1g exam answer: *boosting iterations control model complexity analogously to early stopping or any other regularization knob.*

## Bagging vs boosting

| | **Bagging** (L12) | **Boosting** (L13) |
| --- | --- | --- |
| Base learners | strong, overfit-prone | weak, underfit-prone |
| Targets | **variance** | **bias** |
| Training | parallel, independent | sequential, each fitted to predecessors' residuals |
| Aggregation | average / majority vote | weighted sum |
| Tree depth typical | unbounded (fully grown) | shallow (often stumps) |
| Examples | Random Forest | Gradient boosting, AdaBoost |

## Why boosting can overfit

Unlike bagging, where more base learners almost never hurt, boosting **does** overfit if $T$ is too large. The ensemble keeps fitting to residuals, and eventually the residuals contain mostly noise. Standard fixes:

- **Shrinkage** — multiply each $\alpha_t$ by a small $\eta < 1$ (e.g., 0.1). More iterations needed but smaller steps generalize better.
- **Tree-depth limits** — keep base learners shallow.
- **Subsampling** rows or columns per tree (stochastic gradient boosting).
- **Early stopping** on a validation set.

## Historical note

- 1988 — M. Kearns (UPenn) asked: *can we combine many weak learners to make a strong learner?*
- 1990 — R. Schapire (Princeton) answered yes with the first boosting algorithm.
- 1995 — Freund & Schapire's **AdaBoost** (L14).
- 2001 — Friedman's **gradient boosting** unified the framework under functional gradient descent.
- ~2014 — XGBoost, LightGBM, CatBoost — modern production gradient-boosting implementations.

## Exam-relevant facts

- Boosting = sequential ensemble of **weak learners** combined additively.
- Final form: $H(x) = \sum_t \alpha_t h_t(x)$.
- Targets the **bias** term in the bias-variance decomposition.
- Iterations $T$ control complexity → validation error U-shaped in $T$.
- Weak learner = accuracy > 50% (binary) — any edge over chance is enough.
- "Boosting is gradient descent in function space" — different losses give different algorithms (gradient boosting, AdaBoost).

## Related

- [[gradient-boosting]] — the squared-loss instance.
- [[adaboost]] — the exponential-loss instance (sibling of gradient boosting).
- [[weak-learner]] — the > 50% requirement made precise.
- [[bagging]] — the variance-targeting counterpart.
- [[bias-variance-decomposition]] — what boosting and bagging both navigate.
- [[exponential-loss]] — the AdaBoost loss.
- [[decision-tree]] — the canonical base learner.
- [[lecture-13-boosting|SLP L13]] / [[lecture-14-adaboost|SLP L14]] — sources.
