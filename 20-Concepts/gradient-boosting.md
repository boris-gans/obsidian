---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Boosting.pdf
created: 2026-05-03
---

# Gradient boosting

The **squared-loss instance** of the [[boosting]] framework: at each iteration, fit a regression tree (CART) to the **residuals** of the current ensemble, then add it to the ensemble with a small step size.

In one line: *fit successive trees to what's left over.*

## The recipe

Initialize $H_0(x) = 0$. For $t = 1, \ldots, T$:

1. **Compute residuals** at every training point:
$$
t_i = y_i - H_{t-1}(x_i).
$$
2. **Train a regression tree** $h_t$ on the dataset $\{(x_i, t_i)\}_{i=1}^N$ — minimizing squared loss against the residuals (which is what a CART regression tree already does).
3. **Update the ensemble**:
$$
H_t(x) = H_{t-1}(x) + \alpha\, h_t(x),
$$
with a small fixed step $\alpha$ (typically called the **shrinkage** or learning rate, often $\eta = 0.1$).
4. **Repeat** until $T$ iterations.

Final predictor:

$$
H(x) = \sum_{t=1}^{T} \alpha\, h_t(x).
$$

## Why fitting trees to residuals is correct

The general boosting subproblem at step $t$ is

$$
h_t = \arg\min_{h \in \mathcal{H}} \left\langle \frac{\partial \ell}{\partial H},\, h \right\rangle.
$$

For squared loss $\ell(H) = \tfrac{1}{2}\sum_i (H(x_i) - y_i)^2$, $\partial \ell / \partial H(x_i) = H(x_i) - y_i$. Define $t_i = -\partial \ell / \partial H(x_i) = y_i - H(x_i)$ (the residual). The argmin objective becomes (after multiplying by 2, completing the square, and constraining $\sum_i h(x_i)^2 = 1$):

$$
h_t = \arg\min_h \sum_i (h(x_i) - t_i)^2.
$$

That's **squared loss with $t_i$ as the new target** — exactly the objective a CART regression tree minimizes. So **training a regression tree on $(x_i, t_i)$ is the gradient-boosting step for squared loss**. No new algorithm needed; reuse CART.

The lecture's exact phrasing: *"At every iteration, we update the target with the remainder!"*

## The full algorithm

```
H ← 0
for t = 1 to T:
    for i = 1 to N:
        t_i ← y_i − H(x_i)        # residuals = negative gradient
    h_t ← CART regression tree fit to {(x_i, t_i)}
    H ← H + α · h_t                # add scaled tree to ensemble
return H
```

## Hyperparameters

| Knob | Typical | What it does |
| --- | --- | --- |
| $T$ (number of trees) | 100–1000+ | More iterations = more capacity = can overfit. Use early stopping. |
| $\alpha$ (shrinkage / learning rate) | 0.01–0.1 | Smaller = needs more $T$ but generalizes better |
| Tree depth | 3–8 (shallow!) | Each tree should be a weak learner. Deeper = stronger per-tree but ensemble overfits sooner |
| Min samples per leaf | small | Lets each tree match local patterns in residuals |
| Subsample rate | 0.5–1.0 (rows) | Stochastic gradient boosting — adds randomization, reduces variance further |
| Colsample rate | 0.5–1.0 (features per tree) | Like Random Forest's feature subsampling |

The rule of thumb: **smaller $\alpha$ + larger $T$ generalizes better** than fewer iterations with bigger steps, at the cost of training time.

## Why each tree should be **shallow** (not fully grown)

This is the inverse of [[random-forest|Random Forest]]'s rule. In bagging, fully grown high-variance trees are the right base — averaging removes variance. In boosting, **shallow high-bias trees** (often stumps or trees of depth 3–6) are the right base because:

- The ensemble already corrects for bias by adding $T$ trees additively.
- Each tree only needs to capture *the dominant residual pattern* — not the full structure.
- Deep trees per round overfit the residuals quickly, causing the ensemble to overfit the data sooner.

## Generalization beyond squared loss

Replace the squared-loss derivation with another differentiable loss to get a different gradient-boosting variant. The general step is *fit a regression tree to the negative gradient of the loss at the current ensemble's predictions*.

| Loss | Negative gradient | Resulting algorithm |
| --- | --- | --- |
| Squared $\tfrac{1}{2}(H - y)^2$ | $y - H$ (residual) | classical gradient boosting (this note) |
| Logistic $\log(1 + e^{-yH})$ | $y / (1 + e^{yH})$ | LogitBoost / GBDT for classification |
| Exponential $e^{-yH}$ | $y\, e^{-yH}$ | (relates to AdaBoost — L14) |
| Huber | piecewise linear | robust regression boosting |
| Quantile | piecewise constant | quantile regression |

Modern implementations (XGBoost, LightGBM, CatBoost) wrap this framework with second-order Taylor expansions, regularization on tree complexity, sparse-feature handling, and clever data structures — but the core idea remains *fit a tree to the residual.*

## Why gradient boosting is currently the strongest tabular baseline

- Same advantages as Random Forest (no scaling needed, handles mixed features, robust to outliers when shallow).
- Plus **bias reduction** via additive composition — climbs past Random Forest's variance floor.
- Strong defaults; sensible regularization knobs.
- Modern implementations have heavy engineering for speed.

## Exam-relevant facts

- Gradient boosting = additive ensemble of regression trees fit sequentially to residuals.
- Final form: $H(x) = \sum_{t=1}^{T} \alpha\, h_t(x)$.
- Each tree minimizes squared loss against $t_i = y_i - H(x_i)$ (the current residuals).
- Trees should be **shallow** (high-bias / low-variance base learners).
- Number of iterations $T$ acts as a regularizer; validation error is U-shaped in $T$.
- "Linear combination of stumps" — §1h answer — describes the structural form of any gradient-boosted-stump or AdaBoost ensemble.

## Related

- [[boosting]] — the general framework.
- [[weak-learner]] — what each $h_t$ is.
- [[decision-tree]] — the base learner.
- [[mean-squared-error]] — the loss whose gradient gives residuals.
- [[bias-variance-decomposition]] — the lens for *why* it works.
- [[lecture-13-boosting|SLP L13]] — source.
