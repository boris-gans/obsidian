---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Bagging.pdf
created: 2026-05-03
---

# Random Forest

**Bagging applied to fully-grown decision trees**, plus one extra trick — at each split, choose the best feature from a *random subset* of features instead of all of them. The combination "works out of the box" and is one of the strongest baselines for tabular data.

## The recipe

Given dataset $D$ of $N$ points and $d$ features:

1. **Bootstrap**: sample $m$ datasets $D_1, \ldots, D_m$ from $D$ **with replacement**, each of size up to $N$.
2. **Train each tree** $T_j$ on $D_j$ **all the way to the end** — *let it overfit*. Don't prune; let leaves be pure.
3. **At every split during training**, instead of choosing the best split over all $d$ features, **choose the best split from a random subset of $k$ features**, with $k < d$.
4. **Aggregate**: for classification use majority vote across $T_1, \ldots, T_m$; for regression take the mean.

## Hyperparameter cheat sheet

| Hyperparameter | Default | Why |
| --- | --- | --- |
| $m$ (number of trees) | **as large as you can afford** | More trees never hurt; diminishing returns past a few hundred |
| $k$ (features per split) | **$k = \sqrt{d}$** | Statistically a good default; classification heuristic |
| Tree depth | **unbounded** | Each tree should overfit — bagging fixes the variance |
| Min samples per leaf | tiny (1–5) | Same reason — let each tree be expressive |

The slide deck states the $k = \sqrt{d}$ heuristic explicitly ([[30-Sources/Statistical-Learning/pdf/SLP-Bagging.pdf#page=37|slide 37]]). For regression, $k = d/3$ is sometimes used instead.

## Why fully-grown trees and not pruned ones?

A pruned tree has lower variance but higher bias. Random Forest doesn't want lower variance per tree — bagging will reduce variance for free, and reducing bias is much harder. So each individual tree is **maximally expressive** (low bias, high variance), and the forest's aggregation is what handles variance. Pruning would waste bias-reduction budget.

## Why the random feature subset?

Plain bagging on trees is good but gets stuck on a known problem: if there's one strongly predictive feature, **every** bootstrap tree will split on it at the root. The trees end up similar at the top — **correlated**. Correlated predictors don't average well: variance of the mean has a floor at $\rho \sigma^2$ where $\rho$ is the pairwise correlation.

Forcing each split to consider only $k = \sqrt{d}$ random features:

- A given tree might not even *see* the strongest feature at the root.
- Different trees split on different features at each level.
- Pairwise correlation $\rho$ drops; variance of the mean drops further.

This **decorrelation** is what separates Random Forest from "just bagged trees" and is responsible for the bulk of its empirical strength.

## Bias-variance argument made concrete

Each individual tree has:

- **High variance** — small data perturbation can re-build the entire tree.
- **Low bias** — fully grown, the tree can match the training labels exactly (modulo discrete leaf assignments).

Aggregating $m$ such trees:

- **Bias**: unchanged. Stays low.
- **Variance**: shrinks as $\sigma^2 \cdot \big[\rho + (1-\rho)/m\big]$. Random feature subsets reduce $\rho$; large $m$ shrinks the second term. Both work toward zero.
- **Noise**: irreducible.

Net result: low bias + low variance = strong test performance.

## Out-of-bag (OOB) error

Each bootstrap leaves ~37% of the points out (the OOB set). Random Forest gives a free validation estimate by, for each point $x_i$ in $D$, averaging the predictions of only those trees whose bootstrap didn't include $x_i$. This is the **OOB error**, and it's an unbiased estimate of test error — no separate held-out set required.

## Variable importance (practical bonus)

Random Forest provides two free measures of feature importance:

1. **Mean decrease in impurity (Gini importance)**: average across trees of the impurity reduction each feature contributes at its splits.
2. **Permutation importance**: shuffle a feature's values across the OOB set and measure the increase in error. The bigger the increase, the more important the feature.

Not introduced in this SLP lecture but standard in any practical use.

## Why "works out of the box"

- Few hyperparameters, all with sensible defaults.
- Insensitive to feature scaling (it's a tree-based method).
- Handles mixed feature types (numeric + categorical).
- Robust to outliers.
- Built-in OOB validation, built-in feature importance.
- Embarrassingly parallel training.

The classic result: on most tabular problems, a vanilla Random Forest is hard to beat without serious feature engineering or moving to gradient-boosted trees (L13).

## Exam-relevant facts

- Random Forest = **bagging on fully-grown decision trees** + **random feature subsets at each split**.
- $k = \sqrt{d}$ is the canonical feature-subset size.
- $m$ (number of trees) should be as large as possible.
- Trees are **not pruned** — let them overfit; bagging fixes variance.
- Random feature subsets **decorrelate** the trees, allowing variance to drop further than plain bagging would.

## Related

- [[bagging]] — the technique Random Forest extends.
- [[bootstrap-sampling]] — the resampling step.
- [[decision-tree]] — the base learner.
- [[bias-variance-decomposition]] — the lens for *why* it works.
- [[lecture-12-bagging|SLP L12]] — source.
