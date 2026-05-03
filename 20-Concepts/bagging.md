---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Bagging.pdf
created: 2026-05-03
---

# Bagging

**B**ootstrap **Ag**gregating: an ensemble method that fights the **variance** term of the [[bias-variance-decomposition]] by training many predictors on bootstrap-resampled datasets and averaging their predictions.

## The recipe

Given a dataset $D$ of $N$ points and a learning algorithm $A$:

1. **Bootstrap**: sample $m$ new datasets $D_1, \ldots, D_m$ from $D$ **with replacement**, each of size $N$. About 63% of the original unique points appear in each $D_j$; the rest are out-of-bag.
2. **Train**: apply $A$ to each $D_j$, producing $m$ predictors $h_1, \ldots, h_m$. Embarrassingly parallel — each $h_j$ runs independently.
3. **Aggregate**: take the average:

$$
\bar{h}(x) = \frac{1}{m}\sum_{j=1}^{m} h_j(x).
$$

For classification, "aggregate" means **majority vote** over the hard predictions or mean of class probabilities. For regression, it's the literal arithmetic mean.

## Why it works

At a fixed test point $x$, individual predictors $h_j(x)$ scatter around their mean $\bar{h}(x)$ — that scatter is what the bias-variance decomposition calls **variance**. Picking the mean $\bar{h}(x)$ has zero scatter against itself. In the i.i.d. limit ($m \to \infty$ truly independent predictors), the variance of $\bar{h}$ shrinks as $\sigma^2 / m$.

In practice the $h_j$ are **correlated** (they share data via the bootstrap), so the variance of $\bar{h}$ has a floor:

$$
\text{Var}(\bar{h}) = \rho \sigma^2 + \frac{1 - \rho}{m}\sigma^2.
$$

The first term is set by pairwise correlation $\rho$ between predictors. Reducing $\rho$ — making the predictors **more diverse** — pushes the floor down. [[random-forest|Random Forest]]'s extra trick (random feature subsets at each split) does exactly this for trees.

**Bias is unchanged.** Averaging unbiased estimators preserves unbiasedness; averaging biased estimators preserves the same bias. This is why bagging is the right tool only for **high-variance / low-bias** base learners — making each one *less* would hurt bias for free.

## When to use bagging

- The base learner has **high variance**: deep decision trees, large unregularized polynomial models, $k$-NN with small $k$.
- You want a **strong baseline** without much hyperparameter tuning. Random Forests *"work out of the box."*
- You can afford the compute — training $m$ predictors is $m \times$ the cost of one (though they're parallelizable).

When **not** to use bagging:

- Base learner has **high bias** (e.g., a linear model or a stump). Bagging will average $m$ biased predictors and inherit that bias. Use **boosting** instead.
- Base learner already has very low variance (e.g., heavily regularized linear model). Nothing left to fix.

## Bagging vs boosting

The two main ensemble paradigms have opposite design rationales:

| | **Bagging** | **Boosting** |
| --- | --- | --- |
| Base learners | strong, overfit-prone (e.g., fully grown trees) | weak, underfit-prone (e.g., stumps) |
| Targets | **variance** | **bias** |
| Training | parallel, independent | sequential, each fitted to predecessors' errors |
| Aggregation | average / majority vote | weighted sum |
| Examples | Random Forest | AdaBoost, gradient boosting |

L12 introduces bagging; L13/L14 introduce boosting.

## Bias-variance precise statement

If $\bar{h}$ is the bagged predictor, the bias-variance decomposition reads:

- **Bias of $\bar{h}$**: same as bias of any individual $h_j$ (averaging preserves it).
- **Variance of $\bar{h}$ at a point $x$**: $\rho\sigma^2 + (1-\rho)\sigma^2/m$, lower than $\sigma^2$ of one predictor. Approaches $\rho\sigma^2$ as $m \to \infty$.
- **Noise**: unchanged (set by the data).

The precise reason "bagging reduces variance" — verbatim ML slogan.

## Out-of-bag (OOB) error

Each bootstrap leaves about $1/e \approx 36.8\%$ of unique training points out of $D_j$. Those out-of-bag points serve as a per-tree held-out set: predict on them with $h_j$, then average across all $j$ that didn't see point $i$ to get an OOB estimate of test error. This is a **free validation** estimate built into bagging — no separate held-out set needed. (Not introduced explicitly in SLP L12 but standard in practice.)

## Exam-relevant facts

- Bagging = **B**ootstrap + **Ag**gregate.
- Bootstrap = sample $m$ datasets from $D$ with replacement.
- Aggregate = average ($\bar{h}(x) = \frac{1}{m}\sum h_j(x)$) or majority vote.
- Targets the **variance** term of the bias-variance decomposition.
- **Bias is unchanged** — base learners must already be low-bias / high-variance.
- Embarrassingly parallel — each base learner trains independently.
- Random Forest = bagging applied to fully-grown decision trees, plus random feature subsets at each split.

## Related

- [[bootstrap-sampling]] — the resampling step in detail.
- [[random-forest]] — bagging's flagship application.
- [[bias-variance-decomposition]] — the lens that names what bagging fixes.
- [[decision-tree]] — the canonical bagged base learner.
- [[expected-predictor]] — what bagging empirically approximates.
- [[lecture-12-bagging|SLP L12]] — source.
