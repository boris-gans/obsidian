---
tags: [flashcards, Statistical-Learning]
---

# Lecture 12 — Bagging and Random Forests

What does **bagging** stand for?
?
**B**ootstrap **Ag**gregating. The two-step pipeline: (1) bootstrap-sample $m$ datasets from the training set; (2) train one predictor per sample and aggregate (average for regression, majority vote for classification).

What is **bootstrap sampling**?
?
Sampling with replacement from a fixed dataset $D$ of size $N$ to produce a new dataset $D^*$ of the same size $N$. The new dataset typically has repeated points and omits some original points (~37% are out-of-bag).

What is the **bagged predictor formula** for regression?
?
$\bar{h}(x) = \frac{1}{m}\sum_{j=1}^{m} h_j(x)$, where each $h_j$ is trained on a bootstrap sample $D_j$.

How does bagging aggregate predictions for **classification**?
?
**Majority vote** over the hard predictions of the $m$ classifiers (or mean of their class probabilities, then argmax).

Which term in the bias-variance decomposition does bagging target?
?
**Variance.** Bagging averages the per-predictor variance toward zero (in the i.i.d. limit), while leaving bias unchanged.

What kind of base learner is bagging the right tool for?
?
**High-variance, low-bias** base learners — e.g., fully grown decision trees, $k$-NN with small $k$, large unregularized polynomial models. Bagging cures the variance; the bias floor is already low.

What kind of base learner is bagging **wrong** for?
?
**High-bias** base learners (e.g., linear models or stumps). Averaging biased predictors preserves the bias. Use **boosting** instead.

What's the formula for the variance of the bagged predictor when individual predictors are correlated?
?
$\text{Var}(\bar{h}) = \rho \sigma^2 + \frac{1-\rho}{m}\sigma^2$, where $\sigma^2$ is the variance of one predictor and $\rho$ is the pairwise correlation. The first term is the floor set by correlation; the second shrinks with $m$.

Why are bootstrap-trained trees correlated, and why does that matter?
?
They share data — each bootstrap sees ~63% of the original points, with high overlap. Correlated predictors don't average as well: variance has a floor at $\rho \sigma^2$ that doesn't shrink with $m$.

What is a **Random Forest**?
?
Bagging applied to **fully-grown decision trees**, plus the additional trick of choosing each split's feature from a **random subset of $k$ features** instead of all $d$.

What is the canonical default for the feature-subset size $k$ in a Random Forest?
?
$k = \sqrt{d}$. The lecture states this as the statistically good default.

Why do Random Forests use **fully-grown** (unpruned) trees?
?
Each tree is meant to have **low bias / high variance**. Pruning would reduce the variance per tree, but bagging will reduce variance for free anyway, so pruning would just add bias. Maximizing per-tree expressiveness gives the best raw material for aggregation.

Why does Random Forest's "random feature subset at each split" trick matter beyond plain bagging?
?
It **decorrelates** the trees. Without it, every bootstrap tree splits on the strongest feature at the root, making trees similar (high $\rho$), which limits variance reduction. Random features per split forces different trees to make different choices, lowering $\rho$ and pushing the variance floor down.

How should you set the number of trees $m$ in a Random Forest?
?
**As large as you can afford.** More trees never hurt — they only have diminishing returns past a few hundred. There's no overfitting risk from $m$ alone.

How are **out-of-bag (OOB) predictions** computed?
?
For each training point $x_i$, average the predictions of only those trees whose bootstrap sample didn't include $x_i$ (~37% of trees per point). The resulting OOB error is an unbiased estimate of test error — built-in validation, no separate held-out set needed.

What fraction of unique training points appear in a typical bootstrap sample?
?
About **63.2%** appear at least once; the remaining ~36.8% are out-of-bag. (As $N \to \infty$: $1 - 1/e$.)

Bagging vs boosting: which targets variance and which targets bias?
?
**Bagging targets variance** (averages many strong learners that overfit). **Boosting targets bias** (combines many weak learners that underfit).

Bagging vs boosting: parallel or sequential training?
?
Bagging is **embarrassingly parallel** — each base learner trains independently on its own bootstrap. Boosting is **sequential** — each new learner is fitted to the errors of all previous ones.

What does the "let it overfit!" instruction in Random Forest mean for the trees?
?
Don't apply pre-pruning (no max depth, no min-samples-per-leaf) and don't do post-pruning. Each tree is grown until its leaves are pure (or training data is exhausted). The forest's averaging will fix the resulting variance.

Which classical question motivates the boosting framework introduced at the end of L12?
?
M. Kearns (1988): *"Can we combine many weak learners to make a strong learner?"* R. Schapire (1990) answered yes with **AdaBoost**.

How is bagging related to the expected predictor $\bar{h}$ from L11?
?
The bagged predictor empirically **approximates** $\bar{h}(x) = \mathbb{E}_D[h_D(x)]$. As $m \to \infty$ and the bootstrap mimics fresh draws from $P$, the bagged $\bar{h}$ converges to L11's expected predictor — which by definition has zero variance against itself.
