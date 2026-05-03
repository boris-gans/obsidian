---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Adaboost(1).pdf
created: 2026-05-03
---

# AdaBoost

The original boosting algorithm — **Ada**ptive **Boost**ing (Freund & Schapire, 1995, Gödel-prize-winning). Builds a strong binary classifier by combining many [[weak-learner|weak learners]] (typically [[decision-tree|decision stumps]]) into a weighted vote. AdaBoost is the [[boosting|general boosting framework]] instantiated with [[exponential-loss|exponential loss]] — a **sibling** of [[gradient-boosting]], not a parent or child.

**Three signature features** distinguish AdaBoost from generic gradient boosting:

1. **Adaptive learning rate via closed-form line search.** Each round picks $\alpha_t = \tfrac{1}{2}\ln\tfrac{1-\epsilon_t}{\epsilon_t}$ — the exact minimizer of exponential loss along $h_t$. No fixed shrinkage hyperparameter to tune.
2. **Per-point weights**, not residuals. AdaBoost keeps the original $y_i$ and reweights training points so the next weak learner focuses on what the previous ones got wrong. Gradient boosting, by contrast, fits each new tree to a *new target* (the residual $y - H$).
3. **Binary by construction**: $y_i \in \{-1, +1\}$, $h(x) \in \{-1, +1\}$. Multi-class needs an extension (e.g., SAMME).

## The algorithm

**Initialize:** $w_i^1 = 1/N$ for $i = 1, \ldots, N$ (uniform).

**For** $t = 1, 2, \ldots, T$:

1. **Fit weak learner**: $h_t = \arg\min_{h \in \mathcal{H}} \epsilon_t(h)$ where the **weighted error** is
$$
\epsilon_t = \sum_{i: h(x_i) \ne y_i} w_i^t.
$$
2. **Stump weight** (closed-form line search):
$$
\alpha_t = \tfrac{1}{2} \ln \frac{1 - \epsilon_t}{\epsilon_t}.
$$
3. **Reweight points**:
$$
w_i^{t+1} \propto w_i^t \exp(-\alpha_t y_i h_t(x_i)),\quad \text{then renormalize}.
$$

**Output:**
$$
H(x) = \mathrm{sign}\!\Big(\sum_{t=1}^{T} \alpha_t h_t(x)\Big).
$$

## Equivalent normalized weight-update form

After plugging in $\alpha_t = \tfrac{1}{2}\ln\tfrac{1-\epsilon_t}{\epsilon_t}$ and dividing by the normalization, the weight update simplifies:

$$
w_i^{t+1} =
\begin{cases}
\dfrac{w_i^t}{2(1 - \epsilon_t)} & \text{if } h_t(x_i) = y_i\ \text{(correct)} \\[6pt]
\dfrac{w_i^t}{2 \epsilon_t} & \text{if } h_t(x_i) \ne y_i\ \text{(wrong)}
\end{cases}
$$

## The 1/2 invariant (sanity check during by-hand exams)

After every weight update, the **sum of weights on correctly-classified points = sum of weights on misclassified points = 1/2**. Each round's reweighting bisects total mass between "what I got right" and "what I got wrong." Use this as a quick arithmetic check on every round during a by-hand AdaBoost run.

## Why $\alpha_t = \tfrac{1}{2}\ln\tfrac{1-\epsilon_t}{\epsilon_t}$ — the line-search derivation

After fixing $h_t$, AdaBoost finds $\alpha_t$ by minimizing exponential loss along the line $H + \alpha h_t$:
$$
\alpha_t = \arg\min_\alpha \big[(1 - \epsilon_t) e^{-\alpha} + \epsilon_t e^{+\alpha}\big].
$$
Differentiate, set to zero: $-(1 - \epsilon_t)e^{-\alpha} + \epsilon_t e^{\alpha} = 0$ → $e^{2\alpha} = (1 - \epsilon_t)/\epsilon_t$ → the closed form above.

**Sanity check:** $\epsilon_t = 0.5 \Rightarrow \alpha_t = 0$ (worthless learner gets zero vote). $\epsilon_t \to 0 \Rightarrow \alpha_t \to \infty$ (perfect learner dominates the ensemble). $\epsilon_t > 0.5$ would give $\alpha_t < 0$ — meaning "flip the prediction" — but the L13 weak-learner setup already requires $\epsilon_t < 0.5$.

## Tie-break rule (mock-exam conventionn)

When two stumps tie on minimum weighted error, **prefer the lower-numbered stump**. This is a course-specific convention from the SLP mock; carry it onto the final.

## What loss does AdaBoost minimize?

Exponential loss:
$$
\ell(H) = \sum_{i=1}^{N} e^{-y_i H(x_i)}.
$$

The functional-gradient view (L13) shows that AdaBoost = forward-stagewise additive modeling under exponential loss. This is the unifying lens:
- **Gradient boosting (L13)** = same framework, **squared loss** → fit tree to residuals.
- **AdaBoost (L14)** = same framework, **exponential loss** → reweight points by weighted error.
- **LogitBoost** = same framework, logistic loss → similar but bounded per-point influence.

## Strengths

- **Fast convergence** on clean data — closed-form line search means each step is optimal.
- **No hyperparameter tuning** for step size — line search picks it automatically.
- **Easy to implement**: each round is one weighted-error minimization plus an arithmetic update.
- **Principled — provably equivalent** to forward-stagewise additive modeling under exponential loss.

## Weaknesses

- **Brittle on noisy / mislabeled data.** Exponential loss is unbounded — a single mislabeled point with margin $-10$ has loss $e^{10} \approx 22{,}000$, dominating the next reweight. LogitBoost addresses this.
- **Binary only** in vanilla form — multi-class needs SAMME / AdaBoost.M2.
- **No native probability output** — the score $\sum_t \alpha_t h_t(x)$ is uncalibrated.
- **Sequential** — can't parallelize across rounds (unlike bagging / Random Forest).

## Decision stumps as the canonical weak learner

A **decision stump** is a depth-1 tree: one feature, one threshold, two leaves. Why stumps:
- Trivial to enumerate and train (sweep one threshold per feature, pick the lowest-weighted-error split).
- Almost always achieves $\epsilon_t < 0.5$ on any nontrivial dataset.
- Maximal bias / minimal variance — exactly the right base learner for boosting (which reduces bias).
- Final ensemble's decision boundary is a sum of axis-aligned splits, which can approximate complex boundaries arbitrarily closely.

## Worked example skeleton (for cold exam recall)

Given: $N$ training points and a numbered list of candidate stumps.

**Round 1.**
1. $w_i^1 = 1/N$ for all $i$.
2. For each candidate stump $h_k$, compute weighted error $\epsilon = \sum_{i: h_k(x_i) \ne y_i} w_i^1$. Pick the one with minimum $\epsilon$ (tie-break: lowest number).
3. $\alpha_1 = \tfrac{1}{2}\ln\tfrac{1 - \epsilon_1}{\epsilon_1}$.
4. Update: correct points get $w \cdot \tfrac{1}{2(1 - \epsilon_1)}$; wrong points get $w \cdot \tfrac{1}{2\epsilon_1}$.
5. **Sanity check** the 1/2 invariant.

**Round 2 & 3.** Same procedure with the updated weights.

**Final ensemble**: $H(x) = \mathrm{sign}(\alpha_1 h_1(x) + \alpha_2 h_2(x) + \alpha_3 h_3(x))$.

**Training error:** count how many of the $N$ training points have $H(x_i) \ne y_i$, divide by $N$.

## Exam-relevant facts

- Initial weights are **uniform** $1/N$.
- Weighted error $\epsilon_t = \sum_{i: h_t(x_i) \ne y_i} w_i^t$.
- Stump weight $\alpha_t = \tfrac{1}{2}\ln\tfrac{1 - \epsilon_t}{\epsilon_t}$.
- Weight update (normalized closed form): correct → $w / [2(1 - \epsilon)]$; wrong → $w / [2\epsilon]$.
- Final ensemble $H(x) = \mathrm{sign}(\sum_t \alpha_t h_t(x))$.
- The **1/2 invariant**: after each update, correct group sums to 1/2, incorrect group sums to 1/2.
- Tie-break: prefer **lowest-numbered stump**.
- Loss minimized: **exponential loss** $\sum_i e^{-y_i H(x_i)}$.

## Related

- [[boosting]] — the general framework. AdaBoost is the exponential-loss instance.
- [[gradient-boosting]] — sibling algorithm under squared loss.
- [[weak-learner]] — the > 50%-accuracy primitive.
- [[decision-tree]] — the stump (depth-1) is the canonical AdaBoost base learner.
- [[exponential-loss]] — what AdaBoost implicitly minimizes.
- [[bias-variance-decomposition]] — boosting (and AdaBoost) reduce bias.
- [[lecture-14-adaboost|SLP L14]] — full derivation and worked example.
- [[lecture-13-boosting|SLP L13]] — the general framework AdaBoost specializes.
