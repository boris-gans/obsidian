---
tags: [flashcards, Statistical-Learning]
---

# Lecture 14 — AdaBoost

What does AdaBoost stand for, and who invented it?
?
**Ada**ptive **Boost**ing — **Freund & Schapire, 1995** (winners of the 2003 Gödel prize).

Where does AdaBoost sit relative to gradient boosting (L13)?
?
**Sibling**, not parent or child. Both are instances of the general boosting framework (functional gradient descent). Gradient boosting uses **squared loss** → fit tree to residuals. AdaBoost uses **exponential loss** → reweight points by weighted error.

What loss does AdaBoost (implicitly) minimize?
?
**Exponential loss**: $\ell(H) = \sum_{i=1}^N e^{-y_i H(x_i)}$.

What are the labels and weak-learner outputs in AdaBoost?
?
Both **binary**: $y_i \in \{-1, +1\}$ and $h(x) \in \{-1, +1\}$. The whole derivation requires this; multi-class needs a separate extension (e.g. SAMME).

What are the **initial weights** in AdaBoost?
?
**Uniform**: $w_i^1 = 1/N$ for all $i = 1, \ldots, N$.

What does the algorithm do at iteration $t$ (3 steps)?
?
1. Pick the weak learner $h_t \in \mathcal{H}$ that **minimizes the weighted error** $\epsilon_t = \sum_{i: h_t(x_i) \ne y_i} w_i^t$.
2. Compute the stump weight $\alpha_t = \tfrac{1}{2} \ln \tfrac{1 - \epsilon_t}{\epsilon_t}$.
3. Reweight points: $w_i^{t+1} \propto w_i^t \exp(-\alpha_t y_i h_t(x_i))$, then renormalize.

Write the **weighted error** formula.
?
$\epsilon_t = \sum_{i: h_t(x_i) \ne y_i} w_i^t = \sum_{i=1}^N w_i^t \mathbb{1}[h_t(x_i) \ne y_i]$.

Write the **closed-form formula** for the AdaBoost stump weight $\alpha_t$.
?
$\alpha_t = \tfrac{1}{2} \ln \dfrac{1 - \epsilon_t}{\epsilon_t}$.

Write the AdaBoost **weight update** (proportional form).
?
$w_i^{t+1} \propto w_i^t \exp(-\alpha_t\, y_i\, h_t(x_i))$, then renormalize so $\sum_i w_i^{t+1} = 1$.

Write the AdaBoost weight update in **normalized closed form** (after dividing by the partition).
?
- Correct $(h_t(x_i) = y_i)$: $w_i^{t+1} = \dfrac{w_i^t}{2(1 - \epsilon_t)}$.
- Incorrect $(h_t(x_i) \ne y_i)$: $w_i^{t+1} = \dfrac{w_i^t}{2\,\epsilon_t}$.

What is the **1/2 invariant** of AdaBoost weight updates?
?
After every update, the **sum of weights on correctly-classified points = 1/2** and the **sum of weights on misclassified points = 1/2**. Each round bisects the total mass between "what the previous stump got right" and "what it got wrong." Use this as a sanity check during by-hand exam runs.

What is the AdaBoost **final classifier**?
?
$H(x) = \mathrm{sign}\!\big(\sum_{t=1}^T \alpha_t\, h_t(x)\big)$ — a sign of the weighted vote of all weak learners.

What is the **tie-break rule** for AdaBoost on the SLP exam?
?
**Prefer the lowest-numbered stump.** If two candidate stumps achieve the same minimum weighted error, pick the one with the smaller index.

Why is the AdaBoost step size called an "adaptive learning rate"?
?
Because $\alpha_t = \tfrac{1}{2}\ln\tfrac{1-\epsilon_t}{\epsilon_t}$ is the **closed-form line-search optimum** of exponential loss along $h_t$ — not a fixed shrinkage hyperparameter. Each stump's vote weight depends adaptively on how well it performed.

What happens to $\alpha_t$ when $\epsilon_t = 0.5$? When $\epsilon_t \to 0$?
?
$\epsilon_t = 0.5 \Rightarrow \alpha_t = 0$ (worthless learner gets zero vote — algorithm terminates). $\epsilon_t \to 0 \Rightarrow \alpha_t \to \infty$ (perfect learner dominates).

You start AdaBoost with $N = 10$ uniform weights. The first stump misclassifies 2 points. What are $\epsilon_1$ and $\alpha_1$?
?
$\epsilon_1 = 2 \cdot 0.1 = 0.2$. $\alpha_1 = \tfrac{1}{2}\ln(0.8/0.2) = \tfrac{1}{2}\ln 4 \approx 0.69$.

Same setup ($N = 10$, $\epsilon_1 = 0.2$, $\alpha_1 \approx 0.69$). What are the new weights for correct vs incorrect points after round 1?
?
- Correct (8 points): $w^2 = 0.1 / (2 \cdot 0.8) = 0.0625$.
- Incorrect (2 points): $w^2 = 0.1 / (2 \cdot 0.2) = 0.25$.
Sanity check: $8 \cdot 0.0625 + 2 \cdot 0.25 = 0.5 + 0.5 = 1$ ✓ (1/2 invariant).

After round 1 ($N = 10$, weights as above), the second stump achieves weighted error $\epsilon_2 \approx 0.19$. What is $\alpha_2$?
?
$\alpha_2 = \tfrac{1}{2}\ln(0.81/0.19) \approx \tfrac{1}{2}\ln 4.26 \approx 0.73$.

Why is a **decision stump** (depth-1 tree) the canonical AdaBoost weak learner?
?
- Trivial to enumerate (sweep one threshold per feature).
- Almost always achieves $\epsilon < 0.5$ on any nontrivial dataset.
- High bias / low variance — exactly what boosting wants.
- Final ensemble's boundary is a sum of axis-aligned splits — can approximate complex boundaries.

When AdaBoost is asked "boosted (stump) trees grown fully?" on a T/F (mock §1l), what's the answer?
?
For **stumps** (the canonical AdaBoost case): yes — they're already depth-1, no pruning needed, no concept of "growing fully." For deeper boosted trees the answer can flip — read the question carefully.

What is AdaBoost's main weakness?
?
**Brittleness to label noise.** Exponential loss is unbounded — a single mislabeled point with margin $-10$ has loss $e^{10} \approx 22{,}000$, dominating the next reweight. **LogitBoost** (logistic loss) addresses this with bounded per-point influence.

How does the AdaBoost ensemble's decision boundary relate to its base stumps?
?
$H(x) = \mathrm{sign}(\sum_t \alpha_t h_t(x))$ is a **weighted linear combination of stumps**. Even though each stump is just one axis-aligned split, the combined boundary can be far more complex than any individual stump — as in mock §1h.

What distinguishes AdaBoost reweighting from gradient-boosting residual fitting?
?
**Gradient boosting**: each new tree fits a *new target* (the residual $y - H$); the original $y_i$ is dropped after round 1. **AdaBoost**: each new stump fits the *original* $y_i$, but with **per-point weights** that grow on misclassified examples and shrink on correct ones.

Where does AdaBoost terminate?
?
When no $h \in \mathcal{H}$ achieves $\epsilon_t < 0.5$ (no useful direction; $\alpha_t \le 0$). In practice, stop also when training error hits zero or after $T$ rounds.

What three quantities should you compute on a §5-style by-hand AdaBoost run, in order?
?
For each round: (1) weighted error $\epsilon_t$ (sum the weights of misclassified points); (2) stump weight $\alpha_t = \tfrac{1}{2}\ln\tfrac{1-\epsilon_t}{\epsilon_t}$; (3) updated weights — correct: $w / [2(1 - \epsilon_t)]$, incorrect: $w / [2\epsilon_t]$. Then verify the 1/2 invariant.
