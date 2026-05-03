---
tags: [flashcards, Statistical-Learning]
---

# Lecture 13 — Boosting and gradient-boosted trees

What is **boosting** in one sentence?
?
A sequential ensemble method that combines many **weak learners** (high-bias / low-variance predictors that underfit) into a strong predictor by **reducing bias**. Mirror image of bagging.

Bagging vs boosting: which targets variance, which targets bias?
?
**Bagging targets variance** (averages many strong learners that overfit). **Boosting targets bias** (sequentially combines many weak learners that underfit).

What does the slogan "boosting is gradient descent in function space" mean?
?
At each iteration, train a weak learner whose vector $\vec{h}$ in label space $\mathbb{R}^N$ points in (approximately) the negative gradient direction of the loss; take a small step that way. The "function" being optimized over is the predictor itself.

What is the form of the final boosting predictor?
?
$H(x) = \sum_{t=1}^{T} \alpha_t\, h_t(x)$ — a weighted sum of $T$ weak learners, where $\alpha_t$ is the step size at iteration $t$.

What is a **weak learner**?
?
A predictor with accuracy **strictly greater than 50%** on binary classification — any edge over random guessing. Even 50.1% is enough; boosting compensates with iteration count.

What's the unique pathological case for a weak learner?
?
**Exactly 50% accuracy** = the weak-learner vector is perfectly orthogonal to the residual direction. No progress possible — boosting terminates.

If a candidate weak learner has accuracy below 50%, what's the fix?
?
**Flip its sign** (predict the opposite). The flipped predictor has accuracy $1 - \text{err} > 50\%$ and becomes a valid weak learner. Doesn't help only in the orthogonal (exact 50%) case.

What's the canonical weak learner for AdaBoost? For gradient boosting?
?
**AdaBoost: decision stumps** (depth-1 trees — one feature test, two leaves). **Gradient boosting: shallow regression trees** (depth 3–8). Both are intentionally low-capacity.

Why does each step in boosting move us closer to the target $\vec{y}$ in label space?
?
By Pythagoras: if the weak learner's vector $\vec{h}$ has angle $< 90°$ with the residual $\vec{y} - \vec{H}$, then a small enough step in $\vec{h}$'s direction strictly decreases the distance to $\vec{y}$ (provided we don't overshoot). Any weak learner with > 50% accuracy satisfies the angle condition.

Write the negative gradient of squared loss with respect to $H(x_i)$.
?
$-\partial \ell / \partial H(x_i) = y_i - H(x_i)$ — the **residual** at point $x_i$.

What does "fit a regression tree to the residuals" mean in gradient boosting?
?
At iteration $t$, compute residuals $t_i = y_i - H_{t-1}(x_i)$, then train a CART regression tree minimizing $\sum_i (h(x_i) - t_i)^2$. The tree is the next weak learner; add $\alpha \cdot h_t$ to the ensemble.

Why is fitting a regression tree to residuals equivalent to functional gradient descent on squared loss?
?
The general boosting subproblem $\arg\min_h \langle \partial \ell / \partial H, h \rangle$ for squared loss becomes (after sign-flipping, completing the square, and a unit-norm constraint on $h$) $\arg\min_h \sum_i (h(x_i) - t_i)^2$ — exactly the CART regression-tree objective.

What plays the role of "learning rate" in boosting?
?
The step size $\alpha_t$ (or shrinkage parameter $\eta$). Smaller $\alpha$ → more iterations needed but more accurate / better generalizing final ensemble.

Are AdaBoost and gradient boosting parent-child or sibling algorithms?
?
**Siblings.** Both descend from the general functional-gradient-descent boosting framework. AdaBoost = exponential loss; gradient boosting = squared loss (or other differentiable losses). Neither subsumes the other.

Why can boosting **overfit** if $T$ is too large?
?
The ensemble keeps fitting to residuals; eventually those residuals are mostly noise and additional iterations memorize the noise. Validation error is **U-shaped** in $T$ — the §1g answer: iteration count is a regularization knob analogous to $\lambda$.

In bagging, base trees are grown fully. In boosting, what depth do you want?
?
**Shallow** — typically depth 1 (stumps) for AdaBoost or depth 3–6 for gradient boosting. Each base learner should have **high bias / low variance**; the ensemble's additive structure handles bias reduction.

What's the §1h exam answer about gradient boosting and stumps?
?
"Gradient-boosted = linear (additive) combination of stumps" → **true.** $H(x) = \sum_t \alpha_t h_t(x)$ is literally a linear combination. With stumps as the $h_t$, the ensemble's decision boundary is far more complex than any single stump.

How does boosting reduce bias but bagging cannot?
?
Bagging averages predictions: $\bar{h} = \frac{1}{m}\sum h_j$ — averaging biased predictors preserves the same bias. Boosting accumulates predictions: $H = \sum_t \alpha_t h_t$ where each new $h_t$ is fit to **what previous learners missed** (the residuals). The additive structure lets each new term *reduce* bias rather than just average it out.

What constraint does the lecture impose on $h$ during the gradient-boosting derivation, and why?
?
$\sum_i h(x_i)^2 = 1$ (or any constant $C$) — i.e., $h$ lives on a sphere of fixed radius in $\mathbb{R}^N$. Without this constraint, $h$ could be made arbitrarily large to drive the inner product $\langle \partial \ell / \partial H, h\rangle$ to $-\infty$ trivially.

What modifications turn gradient boosting into AdaBoost?
?
Use **exponential loss** $\ell = \sum_i e^{-y_i H(x_i)}$ instead of squared loss. The negative-gradient at point $i$ becomes $y_i e^{-y_i H(x_i)}$ — the residual is replaced by a per-point weight that grows exponentially with mis-prediction. Training a stump to minimize the weighted classification error gives the AdaBoost update with $\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$ (L14).

What's the historical timeline of boosting?
?
1988 — Kearns: can weak learners be combined to make a strong learner? 1990 — Schapire: yes (first boosting algorithm). 1995 — Freund & Schapire: AdaBoost. 2001 — Friedman: gradient boosting framework. ~2014 — XGBoost / LightGBM / CatBoost.
