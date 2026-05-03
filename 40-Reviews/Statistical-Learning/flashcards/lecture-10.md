---
tags: [flashcards, Statistical-Learning]
---

# Lecture 10 — Loss functions and regularization

What is the master form of a regularized supervised-learning objective?
?
$\min_w \frac{1}{N}\sum_i \ell(h_w(x_i), y_i) + \lambda \, \Omega(w)$ — data loss plus $\lambda$ times a regularizer on the parameters.

What is the role of the regularizer $\Omega(w)$ in supervised learning?
?
When multiple $w$ minimize the data loss equally, $\Omega(w)$ breaks the tie — pushes toward a particular kind of solution (small / sparse / smooth). $\lambda$ controls the trade-off.

For SVMs, what was the "first concrete regularizer" the SLP course introduced?
?
The soft-margin parameter $C$. The SVM objective $\frac{1}{2}\|w\|^2 + C\sum_i \xi_i$ is $L_2$-regularized hinge loss with $C$ playing the role of $1/\lambda$.

State the four canonical classification losses, all as functions of the margin $z = y\,h(x)$.
?
0/1: $\mathbb{1}[z \le 0]$. Hinge: $\max(1-z, 0)$. Log-loss: $\log(1 + e^{-z})$. Exponential: $e^{-z}$.

Why can't we directly optimize the 0/1 loss for classification?
?
It's discontinuous and piecewise constant — gives no useful gradient. Convex surrogates (hinge, log, exponential) are used instead.

Which classification loss is implicitly minimized by AdaBoost?
?
**Exponential loss**, $e^{-z}$.

Which classification loss gives **well-calibrated probabilities**?
?
**Log-loss** (logistic loss). Its $\hat{p} = \sigma(z)$ converges toward $P(y=+1 \mid x)$. Hinge and exponential give margin scores, not calibrated probabilities.

Why is exponential loss problematic on noisy data?
?
The loss grows **exponentially** in the magnitude of a mis-prediction, so a single mislabeled example can dominate the optimizer.

State the four canonical regression losses, as functions of the residual $r = h(x) - y$.
?
Squared: $r^2$. Absolute: $\|r\|$. Huber ($\delta$): $\frac{1}{2}r^2$ if $\|r\| < \delta$, else $\delta(\|r\| - \delta/2)$. Log-cosh: $\log(\cosh(r))$.

Squared loss vs absolute loss: what statistic of $y \mid x$ does each estimate?
?
Squared loss → conditional **mean**. Absolute loss → conditional **median**. The constant minimizing $\sum (y_i - c)^2$ is the mean; for $\sum \|y_i - c\|$ it's the median.

What does Huber loss combine, and why?
?
Squared loss near zero (smooth gradient near the optimum) plus absolute loss far from zero (bounded influence of outliers). "Best of both worlds" / "smooth absolute loss."

What's the difference between the **penalty form** and the **constraint form** of regularization?
?
Penalty: $\min \mathcal{L}(w) + \lambda \Omega(w)$. Constraint: $\min \mathcal{L}(w)$ s.t. $\Omega(w) \le B$. Equivalent — for every $\lambda$ there is a $B$ giving the same optimum, related **inversely** (large $\lambda$ ↔ small $B$).

Why does $L_1$ regularization induce **sparse** solutions but $L_2$ does not?
?
The $L_1$ ball ($\sum \|w_j\| \le B$) is a polytope with corners on the coordinate axes; loss contours typically touch a corner where some $w_j = 0$. The $L_2$ ball is smooth (a disk), so contours touch at non-axis points where all components are small but nonzero.

Is the §1e statement *"L2 regularization leads to sparse models"* true or false?
?
**False.** $L_2$ shrinks coefficients toward zero but does not zero them out exactly. Sparsity is an $L_1$-only property. The trap is the conclusion at the tail of the sentence.

Is **Lasso** strictly convex?
?
**No.** Lasso (squared loss + $L_1$) is convex but **not strictly** convex — the $L_1$ Hessian is zero almost everywhere, so correlated/duplicated features create non-unique optima.

Is **Elastic Net** strictly convex (and why)?
?
**Yes** when $\alpha < 1$. Elastic Net = $\alpha \|w\|_1 + (1-\alpha)\|w\|_2^2$. The $L_2$ component has Hessian $\propto I$ (positive definite, strictly convex); adding the weakly-convex $L_1$ piece can't undo that. So unique optimum, even with correlated features.

What's the closed-form solution for **Ridge regression**?
?
$w^* = (XX^\top + \lambda I)^{-1} X y^\top$. The $\lambda I$ term is the only difference from OLS and makes the matrix invertible even with correlated features.

What's the closed-form solution for **OLS** (Ordinary Least Squares)?
?
$w^* = (XX^\top)^{-1} X y^\top$. Squared loss, no regularization.

What's the relationship between Lasso, Ridge, and Elastic Net?
?
Lasso = squared loss + $L_1$. Ridge = squared loss + $L_2$. Elastic Net = squared loss + both, with $\Omega(w) = \alpha\|w\|_1 + (1-\alpha)\|w\|_2^2, \alpha \in [0, 1)$.

Sketch the train and validation error curves as functions of $\lambda$.
?
Train error: monotonically **rises** with $\lambda$ (more regularization → worse data fit). Validation error: **U-shaped** — high at small $\lambda$ (overfit), high at large $\lambda$ (underfit), minimum at the sweet spot in between.

What's the analog between iteration count $M$ and regularization strength $\lambda$?
?
They play the same role inversely: small $M$ underfits (like large $\lambda$), large $M$ overfits (like small $\lambda$). Validation curves are U-shaped against either. Early stopping picks $M^*$ without an explicit $\lambda$.

Why is squared hinge loss differentiable everywhere, but standard hinge loss is not?
?
Standard hinge $\max(1-z, 0)$ has a kink at $z=1$. Squared hinge $\max(1-z, 0)^2$ has slope zero at $z=1$ on both sides — the corner is rounded out, so it's $C^1$ smooth.

What does it mean that very large $B$ (constraint form) effectively disables regularization?
?
The constraint $\Omega(w) \le B$ becomes non-binding — the unconstrained optimum already lies inside the feasible set, so regularization has no effect. Equivalent to setting $\lambda$ very small.

What is "weight decay" and how does it relate to $L_2$ regularization?
?
Weight decay = the $-\eta \cdot 2\lambda w$ contribution to each gradient step from $\lambda \|w\|_2^2$. Each step shrinks $w$ by a constant fraction toward zero before the data gradient pulls it back. **Weight decay = $L_2$ regularization**, viewed as an SGD update rule.
