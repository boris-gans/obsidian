---
tags: [flashcards, Statistical-Learning]
course: Statistical-Learning
lecture: 03
source: SLP-lec3(1).pdf
created: 2026-05-03
---

# Lecture 03 — Introduction to neural nets

How does the model change when going from binary to multi-class classification?
?
Replace the weight *vector* $w \in \mathbb{R}^d$ with a weight *matrix* $W \in \mathbb{R}^{C \times d}$ (one row per class), and the bias *scalar* with a bias *vector* $b \in \mathbb{R}^C$. Each row is a "neuron" specializing in one class. Score vector: $z = W x + b \in \mathbb{R}^C$.

What does each row of $W$ represent geometrically and intuitively?
?
Geometrically: a vector normal to that class's separating hyperplane, pointing toward the positive class. Intuitively: a *template* for that class — its coordinates literally resemble pixel values of typical class examples (e.g. a "lemon template").

Why can't we just use $C$ independent sigmoids for $C$-class classification?
?
Their outputs don't add up to 1 — there's no joint constraint forcing a probability distribution over classes. Each neuron decides independently, so they can't "cooperate" on mutually exclusive class predictions. We need softmax.

What's the softmax formula and what does it produce?
?
$a_c = \dfrac{e^{z_c}}{\sum_{c'} e^{z_{c'}}}$. Output is non-negative and sums to 1 — a valid probability distribution over classes.

Why is binary classification with sigmoid + log-loss equivalent to 2-class softmax + cross-entropy?
?
Two-unit softmax with one logit fixed at 0 collapses to $\sigma(z_1)$. Either choice is the same model; the sigmoid version uses fewer parameters because the second probability is automatically $1 - p$.

Write the multi-class cross-entropy loss using a one-hot label.
?
$\ell_i = -\log a_{y_i} = -\log\dfrac{e^{z_{y_i}}}{\sum_{c'} e^{z_{c'}}}$, where $y_i$ is the index of the true class. Only the correct-class probability appears.

In multi-class cross-entropy, why do we not need separate terms for the wrong-class probabilities?
?
The softmax constraint $\sum_c a_c = 1$ couples them implicitly — pushing $a_{y_i} \to 1$ pushes the rest toward 0.

Why is "Idea 1" (random search over weights) rejected as an optimization strategy?
?
You can't tell when to stop; no guarantee how many tries are enough; scales terribly with parameter count. Each random sample throws away information.

What is the gradient descent update rule?
?
$\theta^{t+1} = \theta^{t} - \eta \nabla_\theta \mathcal{L}(\theta^{t})$. Step in the direction of the negative gradient with learning rate $\eta$.

What's the geometric interpretation of the negative gradient?
?
The direction of **steepest descent** in parameter space — the locally fastest way to decrease the loss.

Why does gradient descent's step size automatically shrink near the minimum?
?
Because the gradient itself shrinks (slope flattens) near the minimum. The update size is $\eta \|\nabla \mathcal{L}\|$, and $\|\nabla \mathcal{L}\| \to 0$ at the minimum, even with fixed $\eta$.

Name three failure modes of plain gradient descent.
?
(1) Learning rate too large → overshoot or diverge. (2) Learning rate too small → very slow convergence. (3) Stuck in a poor local minimum on non-convex losses. (4) Saddle points (especially common in high dimensions).

What's the difference between full-batch GD, mini-batch GD, and stochastic GD?
?
Full-batch: uses all $N$ examples per update (true gradient, slow). Mini-batch: uses $|B|$ examples per update (fast, parallel-friendly). Stochastic / "pure" SGD: $|B| = 1$, one example per update (very noisy but very cheap).

In SGD, when does a parameter update happen — per example or per epoch? (Mock §1k trap.)
?
**Per example** (or per mini-batch). One iteration = one parameter update, **not** one full pass over the data (that's an epoch). Pure SGD makes $N$ updates per epoch.

Why can SGD's noise actually help training?
?
The noisy updates can carry the parameters out of *shallow* local minima — a deterministic full-batch step would stay stuck. Trade-off: more total iterations needed because each step is a noisy estimate.

Why is the mini-batch gradient an *unbiased* estimator of the full-batch gradient?
?
Because each example is sampled (effectively) uniformly at random, the expected mini-batch gradient equals the average over the whole dataset. Higher $|B|$ reduces variance but never changes the expectation.

How do you turn a logistic-regression neuron into a linear-regression neuron?
?
Drop the sigmoid (no activation; output is real-valued) and swap the loss from cross-entropy to mean squared error $\ell_i = (y_i - \hat{y}_i)^2$. Same $w^T x + b$ score function.

Why is MSE the standard regression loss instead of "sum of differences" $\sum(y_i - \hat{y}_i)$?
?
Sum of differences cancels out by sign — positive and negative residuals offset each other. Squaring removes the sign and is differentiable everywhere (unlike absolute value), so gradient descent works.

State the gradient of MSE with respect to $w$ for a single example $\hat{y}_i = w^T x_i + b$.
?
$\nabla_w \ell_i = -2(y_i - \hat{y}_i)\, x_i$. The residual $(y_i - \hat{y}_i)$ appears as a multiplier; GD pulls $w$ toward features whose examples have large residuals.

What's the MLE interpretation of MSE?
?
MSE is the negative log-likelihood under $y_i \sim \mathcal{N}(\hat{y}_i, \sigma^2)$ — Gaussian noise around the prediction. Minimizing MSE is MLE under that noise model.

Why are saddle points a *bigger* problem than local minima in high dimensions?
?
At a critical point, having a local minimum requires *all* curvature directions to be positive simultaneously — combinatorially unlikely as $d$ grows. Saddle points (mixed signs) become combinatorially dominant.

What three optimization tricks does L03 mention as "fixes" for plain SGD's weaknesses?
?
(1) **Momentum** — use a velocity term that accumulates past gradients (carries you across saddles, accelerates downhill). (2) **2nd-order methods** (Newton / BFGS) — use curvature to set step size. (3) **Adaptive learning rates** (AdaGrad, RMSProp, Adam) — per-parameter step size from gradient history.

Can softmax + multi-class cross-entropy achieve zero training error on XOR?
?
**No.** The model is still a *linear* classifier in $x$ (boundaries are linear regardless of how many classes). XOR needs a non-linear model — solved by adding hidden layers in L04.

What does "convex loss surface" mean for gradient descent guarantees?
?
A convex loss has only one minimum (the global one). GD with a small enough learning rate is guaranteed to converge to it. Logistic regression and linear regression with squared loss are convex; deep neural-net losses are not.
