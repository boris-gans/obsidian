---
tags: [flashcards, Statistical-Learning]
course: Statistical-Learning
lecture: 06
created: 2026-05-03
---

# Lecture 06 — Improving MLPs: flashcards

Why does cross-entropy beat squared error as a classification loss?
?
At a "terrible" prediction (true-class probability $p \to 0$), CE's derivative $|{-1/p}|$ explodes — gradient descent gets a strong correction signal. SSE's derivative $|p - 1|$ stays bounded by 1, so confident-but-wrong predictions barely move. CE's gradient is sharpest exactly when correction is needed most.

What is the output-layer gradient of cross-entropy w.r.t. the pre-activation $z_k$ under softmax + one-hot targets?
?
$\partial L_i / \partial z_k = a_k - y_k$ — predicted probability minus target probability. Comes from cancellation between softmax's Jacobian ($a_k(1-a_k)$ on the diagonal, $-a_t a_k$ off-diagonal) and CE's $-1/a_t$.

What is $\partial L_i / \partial b_k$ at the output of a softmax + CE classifier?
?
$a_k - y_k$. Same as the pre-activation gradient because $\partial z_k / \partial b_k = 1$.

What is $\partial L_i / \partial w_{ki}$ at the output of a softmax + CE classifier?
?
$(a_k - y_k)\, x_i$, where $x_i$ is the $i$-th input to the output layer (the previous-layer activation in a deep net).

What is the maximum value of the sigmoid derivative, and where does it occur?
?
$\sigma'(z) = \sigma(z)(1 - \sigma(z))$ has maximum $0.25$ at $z = 0$, and decays toward zero as $|z|$ grows. This is the per-layer shrinkage factor that makes deep sigmoid networks suffer from vanishing gradients.

Why do gradients vanish in a deep MLP with sigmoid activations?
?
Backprop multiplies one $\sigma'(z) \le 0.25$ factor per layer. Across $L$ layers the chain-rule product is bounded above by $0.25^L$ — for $L = 10$ that's already $\approx 10^{-6}$. Early-layer weights receive negligible gradient and don't train.

Why does ReLU break the per-layer shrinkage problem?
?
$\text{ReLU}'(z) = 1$ for $z > 0$, so on the active side ReLU contributes no shrinkage factor to the chain-rule product. Combined with He initialization keeping pre-activations near zero, a typical unit fires for ~half its inputs.

What is the dead-ReLU problem?
?
A ReLU unit whose pre-activation $z = w^\top x + b \le 0$ for every training example outputs zero on every example, has local gradient zero on every example, never receives a nonzero weight update, and stays dead forever. Caused by bad init or one too-large gradient step.

How do Leaky ReLU and Parametric ReLU avoid dead units?
?
They use a small positive slope $\alpha > 0$ on the negative side instead of zero: $\text{LeakyReLU}(z) = \alpha z$ for $z < 0$. Leaky uses $\alpha = 0.01$; Parametric (PReLU) makes $\alpha$ a learned parameter. Either way, a unit with negative pre-activation still passes a small gradient and can recover.

What is the formula for Xavier/Glorot weight initialization, and which activation is it designed for?
?
$W_{ij} \sim \mathcal{N}(0, 1/D_\text{in})$, designed for $\tanh$. Goal: preserve forward-pass activation variance across layers.

What is the formula for He/Kaiming weight initialization, and which activation is it designed for?
?
$W_{ij} \sim \mathcal{N}(0, 2/D_\text{in})$, designed for ReLU. The factor of 2 compensates for ReLU outputting zero on half the inputs in expectation — doubling input variance restores layer-to-layer balance.

Why does naive initialization like `W = 0.01 * randn(...)` fail in deep networks?
?
With $\sigma^2 = 0.0001$, the forward-pass variance shrinks by a factor of $D_\text{in} \cdot 0.0001$ per layer; after a few layers all activations are crammed near zero. Sigmoid units output ~0.5 everywhere, ReLU units sit at zero — the network can't represent anything useful.

What does dropout do during training?
?
Randomly zero out each hidden unit with probability $p$ on each forward pass; backprop only through the surviving units. A different mask per example, per layer, per pass. Standard $p = 0.5$ for fully-connected hidden layers.

What is "inverted dropout" and why is it used?
?
Multiply training-time activations by $1/(1-p)$ after the dropout mask, so test-time activations need no rescaling. Pushes the magnitude correction into training, leaving the test-time forward pass as plain forward propagation.

What is the "ensemble interpretation" of dropout?
?
Each dropout mask defines a subnetwork that shares weights with all other masks. Training with dropout is approximately training an exponentially large ensemble of weight-shared subnetworks; test-time evaluation with dropout off averages this ensemble.

In a deep network with sigmoid activations, what happens to the gradient of an early-layer weight?
?
It vanishes exponentially in depth. Each layer multiplies the chain-rule product by a $\sigma'(z) \le 0.25$ factor, so for $L$ layers the gradient shrinks at least as fast as $0.25^L$.

Why are output activation choice and loss tightly paired?
?
The chain rule simplifies cleanly only for matched pairs: sigmoid + binary CE → gradient $a - y$; softmax + categorical CE → gradient $a_k - y_k$; identity + squared error → gradient $a - y$. Mismatched pairs (e.g. softmax + SSE) give weaker, more saturated gradients.

What happens to gradient descent if the learning rate $\eta$ is too large?
?
Updates overshoot — the loss can oscillate or diverge instead of decreasing. Symptoms: training loss jumps around or grows, weights blow up. Fix: smaller $\eta$, or adaptive learning-rate schedules.

What's the difference between "saturation" and "the dead-ReLU problem"?
?
Saturation: a sigmoid/tanh unit with large $|z|$ has near-zero local gradient — it can recover if $z$ moves back to the active region (because the gradient is small, not exactly zero). Dead ReLU: a unit with $z \le 0$ on every example has local gradient *exactly* zero, so it cannot recover.

Why is L06 not directly tested in the past mock exam?
?
The mock tests applied skills (decision-tree compute, AdaBoost, PCA, k-means) — L06 is conceptual machinery (activation choice, init, dropout) that *enables* training but isn't a "compute by hand" topic. The prof flagged "more MCQs likely" for the actual final, where L06 is exactly the kind of conceptual material that drops into MCQs.

What is the empirical convergence-speed gain of ReLU vs. sigmoid/tanh?
?
About 6× faster in practice (per L06's slide note). The combination of no positive-side saturation + cheap arithmetic (max instead of exp) lets ReLU networks train much faster on the same data.
