---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: Lec-06-improving-MLPs(1).pdf
created: 2026-05-03
---

# Softmax + cross-entropy: the clean output gradient

The pairing of **softmax output** + **cross-entropy loss** + **one-hot targets** gives the cleanest possible gradient at the output layer:

$$
\frac{\partial L_i}{\partial z_k} = a_k - y_k.
$$

This identity is *why* CE is the standard classification loss. It's also why CE vs. SSE matters in practice: SSE on softmax outputs gives a much weaker, more saturated gradient that gradient descent can't act on usefully.

## The setup

Multi-class classifier with $K$ output units. Let:

- $z = W^{(L)} a^{(L-1)} + b^{(L)} \in \mathbb{R}^K$ — pre-activations.
- $a = \text{softmax}(z)$, i.e. $a_k = e^{z_k} / \sum_j e^{z_j}$ — class probabilities.
- $y \in \{0, 1\}^K$ — one-hot target with $y_t = 1$ for the true class $t$.
- $L_i = -\log a_t = -\sum_k y_k \log a_k$ — cross-entropy loss.

## The derivation (chain rule, two cases)

L06 walks through both cases by writing out the path on the computational graph.

**Case 1: $k = t$ (the true-class index).** Only $z_t$ feeds $a_t$ via the diagonal of softmax's Jacobian:

$$
\frac{\partial L_i}{\partial z_t}
= \underbrace{\frac{\partial L_i}{\partial a_t}}_{-1/a_t}
\cdot \underbrace{\frac{\partial a_t}{\partial z_t}}_{a_t(1 - a_t)}
= -\frac{1}{a_t} \cdot a_t(1 - a_t) = -(1 - a_t) = a_t - 1.
$$

**Case 2: $k \ne t$ (a wrong-class index).** Only $a_t$ enters the loss, but $a_t$ depends on $z_k$ through softmax's normalization (off-diagonal Jacobian):

$$
\frac{\partial L_i}{\partial z_k}
= \underbrace{\frac{\partial L_i}{\partial a_t}}_{-1/a_t}
\cdot \underbrace{\frac{\partial a_t}{\partial z_k}}_{-a_t a_k}
= -\frac{1}{a_t} \cdot (-a_t a_k) = a_k.
$$

**Both cases at once.** With one-hot encoding, $y_k = 1$ if $k = t$ else $0$. The two formulas collapse into

$$
\boxed{\frac{\partial L_i}{\partial z_k} = a_k - y_k.}
$$

This is the L06 punch-line: *"we can write both cases with a single expression: $a_k - y_k$ if we are using one-hot encoding"* ([[30-Sources/Statistical-Learning/pdf/Lec-06-improving-MLPs(1).pdf#page=46|slide ~46]]).

## Output bias and weight gradients

Once you have $\partial L / \partial z_k$, the output-layer parameter gradients drop out by one more chain-rule step. With $z_k = \sum_i w_{ki} x_i + b_k$ where $x$ is the input to the output layer (the previous-layer activation):

$$
\frac{\partial L_i}{\partial b_k} = \frac{\partial L_i}{\partial z_k} \cdot \underbrace{\frac{\partial z_k}{\partial b_k}}_{=1} = a_k - y_k.
$$

$$
\frac{\partial L_i}{\partial w_{ki}} = \frac{\partial L_i}{\partial z_k} \cdot \underbrace{\frac{\partial z_k}{\partial w_{ki}}}_{=x_i} = (a_k - y_k)\, x_i.
$$

L06's table format ([[30-Sources/Statistical-Learning/pdf/Lec-06-improving-MLPs(1).pdf#page=48|slides ~46–48]]):

| | bias $b_k$ | weight $w_{ki}$ |
| --- | --- | --- |
| $k = t$ (true class) | $a_k - 1$ | $(a_k - 1)\, x_i$ |
| $k \ne t$ (wrong class) | $a_k$ | $a_k\, x_i$ |
| both cases (one-hot) | $a_k - y_k$ | $(a_k - y_k)\, x_i$ |

For hidden layers, replace $x_i$ with the previous layer's activation $a_i^{(L-1)}$. The pattern continues all the way back through backprop.

## Why this matters: CE beats SSE on the output

L06's other thread: **why use CE, not SSE, on softmax outputs?** SSE on $a_t$ (the true-class probability) gives a derivative

$$
\frac{\partial}{\partial a_t}\; \tfrac{1}{2}(a_t - 1)^2 = a_t - 1,
$$

bounded in $[-1, 0]$. CE gives

$$
\frac{\partial}{\partial a_t}\; (-\log a_t) = -\frac{1}{a_t},
$$

which **explodes** as $a_t \to 0$. Practical consequence: when the model is *very wrong* (true class probability near zero), CE produces a huge gradient and SGD takes a big corrective step. SSE produces a derivative of magnitude $\le 1$ — barely a step at all on a confident-but-wrong prediction. The L06 numerical example: $a_t = 0.353$, SSE gives $|\partial L / \partial a_t| \approx 0.65$, CE gives $\approx 2.83$. **The CE gradient is sharper exactly when correction is needed most.**

Combine that with the cancellation we just derived ($a_k - y_k$ at the pre-activation level) and CE+softmax is the obviously-right choice for classification.

## Exam-relevant facts

- Output-layer gradient under softmax + CE + one-hot: $\partial L / \partial z_k = a_k - y_k$. **Memorize.**
- Bias gradient at the output: $a_k - y_k$. Weight gradient: $(a_k - y_k) \cdot \text{input}$.
- For hidden layers, the input is the previous-layer activation.
- The cancellation between softmax's Jacobian and CE's $-1/a_t$ is **why this form is so clean** — neither softmax + SSE nor sigmoid + CE on multi-class gives this.
- CE > SSE for classification because CE's derivative explodes at bad predictions; SSE's stays bounded.

## Related

- [[cross-entropy]] — the loss function itself.
- [[softmax]] — the output activation.
- [[backpropagation]] — uses this gradient as the seed at the output layer.
- [[activation-function]] — output-vs-hidden distinction.
- [[lecture-06-improving-mlps|SLP L06]] — source of the derivation.
