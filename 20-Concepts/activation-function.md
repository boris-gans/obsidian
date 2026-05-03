---
tags: [concept]
courses: [NLP, Statistical-Learning]
sources:
  - course: NLP
    file: Session 03 - Fundamental Concepts.pdf
  - course: NLP
    file: Session 16 - Neural Networks-1.pdf
  - course: Statistical-Learning
    file: SLP-04(1).pdf
  - course: Statistical-Learning
    file: Lec-06-improving-MLPs(1).pdf
created: 2026-05-02
---

# Activation function

The non-linear function applied to the weighted sum in a [[perceptron|neuron]]: $y = f(\mathbf{w}\!\cdot\!\mathbf{x} + b)$. Without it, stacked layers collapse to a single affine map — non-linearity is what makes deep networks expressive.

## Common choices

| Function | Formula | Notes |
|---|---|---|
| **Heaviside (step)** | $H(z) = \begin{cases}1 & z\geq0 \\ 0 & z<0\end{cases}$ | Original McCulloh-Pitt; non-differentiable |
| **Sigmoid** | $\sigma(z) = \dfrac{1}{1+e^{-z}}$ | Smooth, differentiable; saturates at extremes; output $\in (0,1)$ |
| **Hyperbolic tangent** | $\tanh(z)$ | Zero-centered output $\in (-1,1)$ |
| **ReLU** | $f(z) = \max(0, z)$ | Modern default; large gradients when active |
| **Leaky ReLU** | $f(z) = z$ if $z>0$, else $0.02z$ | Avoids dead-unit problem |
| **Parametric ReLU** | $f(z) = z$ if $z>0$, else $a z$ | Learned slope $a$ |

## Why ReLU & friends

Smooth non-linearities like sigmoid **saturate** for large $|z|$, killing gradients. ReLU and its variants ensure the **derivative remains large whenever the unit is active**, which keeps gradients flowing during backpropagation through deep networks.

## Why activations are necessary at all (Session 16, [[30-Sources/NLP/pdf/Session 16 - Neural Networks-1.pdf#page=9|slide 9]])

Without a nonlinear activation, a stack of layers $h = W_1 x$, $y = W_2 h$ collapses to $y = (W_2 W_1) x$ — a single linear transformation. **Stacking layers adds no expressive power** without nonlinearity. The nonlinear activation is what allows neural networks to **represent complex relationships** between inputs and outputs.

## In NLP

- Sigmoid is the activation in [[logistic-regression]] — the perceptron form $z = wx+b$ followed by $\sigma$ is exactly logistic regression.
- Softmax (a multi-output generalization of sigmoid) is the standard output activation for classifiers and for [[attention|attention weights]].
- $\tanh$ appears in the [[rnn-recurrence|RNN update]] $h_t = \tanh(W h_{t-1} + U x_t + b)$.

## In Statistical Learning (L04): hidden vs. output activations

[[lecture-04-mlps|SLP L04]] makes the **hidden vs. output** distinction explicit:

- **Output activation** depends on the task: sigmoid for binary classification, softmax for multi-class, identity (none) for regression.
- **Hidden activation** must be non-linear *and* component-wise. Softmax is **wrong** for hidden layers ([[30-Sources/Statistical-Learning/pdf/SLP-04(1).pdf#page=27|slides ~26–29]]) because it couples outputs through normalization — that destroys the rich multi-feature representation hidden layers need to learn. Sigmoid, $\tanh$, and ReLU are all valid; ReLU is the modern default for "cheap to compute, good gradient properties."

The L04 slogan: **non-linearity introduces more powerful representations** ([[30-Sources/Statistical-Learning/pdf/SLP-04(1).pdf#page=82|slide ~82]]). Without it, $W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)$ — a single linear layer. Activation choice is *what makes depth work.*

## In Statistical Learning (L06): saturation kills gradients in deep nets

[[lecture-06-improving-mlps|SLP L06]] makes the cost of choosing wrong activations precise. The chain rule multiplies a local gradient at every layer; if those local gradients are small, the product collapses to zero exponentially in depth — **vanishing gradients**.

**Sigmoid is the worst.** $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ has max $0.25$ at $z = 0$ and decays toward zero in both directions. Stack 10 sigmoid layers: best-case gradient attenuation is $0.25^{10} \approx 10^{-6}$, before any weight matrix shrinkage. A network that deep simply will not train.

**$\tanh$ is mildly better.** $\tanh'(z) = 1 - \tanh^2(z)$ peaks at $1$ — no built-in shrinkage factor at $z = 0$ — but still saturates symmetrically for $|z| \gg 1$. Better than sigmoid for hidden layers; still not the modern choice.

**ReLU is the structural fix on the active side.** $\text{ReLU}'(z) = 1$ for $z > 0$, so the chain-rule product collects no shrinkage factor when the unit is firing. The trade-off: ReLU has its own pathology, the [[relu#The dead-ReLU problem L06|dead-unit problem]], where a unit with $z \le 0$ on every example has gradient $0$ forever.

**Empirical convergence speed.** L06 cites *6× faster than sigmoid/tanh* for ReLU on standard benchmarks ([[30-Sources/Statistical-Learning/pdf/Lec-06-improving-MLPs(1).pdf#page=65|slide ~65]]) — the combination of no positive-side saturation + cheap arithmetic.

## In Statistical Learning (L06): output activations and the softmax + CE pairing

L06 also closes the loop on **output activations and their loss pairing**:

| Task | Output activation | Loss | Reason |
| --- | --- | --- | --- |
| Regression | identity (none) | squared error | direct continuous output |
| Binary classification | sigmoid | binary cross-entropy | $a \in (0,1)$ as a probability; CE gives sharp gradient at bad predictions |
| Multi-class | softmax | categorical cross-entropy | $\sum_k a_k = 1$ as a categorical distribution; same CE-gradient argument applies |

The output-layer gradient simplifies to $\partial L / \partial z_k = a_k - y_k$ in the softmax + CE case (one-hot $y$). See [[softmax-cross-entropy-gradient]] for the full derivation.
