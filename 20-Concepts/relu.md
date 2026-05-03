---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-04(1).pdf
  - course: Statistical-Learning
    file: Lec-06-improving-MLPs(1).pdf
created: 2026-05-03
---

# ReLU (rectified linear unit)

The default hidden-layer activation in modern neural networks. Cheap to compute, has good gradient properties on positive inputs, and produces sparse activations (lots of exact zeros). L04 introduces it as the practical answer to "what activation should I use for hidden layers?"; L06 deepens the story with the **dead-unit failure mode** and motivates the variants.

## Definition

$$
\mathrm{ReLU}(z) = \max(0, z) = \begin{cases} z & z > 0 \\ 0 & z \le 0. \end{cases}
$$

Derivative:

$$
\frac{d}{dz}\mathrm{ReLU}(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0. \end{cases}
$$

(At $z = 0$ the derivative is undefined; in practice we set it to either $0$ or $1$ — implementations differ but it doesn't matter much.)

## Why it's the default

[[lecture-04-mlps|SLP L04]] characterizes ReLU as *"cheap to compute, good gradient properties"* ([[30-Sources/Statistical-Learning/pdf/SLP-04(1).pdf#page=85|slide ~85]]). Three reasons:

1. **Computationally trivial** — a `max` and a comparison; orders of magnitude cheaper than the exponentials inside sigmoid or $\tanh$.
2. **Non-saturating on the positive side** — for $z > 0$, the derivative is exactly $1$. No "vanishing gradient" the way sigmoid suffers when $|z|$ is large. (Detailed analysis in L06.)
3. **Sparse activations** — many units output exact zero on a given input. This makes the representation high-dimensional but "uses" only a fraction of the dimensions per example, which empirically helps generalization and speed.

## Drawbacks

- **Dead-unit problem** (see below — fully analyzed in L06).
- **Not zero-centered.** Outputs are $\geq 0$, so the gradient flow has a positive bias that can slow learning. ($\tanh$ doesn't have this issue.)
- **Unbounded above** — with bad initialization, activations can blow up.

## The dead-ReLU problem (L06)

[[lecture-06-improving-mlps|SLP L06]] makes this concrete ([[30-Sources/Statistical-Learning/pdf/Lec-06-improving-MLPs(1).pdf#page=70|slides ~68–73]]):

> *"Suppose a neuron in our network happens to get a very negative weight... that neuron gets a negative output for every data point — so gradients are always zero... that neuron's decision boundary gets pushed away from the data, and it will never train again."*

**Mechanism.** Consider a hidden unit with pre-activation $z = w^\top x + b$. If $z \le 0$ for **every** training example $x$, then:

1. Forward pass: $\text{ReLU}(z) = 0$ on every example.
2. Local derivative: $\text{ReLU}'(z) = 0$ on every example.
3. Backprop: the upstream gradient is multiplied by $0$ → the gradient w.r.t. $w$ and $b$ is exactly $0$.
4. SGD update: $w$ and $b$ don't change.
5. Goto 1 — the unit is dead **forever**.

**Triggers.**
- **Bad weight initialization.** Picking $w$ too large in magnitude can put $z$ deep in the negative region for all data points.
- **Too-large gradient step.** A single update with a large gradient + large $\eta$ can knock $z$ into the negative region for all training examples in one step.

**Why "good initialization helps prevent this."** L06's slide note. [[weight-initialization|He initialization]] ($\sigma^2 = 2/D_\text{in}$) keeps initial pre-activations centered near zero with variance ≈ 1, so a typical unit fires for roughly half the training examples — well clear of the all-negative trap.

**Why ReLU still wins despite this.** Empirically, only a fraction of ReLU units die, and the surviving ones train far faster than sigmoid/tanh would. Net effect on convergence speed: ReLU is *roughly 6× faster than sigmoid/tanh in practice* ([[30-Sources/Statistical-Learning/pdf/Lec-06-improving-MLPs(1).pdf#page=65|slide ~65]]).

## Variants that fix dead units

| Variant | Negative-side formula | Trade-off |
| --- | --- | --- |
| **Leaky ReLU** | $\alpha z$ with $\alpha = 0.01$ | small slope on negative side → no dead units; one extra hyperparameter (rarely tuned) |
| **Parametric ReLU (PReLU)** | $\alpha z$ with $\alpha$ **learned** | adapts slope per unit; risk of overfitting on small data |
| **ELU** | $\alpha(e^z - 1)$ | smooth + saturates softly on negative; expensive (exp) |
| **GELU** | $z \cdot \Phi(z)$ | smooth probabilistic gating; standard in Transformers |

L06 names all four ([[30-Sources/Statistical-Learning/pdf/Lec-06-improving-MLPs(1).pdf#page=72|slides ~72–76]]) and explicitly endorses Leaky/PReLU as the first thing to try when ReLU is dying.

## Comparison with sigmoid for hidden layers

| | Sigmoid | ReLU |
| --- | --- | --- |
| Range | $(0, 1)$ | $[0, \infty)$ |
| Derivative range | $(0, 0.25]$ | $\{0, 1\}$ |
| Saturation | both extremes | only the negative side |
| Computational cost | exp, divide | max, compare |
| Vanishing gradient | yes (severe in deep nets) | no (on the active side) |
| Modern default? | only for binary output | yes, hidden default |

L04's worked example uses sigmoid in the hidden layer initially because the geometry is easier to draw, then notes that ReLU works at least as well for the same toy problem and is the practical default everywhere else.

## Typical use

Hidden layers in MLPs, CNNs, and most modern architectures default to ReLU (or one of its variants like GELU). Sigmoid and $\tanh$ are reserved for output layers (binary classification) or specific cases where bounded outputs matter (gates in LSTMs).

## Related

- [[activation-function]] — the broader family.
- [[hidden-layer]] / [[multilayer-perceptron]] — where ReLU lives.
- [[vanishing-exploding-gradients]] — the problem ReLU partially solves on the positive side.
- [[weight-initialization]] — He init is what stops ReLUs from dying at initialization.
- [[lecture-06-improving-mlps|SLP L06]] — where the dead-unit problem and the variants are introduced.
