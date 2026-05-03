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
created: 2026-05-02
---

# Multilayer perceptron (MLP)

The **deep feedforward neural network** — the cornerstone of deep learning, built by stacking many [[perceptron|perceptrons]] in layers.

## Structure

- **Input layer** — receives the input features
- **Hidden layers** — intermediate layers, "hidden" because they are not directly observed
- **Output layer** — produces the prediction

The **depth** of the model is the number of hidden layers (hence *deep* learning); the **width** is the dimensionality of those layers.

## Goal

Approximate an unknown function $f^*(x)$ by defining a parameterized mapping
$$y = f(x, \theta)$$
and **learning the parameters** $\theta$ from data.

## Universal approximation theorem

Cybenko (1989) and Hornik, Stinchcombe & White (1989) showed that neural networks are **universal approximators**: they can approximate certain classes of functions to arbitrary precision given enough hidden units.

The theorem doesn't say it's *easy* to find those parameters or that the network will generalize — only that the function class is rich enough.

## From n-gram counting to MLP learning

- **N-gram models** ([[n-gram-model]]) rely on **explicit frequency counts**; behaviour changes only as new counts accumulate.
- **MLPs** have **internal parameters** modified through experience. Parameters generalize across inputs in a way that fixed tables cannot.
- The **learning process** = adjust parameters so outputs better match desired targets — typically by gradient-based optimization (backpropagation).

This is the conceptual move from Session 03: from statistics-as-fixed-tables to **learned representations**.

## Formal feedforward equations (Session 16, [[30-Sources/NLP/pdf/Session 16 - Neural Networks-1.pdf#page=7|slide 7]])

A feedforward neural network with **one hidden layer**:
$$h = \sigma(W_1 x + b_1)$$
$$y = W_2 h + b_2$$
Combining:
$$y = W_2 \,\sigma(W_1 x + b_1) + b_2$$

Generalization to many layers ([[30-Sources/NLP/pdf/Session 16 - Neural Networks-1.pdf#page=8|slide 8]]) — repeated application of an [[activation-function]] $\sigma$ to the output of each previous layer:
$$x^{(k+1)} = \sigma\big(W^{(k)} x^{(k)} + b^{(k)}\big)$$
representable as a **directed acyclic graph** with input → hidden layers → output.

The intermediate representations $h^{(k)}$ are **not specified in advance** — they are **learned from the data during training** ([[30-Sources/NLP/pdf/Session 16 - Neural Networks-1.pdf#page=7|slide 7]]). This is the conceptual shift from rule-engineered features to **learned representations**.

## Vector vs neuron computation (Session 16, [[30-Sources/NLP/pdf/Session 16 - Neural Networks-1.pdf#page=5|slides 5–6]])

- **Single neuron:** $y = \sigma(w^\top x + b)$ — a parametric nonlinear function $\mathbb{R}^n \to \mathbb{R}$.
- **Layer of $m$ neurons:** $h = \sigma(W x + b)$ with $W \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$ — $\mathbb{R}^n \to \mathbb{R}^m$. Each row of $W$ corresponds to one neuron's weight vector; the activation is applied componentwise.

## Two views of the hidden layer (Statistical Learning, L04)

[[lecture-04-mlps|SLP L04]] explicitly draws **two equivalent views** of what a hidden layer does ([[30-Sources/Statistical-Learning/pdf/SLP-04(1).pdf#page=50|slides ~45–55]]):

1. **Decision-boundary view.** Each hidden neuron defines a half-plane (its individual linear boundary). The output neuron Boolean-combines these half-planes (AND, OR, etc.) to produce the final decision region. With 3 hidden neurons + an AND output, you get a triangle; with 2 hidden + OR, you get a different shape; with $H$ hidden + a learned output, you get any convex polygon.
2. **Feature-transformation view.** The hidden layer maps the input space $\mathbb{R}^d \to \mathbb{R}^H$ to a new representation where the data **becomes linearly separable**. The output neuron is then "just" a linear classifier in that learned feature space. The lecture uses the **Cartesian → polar coordinates** analogy: concentric rings in $(x, y)$ are not linearly separable, but they are in $(r, \theta)$ — the hidden layer learns the equivalent of that transformation automatically.

The two views describe the *same* network at different levels of abstraction.

## Why hidden activations must be non-linear (L04)

Without non-linear activations, an MLP collapses:

$$
W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2),
$$

— a single linear layer. **Linear transforms can only rotate, reflect, scale, and shear**; grid lines remain parallel and evenly spaced. So depth gives no expressive power without non-linearity ([[30-Sources/Statistical-Learning/pdf/SLP-04(1).pdf#page=70|slides ~65–80]]).

L04 also notes that **softmax is wrong for hidden layers** ([[30-Sources/Statistical-Learning/pdf/SLP-04(1).pdf#page=27|slides ~26–29]]): it couples outputs through normalization, which destroys the rich multi-feature representation hidden layers need. Sigmoid, $\tanh$, and ReLU all preserve per-component independence and are valid hidden-layer choices; ReLU is the modern default.

## Depth → boundary complexity

[[30-Sources/Statistical-Learning/pdf/SLP-04(1).pdf#page=95|Slides ~93–98]] visualize how depth increases expressivity:

- **1 hidden layer:** convex polygon regions (intersections of half-planes via AND).
- **2 hidden layers:** unions of convex polygons → concave shapes, holes (OR of polygons).
- **$\geq 3$ hidden layers:** compositions of compositions → arbitrary decision regions.

This is what justifies "deep" in *deep* learning. See [[universal-approximation-theorem]] for the formal claim.
