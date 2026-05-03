---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-04(1).pdf
created: 2026-05-03
---

# Hidden layer

The intermediate computational layer in a neural network — the one that's neither input nor output. "Hidden" because its activations $h$ aren't directly observed (no labels exist for what $h$ should be); the network *learns* what they should represent during training.

## Forward computation

A single hidden layer with $H$ units takes input $x \in \mathbb{R}^d$ and produces $h \in \mathbb{R}^H$:

$$
h = f(W_1 x + b_1), \qquad W_1 \in \mathbb{R}^{H \times d},\ b_1 \in \mathbb{R}^H.
$$

The non-linear activation $f$ is applied **componentwise**. Each row of $W_1$ defines one *unit* — analogous to one [[perceptron|neuron]].

## Two views of what a hidden layer does (SLP L04)

[[lecture-04-mlps|SLP L04]] frames the hidden layer in two equivalent ways ([[30-Sources/Statistical-Learning/pdf/SLP-04(1).pdf#page=50|slides ~45–55]]):

**View 1 — multiple decision boundaries combined.** Each hidden neuron defines a half-plane in input space. The output neuron Boolean-combines (AND, OR, weighted-and-thresholded) these half-planes into the final decision. With $H$ hidden neurons + an AND-like output, you carve a convex polygon out of input space.

**View 2 — feature transformation.** The hidden layer maps $\mathbb{R}^d \to \mathbb{R}^H$ to a representation in which the data **becomes linearly separable**. The output neuron is then "just" a linear classifier in that learned feature space. The motivating analogy: Cartesian → polar coordinates makes concentric rings linearly separable; the hidden layer learns *its own version* of that transformation, suited to the data.

The two views describe the same network at different abstraction levels — it's not "one or the other."

## Why hidden activations must be non-linear

If $f$ is the identity (no activation), the hidden layer collapses:

$$
W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2),
$$

— back to a single linear classifier. **Linear maps can only rotate, reflect, scale, shear**; they preserve grid-line parallelism. Composing them stays linear.

So *any* non-linearity unlocks expressive power. Sigmoid, $\tanh$, ReLU all work. Softmax does **not** — it couples output dimensions through normalization, which destroys the multi-feature representation the hidden layer is trying to build ([[30-Sources/Statistical-Learning/pdf/SLP-04(1).pdf#page=27|slides ~26–29]]). See [[activation-function]] for the choice.

## Width $H$ as a knob

More units = richer feature space = more complex achievable boundaries, at the cost of more parameters and more risk of overfitting. SLP L04 doesn't give a closed formula for picking $H$ — it's selected empirically (cross-validation).

| Width $H$ | What you can express in 1 hidden layer |
| --- | --- |
| 1 | a single half-plane (just a perceptron) |
| 2–3 | small convex polygons (triangles, etc.) |
| moderate | richer convex polygon regions, smooth-ish boundaries |
| very large | universal-approximator territory ([[universal-approximation-theorem]]) |

## Stacking hidden layers

A deep MLP has multiple hidden layers $h^{(1)}, h^{(2)}, \ldots$ each transforming the previous one:

$$
h^{(k+1)} = f(W^{(k+1)} h^{(k)} + b^{(k+1)}).
$$

This composition is what produces **hierarchical representations** (mock §1b answer). Layer $k$'s features can build on layer $k-1$'s features — edges → textures → parts → objects, etc.

## Hidden layers are unsupervised in spirit

There's no label data telling the network what $h$ should be — the network *figures out* a useful intermediate representation by minimizing the *output* loss. This is what makes hidden layers powerful and also what makes them hard to interpret: the learned features reflect whatever was useful for the task, not anything we explicitly designed.

## Related

- [[multilayer-perceptron]] — the architecture made of one or more hidden layers.
- [[activation-function]] — the non-linearity that makes hidden layers expressive.
- [[universal-approximation-theorem]] — why one hidden layer is enough in theory.
- [[backpropagation]] — how hidden layers' weights are trained from the output loss.
