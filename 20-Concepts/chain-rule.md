---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: Lec-05-backprop(1).pdf
created: 2026-05-03
---

# Chain rule

The calculus identity that lets us differentiate composite functions one piece at a time. **The mathematical foundation of [[backpropagation]]** — every gradient through a deep network is a chain-rule product of local derivatives.

## Single-variable form

For $y = f(g(x))$:

$$
\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}.
$$

The derivative of a composition is the product of derivatives along the chain.

## Multi-step chain

For a longer composition $L = h(g(f(x)))$, with intermediate variables $z = f(x)$, $a = g(z)$, $L = h(a)$:

$$
\frac{dL}{dx} = \frac{dL}{da} \cdot \frac{da}{dz} \cdot \frac{dz}{dx}.
$$

Multi-step chains are what give backprop its power: a deep network is a deep composition, and its gradient is a product of one factor per layer.

## Multivariate / partial form

When $f$ has multiple inputs $f(x_1, \ldots, x_k)$ and you want $\partial L/\partial x_i$:

$$
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial x_i}.
$$

If $x_i$ also influences $L$ through *other* paths (via additional outputs of $x_i$ in the computational graph), sum across paths:

$$
\frac{\partial L}{\partial x_i} = \sum_{\text{paths } p \text{ from } x_i \text{ to } L} \prod_{\text{edges in } p} \frac{\partial \text{output}}{\partial \text{input}}.
$$

This summation over paths is what makes the **backward pass at a fan-out node sum its upstream gradients** ([[computational-graph]]).

## SLP L05 application: derive backprop in three lines

[[lecture-05-backprop|SLP L05]]'s derivation is just chain rule applied to the composition of operations in a network.

For a two-neuron network with $L \leftarrow a_r \leftarrow z_r \leftarrow w_r$:

$$
\frac{\partial L}{\partial w_r} = \frac{\partial L}{\partial a_r} \cdot \frac{\partial a_r}{\partial z_r} \cdot \frac{\partial z_r}{\partial w_r}.
$$

For the deeper weight $w_l$, the chain extends through one more layer:

$$
\frac{\partial L}{\partial w_l} = \frac{\partial L}{\partial a_r} \cdot \frac{\partial a_r}{\partial z_r} \cdot \frac{\partial z_r}{\partial a_l} \cdot \frac{\partial a_l}{\partial z_l} \cdot \frac{\partial z_l}{\partial w_l}.
$$

Each individual factor is "very simple" ([[30-Sources/Statistical-Learning/pdf/Lec-05-backprop(1).pdf#page=60|slides ~55–80]]) — sigmoid derivative is $a(1-a)$, affine derivative is the input, etc. The chain rule does all the gluing.

## Why the chain rule + caching = $O(\text{forward-pass cost})$ backprop

The two gradient products above share a common prefix $\partial L/\partial a_r \cdot \partial a_r/\partial z_r$. **Computing it once and reusing it** is what makes backprop linear in the number of weights rather than quadratic. The chain rule is just the math; caching the prefixes is the algorithm.

This generalizes: for a network with $L$ layers, the gradient w.r.t. layer-$k$ weights shares the prefix with the gradients of all later layers. A single backward sweep computes all gradients in $O(\text{forward})$ time.

## Forward-mode vs. reverse-mode autodiff

The chain rule can be evaluated in either direction:

- **Forward mode:** propagate $\partial \cdot / \partial x$ left-to-right through the graph. Cost: $O(\text{forward})$ for *one* input — but you have to redo the whole sweep for each input.
- **Reverse mode (backprop):** propagate $\partial L/\partial \cdot$ right-to-left through the graph. Cost: $O(\text{forward})$ for *one* output (the loss). But you get gradients w.r.t. **all inputs/parameters** in one sweep.

For neural-net training, $W$ (parameters) ≫ $1$ (loss), so reverse mode wins by a huge margin. This is why "backpropagation" is the algorithm used in practice; forward-mode autodiff exists but is mainly used for sensitivity analysis with few outputs.

## Common chain-rule pitfalls

- **Forgetting the inner derivative.** $d/dx \sin(x^2) = \cos(x^2) \cdot 2x$, not $\cos(x^2)$.
- **Forgetting to sum at fan-outs.** If a variable feeds two downstream paths, both contribute via the multi-path form above.
- **Sign errors with subtraction.** $d(y - \hat{y})^2 / d\hat{y} = -2(y - \hat{y})$, not $2(y - \hat{y})$.

## Related

- [[backpropagation]] — the algorithm that systematically applies the chain rule.
- [[computational-graph]] — the data structure that tracks composition order.
- [[gradient-descent]] — the consumer of the gradients the chain rule produces.
