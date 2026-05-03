---
tags: [flashcards, Statistical-Learning]
course: Statistical-Learning
lecture: 05
source: Lec-05-backprop(1).pdf
created: 2026-05-03
---

# Lecture 05 — Backpropagation

What problem does backpropagation solve?
?
Computing gradients of the loss with respect to *every* parameter in a deep network, efficiently. Without it, training deep networks is computationally infeasible.

What's the chain rule, in one line?
?
For a composition $y = f(g(x))$: $\dfrac{dy}{dx} = \dfrac{df}{dg} \cdot \dfrac{dg}{dx}$. Multi-step compositions multiply more factors along the chain.

Write the chain-rule expansion of $\partial \mathcal{L}/\partial w_r$ for a two-layer toy network with $\mathcal{L} \leftarrow a_r \leftarrow z_r \leftarrow w_r$.
?
$\dfrac{\partial \mathcal{L}}{\partial w_r} = \dfrac{\partial \mathcal{L}}{\partial a_r} \cdot \dfrac{\partial a_r}{\partial z_r} \cdot \dfrac{\partial z_r}{\partial w_r}$.

What is the derivative of the sigmoid $\sigma(z) = 1/(1+e^{-z})$ w.r.t. $z$?
?
$\sigma'(z) = \sigma(z)(1 - \sigma(z))$. Equivalently $a(1 - a)$ if $a = \sigma(z)$. **Memorize this** — it shows up in every sigmoid-network gradient.

What's the derivative of squared-error loss $\mathcal{L} = \tfrac{1}{2}(y - a)^2$ w.r.t. $a$?
?
$\dfrac{\partial \mathcal{L}}{\partial a} = a - y$. Note the sign — derivative w.r.t. the prediction $a$, not the target $y$.

What's the derivative of binary cross-entropy $\mathcal{L} = -\log a$ (positive class, $y = +1$) w.r.t. $a$?
?
$\dfrac{\partial \mathcal{L}}{\partial a} = -\dfrac{1}{a}$.

What's the derivative of $\mathcal{L} = -\log(1 - a)$ (negative class, $y = 0$ or $y = -1$) w.r.t. $a$?
?
$\dfrac{\partial \mathcal{L}}{\partial a} = \dfrac{1}{1 - a}$.

What's the derivative of an affine score $z = w x$ w.r.t. the weight $w$? W.r.t. the input $x$?
?
$\dfrac{\partial z}{\partial w} = x$. $\dfrac{\partial z}{\partial x} = w$.

What's the gradient of ReLU's output $a = \max(0, z)$ w.r.t. $z$?
?
$1$ if $z > 0$, $0$ if $z < 0$. Undefined at $z = 0$ but typically set to $0$ or $1$ in implementations.

For the SLP L05 toy network with squared-error loss, what is $\partial \mathcal{L}/\partial w_r$ in closed form?
?
$\partial \mathcal{L}/\partial w_r = (a_r - y) \cdot a_r(1 - a_r) \cdot a_l$ — loss derivative × sigmoid derivative × input.

For the same toy network, what is $\partial \mathcal{L}/\partial w_l$ in closed form?
?
$\partial \mathcal{L}/\partial w_l = (a_r - y) \cdot a_r(1 - a_r) \cdot w_r \cdot a_l(1 - a_l) \cdot x$ — extends the chain through one more sigmoid + affine layer.

What's the algorithmic insight that makes backprop $O(\text{forward-pass cost})$?
?
**Common subterms in chained gradients are cached and reused** as the backward sweep proceeds. The shared prefix $\partial \mathcal{L}/\partial a_r \cdot \partial a_r/\partial z_r$ is computed once and reused for both $\partial \mathcal{L}/\partial w_r$ and $\partial \mathcal{L}/\partial w_l$. Without caching, naive recomputation gives $O(\text{depth} \times \text{forward})$.

What is a computational graph?
?
A DAG where nodes are operations and edges carry intermediate values. The data structure that lets autograd engines (PyTorch, JAX, TF) compute gradients for *any* architecture: each node only needs its local function and local gradient.

What's the universal node recipe in the backward pass?
?
$\dfrac{\partial \mathcal{L}}{\partial x_i} = \underbrace{\dfrac{\partial \mathcal{L}}{\partial f}}_{\text{upstream}} \cdot \underbrace{\dfrac{\partial f}{\partial x_i}}_{\text{local}}$. Receive upstream gradient, multiply by local gradient, send downstream.

In what order does the forward pass traverse the computational graph? The backward pass?
?
**Forward:** topological order (a node fires when all its inputs are ready, left-to-right). **Backward:** reverse topological order (a node fires when every consumer of its output has reported back, right-to-left).

What happens at a node whose output feeds two downstream consumers, during the backward pass?
?
**Sum the upstream gradients** from the two consumers. The downstream gradient is $\sum_p \partial \mathcal{L}/\partial f_p \cdot \partial f_p/\partial s_0$ — one term per path.

Why does backprop need an explicit graph rather than just "running the chain rule"?
?
For arbitrary architectures (Transformers, ResNets, branching/skip connections), the closed-form per-weight gradient is intractable to write by hand and would change every time the architecture changes. The graph engine handles *any* architecture mechanically, reusing the same machinery.

What's the difference between forward-mode and reverse-mode automatic differentiation?
?
**Forward-mode:** propagate $\partial \cdot / \partial x$ left-to-right; cheap for few inputs / many outputs. **Reverse-mode = backprop:** propagate $\partial L / \partial \cdot$ right-to-left; cheap for many inputs (parameters) / one output (loss). Reverse mode dominates in deep learning.

For an $N$-parameter network, what's the time complexity of a single backprop pass?
?
$O(N)$ — same order as one forward pass. Memory is also $O(N)$ for caching forward activations needed during backward.

True or False: backpropagation requires the activation function to be non-linear.
?
**False.** Backprop computes gradients for *any* differentiable activation, including identity. But **the network itself** is uninteresting without nonlinearity (it collapses to linear). Backprop works either way; nonlinearity is what makes the training meaningful.

What does the "back" in backpropagation refer to?
?
The **backward sweep** through the computational graph in reverse topological order, where gradients flow from the loss back toward the inputs/parameters. (Not "back-propagated through time" — that's BPTT, an extension for RNNs.)

When backprop computes gradients, what direction in parameter space do those gradients point?
?
The direction of **steepest increase** in the loss. Gradient descent then steps in the **opposite** direction (negative gradient) to decrease the loss.

When was backpropagation first applied to neural networks?
?
The **1970s** — about 30 years after the 1957 perceptron, and 15+ years before the 1986 Rumelhart-Hinton-Williams paper that popularized it for deep nets.

In a single PyTorch training step, name the calls in order: forward / loss / backward / step.
?
`pred = model(x)` (forward), `loss = criterion(pred, y)`, `loss.backward()` (autograd computes all parameter gradients via reverse-mode AD), `optimizer.step()` (SGD/Adam applies update).
