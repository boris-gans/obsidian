---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-04(1).pdf
created: 2026-05-03
---

# Universal approximation theorem

The theoretical guarantee that **a feedforward neural network with a single hidden layer of sufficient width can approximate any continuous function on a compact domain to arbitrary precision** — given a non-linear activation function. Cybenko (1989), Hornik–Stinchcombe–White (1989).

## The claim

For any continuous $f: K \subset \mathbb{R}^d \to \mathbb{R}$ on a compact $K$ and any $\epsilon > 0$, there exists an integer $H$, weight matrices $W_1 \in \mathbb{R}^{H \times d}$, $W_2 \in \mathbb{R}^{1 \times H}$, biases $b_1 \in \mathbb{R}^H$, $b_2 \in \mathbb{R}$, and a non-polynomial activation $\sigma$, such that:

$$
\sup_{x \in K} \big|\, f(x) - W_2 \sigma(W_1 x + b_1) - b_2 \,\big| < \epsilon.
$$

In English: **one hidden layer is enough, in principle**, to approximate any well-behaved target function.

## What the theorem does NOT say

[[lecture-04-mlps|SLP L04]] is careful about the caveats:

- **It doesn't bound $H$.** The required width can be astronomical for a target $\epsilon$ — possibly exponential in $d$.
- **It doesn't say the network is *easy to find*.** Stochastic gradient descent on a width-$H$ shallow network might not converge to the approximating parameters even when they exist.
- **It doesn't say single-hidden-layer is *the right architecture*.** Empirically, deeper-narrower architectures find solutions faster and need fewer total parameters.
- **It doesn't bound generalization error.** A wide single-layer net that fits training data perfectly can still fail to generalize.

## Why depth matters in practice anyway

[[30-Sources/Statistical-Learning/pdf/SLP-04(1).pdf#page=92|Slides ~92–98]]:

> *"It can be shown that a single hidden layer is all you need… In practice, solutions with more layers are easier to find with SGD and require fewer neurons in total (by breaking the problem into parts)."*

Three practical reasons:

1. **Optimization.** Empirically, the loss landscape of a deep, narrow network is easier for SGD to navigate than a shallow, wide one of equivalent expressive power.
2. **Parameter efficiency.** Functions with **compositional structure** (parity, multiplication-of-sums, hierarchical features) are exponentially cheaper to express with depth. A target function realizable by a depth-$L$ network with $\Theta(d)$ neurons may require $\Theta(2^d)$ neurons in a depth-1 network.
3. **Hierarchical representations.** Deep nets learn features at multiple levels of abstraction (edges → textures → parts → objects in vision; characters → words → phrases → meaning in NLP). This matches the structure of natural data and is the canonical justification for "deep learning" as opposed to "shallow learning." (Mock §1b answer.)

## What an MLP with $L$ hidden layers can carve out

[[lecture-04-mlps|SLP L04]]'s visualization ([[30-Sources/Statistical-Learning/pdf/SLP-04(1).pdf#page=95|slides ~93–98]], Touretzky / Johnson):

| Layers | Boundary type |
| --- | --- |
| 0 (perceptron) | hyperplane only |
| 1 hidden | convex polygons (intersections of half-planes via AND) |
| 2 hidden | unions of convex polygons (concave shapes, holes) |
| $\geq 3$ | compositions of compositions — arbitrary regions |

So depth doesn't add new "ultimate expressivity" beyond layer 1 (universal approximation), but it adds **finer-grained, easier-to-find** expressivity at lower parameter cost.

## Where this gets tested

- **Mock §1b:** "the model learns ___ representations" → answer **hierarchical**, justified by depth via universal approximation + compositional efficiency.
- **Conceptual questions** about why we use deep networks at all when one hidden layer suffices in theory — the answer is the three practical reasons above.

## Related

- [[multilayer-perceptron]] — the architecture this theorem applies to.
- [[activation-function]] — the theorem requires a non-polynomial activation.
- [[hidden-layer]] — the structural element that enables universal approximation.
- [[gradient-descent]] / [[stochastic-gradient-descent]] — the training algorithm that makes finding the approximating parameters feasible (or sometimes not).
