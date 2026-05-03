---
tags: [concept]
courses: [NLP, Statistical-Learning]
sources:
  - course: NLP
    file: Session 03 - Fundamental Concepts.pdf
  - course: Statistical-Learning
    file: SLP-lec2(1).pdf
created: 2026-05-02
---

# Perceptron

The foundational unit of neural networks. Lineage: McCulloh & Pitt (1943) → Rosenblatt (1957) → Minsky & Papert (1969) → modern deep learning.

## McCulloh-Pitt neuron (1943)

Mimicked human neurons using **propositional logic**. The unit fires when the sum of inputs exceeds a threshold:

$$y = H\!\left(\sum_i x_i - U\right)$$

Inputs are Boolean (excite / inhibit), $H$ is the **Heaviside step function**, $U$ is the **activation threshold**.

## Rosenblatt's perceptron (1957)

Generalized to **non-Boolean inputs with weights**. The unit computes a **weighted sum**:

$$y = H\!\left(\sum_i w_i x_i - U\right)$$

After introducing bias and a general non-linear activation, the unit becomes:

$$\boxed{y = f(\mathbf{w}\!\cdot\!\mathbf{x} + b)}$$

> The perceptron is the **composition of an affine transformation and a non-linear function**.

## Components

- **Weights** $w_i$ — control how each input contributes to the score
- **Bias** $b$ — shifts the function so it can cover the whole function space
- **Activation function** $f$ — typically non-linear (Heaviside, sigmoid, tanh, ReLU); see [[activation-function]]

## Why this matters for NLP

The perceptron is the unit out of which [[multilayer-perceptron|multi-layer perceptrons]], RNNs, LSTMs, and transformers are built. The shift from n-gram counting to perceptron-based models is the shift from **fixed statistics to internal parameters that adapt with experience** — the core conceptual move of Session 03.

The same form $z = wx + b$ followed by a sigmoid is exactly [[logistic-regression]] (Session 10) — perceptron with sigmoid activation = logistic-regression unit.

## Three issues with the perceptron (Statistical Learning, L02)

[[30-Sources/Statistical-Learning/pdf/SLP-lec2(1).pdf#page=2|Slide 2]] of L02 enumerates exactly the problems that motivate every later linear-classifier improvement:

1. **Binary-only.** The hard $\pm 1$ output supports two classes, not $C$. Fix: use $C$ neurons (one per class) — leads to softmax in L03.
2. **Doesn't terminate** unless the data is **perfectly linearly separable**. Fix: define a *good-line* criterion (a loss) so learning still has a target on noisy / non-separable data.
3. **When separable, returns *any* arbitrary separating hyperplane.** Some are demonstrably worse — points lying close to the boundary will easily flip with new data. Fix: replace the hard sign with a smooth output (the [[sigmoid]]) and minimize a [[cross-entropy|cross-entropy]] / [[logistic-loss|logistic]] loss.

This is the "what's wrong with the perceptron, what's the fix" slide that L02–L05 unpack. In SVMs (L09) the same critique #3 motivates **maximum-margin** instead of cross-entropy — different fix, same diagnosis.

## Why this matters for Statistical Learning

The perceptron is the *zero* of the course's neural-network arc:

- L02 fixes issue #3 (sigmoid + log-loss) → [[logistic-regression]] / single neuron.
- L03 fixes issue #1 (multi-class via softmax + multiple neurons) → introduces [[multilayer-perceptron|MLPs]].
- L04 stacks neurons → MLPs as universal approximators.
- L05 derives [[backpropagation]] to actually train the stacked network.
- L09 fixes issue #3 differently (max-margin) → [[linear-svm|linear SVMs]].

The perceptron's score function $w^T x + b$ is also reused as the "[[linear-classifier|template-matching]]" view (Stanford framing): rows of $W$ are class templates and the score is the dot product with the input.
