---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: Lec-06-improving-MLPs(1).pdf
created: 2026-05-03
---

# Dropout

A regularizer for neural networks: during training, randomly **zero out** each hidden unit with probability $p$ on each forward pass. At test time, dropout is off (every unit fires) — but training-time activations are scaled by $1/(1-p)$ so that test-time and training-time expected magnitudes match.

## Training-time mechanics

For each forward pass, sample an i.i.d. mask $m_i \sim \text{Bernoulli}(1 - p)$ for each hidden unit. The masked activation is

$$
\tilde{a}^{(\ell)}_i \;=\; m_i \cdot a^{(\ell)}_i.
$$

A different mask per **example**, per **layer**, per **forward pass**. Backprop through the dropout layer is identical to multiplication: the mask's local gradient is itself, so the dropped units pass zero gradient to their inputs and the surviving units pass a gradient of $1$.

Typical $p$:
- $p = 0.5$ for fully-connected hidden layers (the original L06 / Hinton recommendation)
- $p = 0.1$ to $0.2$ for input layers (less aggressive)
- $p = 0$ on the output layer (never drop the prediction itself)

## Test-time mechanics

Dropout is **off** at test time — every unit fires. But the expected output of each layer at training time was

$$
\mathbb{E}[\tilde{a}_i] = (1 - p) \cdot a_i,
$$

so naïvely turning dropout off at test time leaves test-time activations *too large* by a factor of $1/(1-p)$. Two equivalent fixes:

**Option 1 — scale at test time.** Multiply every test-time activation by $(1-p)$. Matches training expectations.

**Option 2 — inverted dropout (the standard).** Multiply every *training-time* activation by $1/(1-p)$ on top of the mask. Test-time pass needs no rescaling.

$$
\tilde{a}^{(\ell)}_i \;=\; \frac{m_i}{1 - p} \cdot a^{(\ell)}_i.
$$

Inverted dropout is the standard implementation because it pushes the rescaling into training, leaving the test-time path as plain forward propagation. L06's framing ([[30-Sources/Statistical-Learning/pdf/Lec-06-improving-MLPs(1).pdf#page=102|slide ~102]]):

> *"need to scale so the total output coming from the layer using dropout at test time keeps about the same magnitude as observed during training."*

## Why it works — two complementary stories

**Robustness story.** Dropping units forces the network to spread its representation across many redundant paths — no single hidden unit can be a critical bottleneck. The surviving network is more robust to noise / missing inputs / adversarial perturbations.

**Ensemble story.** Each dropout mask defines a *subnetwork* that shares weights with all other masks. With $N$ hidden units and probability $p$, there are $2^N$ possible masks. **Training with dropout is approximately training an exponentially large ensemble of subnetworks** that share weights; test-time evaluation with dropout off and all weights present is approximately *averaging* this ensemble's predictions ([[30-Sources/Statistical-Learning/pdf/Lec-06-improving-MLPs(1).pdf#page=98|slide ~98]]):

> *"Subsequent layers may combine the outputs of these parallel models to reach more robust predictions."*

The two stories agree on what dropout does, just describe it from different angles. Both predict (correctly) that dropout helps **most** in over-parameterized networks, where there's enough capacity to spare for redundancy.

## What does dropout regularize?

It penalizes co-adaptation: hidden units that **rely on specific other units being present**. A unit that is "redundant by design" survives dropout; a unit that learned a brittle co-adaptation gets noisy gradients and weakens.

Empirically dropout reduces test-set overfitting. In bias–variance language: dropout *increases* training-set bias slightly (the model can't fit as tightly with random masking) and *decreases* variance substantially (the ensemble averaging smooths predictions).

## Dropout in modern architectures

- **Fully-connected layers**: standard, $p \in [0.3, 0.5]$.
- **Convolutional layers**: less common; spatial dropout (drop entire feature maps) sometimes used instead.
- **Recurrent layers**: variational dropout (same mask across time steps) outperforms naive per-step dropout.
- **Transformers**: dropout on attention probabilities and on residual outputs, $p \in [0.1, 0.2]$.

L06 keeps the discussion at the basic per-unit level — modern variants are out of scope.

## Exam-relevant facts

- Dropout is **only active during training**.
- Standard practice is **inverted dropout** — scale activations by $1/(1-p)$ at training time so test-time pass is plain.
- Dropout is interpretable as training an **exponential ensemble of weight-shared subnetworks**.
- $p$ is a **hyperparameter**; typical $p = 0.5$ for FC hidden layers.
- Dropout reduces **co-adaptation** between hidden units → less variance, slightly more bias.

## Related

- [[multilayer-perceptron]] — the architecture dropout regularizes.
- [[regularization]] — broader family ([[lecture-10-loss-functions-regularization|L10]]).
- [[lecture-06-improving-mlps|SLP L06]] — source.
