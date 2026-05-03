---
tags: [concept]
courses: [NLP, Statistical-Learning]
sources:
  - course: NLP
    file: Session 17 - Recurrent NN.pdf
  - course: Statistical-Learning
    file: Lec-06-improving-MLPs(1).pdf
created: 2026-05-02
---

# Vanishing / exploding gradients

A structural failure mode of deep neural networks: gradients computed by backpropagation through many layers (or time steps) **shrink to zero or grow without bound exponentially in depth (or sequence length)**. Same phenomenon, two contexts:

- **NLP framing** ([[recurrent-neural-network|RNNs]] under [[bptt|BPTT]]): exponential in *sequence length* via the recurrent matrix's spectral radius ([[30-Sources/NLP/pdf/Session 17 - Recurrent NN.pdf#page=14|slide 14]]).
- **SLP framing** (deep [[multilayer-perceptron|MLPs]] under standard backprop): exponential in *depth* via the chain rule's product of per-layer local-gradient factors.

Same math, different axis. Below: the RNN treatment first, then the SLP treatment.

The blueprint flags this as **high weight**: Quiz IV Q1, Q2 (and Model B variants) target the mechanism.

## The mechanism ([[30-Sources/NLP/pdf/Session 17 - Recurrent NN.pdf#page=14|slide 14]], [[30-Sources/NLP/pdf/Session 17 - Recurrent NN.pdf#page=15|slide 15]])

During BPTT, the gradient of the loss with respect to an early-step parameter passes through the **recurrent weight matrix $W$ at every intervening step**. Mathematically, the gradient depends on a product of Jacobians, dominated by the **singular values of $W$**:

- If the largest singular value of $W$ is **smaller than 1** → gradients **shrink exponentially** with sequence length → **vanishing**
- If the largest singular value is **larger than 1** → gradients **grow exponentially** → **exploding**

Either way, training breaks down for long sequences.

## What vanishing gradients mean in practice ([[30-Sources/NLP/pdf/Session 17 - Recurrent NN.pdf#page=14|slide 14]])

> "Since gradients vanish, the update of the hidden states will depend mostly on the **last tokens** while farther tokens will contribute less and less."

The model can only learn dependencies within a small window — empirically, **about 5–10 tokens** for vanilla Elman RNNs ([[30-Sources/NLP/pdf/Session 17 - Recurrent NN.pdf#page=14|slide 14]]).

> "Empirically, simple RNNs tend to rely mostly on the most recent words, and their effective context length is often limited to a small number of tokens." ([[30-Sources/NLP/pdf/Session 17 - Recurrent NN.pdf#page=15|slide 15]])

## Why this matters: long-range dependencies fail

Consider "The book that I bought yesterday because the reviews were excellent **was** expensive" ([[30-Sources/NLP/pdf/Session 17 - Recurrent NN.pdf#page=15|slide 15]]). To predict "was", the model must remember the subject "book" — but many tokens intervene. With vanishing gradients, the gradient signal connecting "book" to "was" is too weak to learn the dependency reliably.

## What's the fix?

| Problem | Mitigation |
|---|---|
| **Exploding gradients** | **Gradient clipping** — cap the gradient norm during backprop [not in source — practical standard] |
| **Vanishing gradients** | **Gating mechanisms** — [[lstm|LSTM]] (forget/input/output gates with cell state) and [[gru|GRU]] (reset/update gates) protect information across long sequences ([[30-Sources/NLP/pdf/Session 17 - Recurrent NN.pdf#page=15|slide 15]]) |
| **Long-range dependencies** | Ultimately solved by **[[attention|attention mechanisms]]** — direct token-to-token connections, no intermediate bottleneck |

LSTMs and GRUs use gates whose multiplicative paths can preserve gradients across long horizons — they don't *eliminate* vanishing, but they vastly extend the effective context. Transformers eliminate the recurrence altogether.

## Exam framing

| Question | Answer |
|---|---|
| Why do RNN gradients vanish or explode? | BPTT multiplies by the recurrent matrix $W$ at every step. Spectral radius < 1 → exponential shrinkage; > 1 → exponential growth (Quiz IV Q1, Q2) |
| What's the practical consequence for the RNN's effective context? | Limited to ~5–10 tokens — distant tokens contribute negligibly to gradient updates ([[30-Sources/NLP/pdf/Session 17 - Recurrent NN.pdf#page=14|slide 14]]) |
| What architectural fix preserves long-term information? | **Gating** — LSTM (forget/input/output) or GRU (reset/update). Eventually: attention. |

## SLP framing — deep MLPs under standard backprop

[[lecture-06-improving-mlps|SLP L06]] frames the same problem as a chain-rule product across **layers**, not time steps. The two framings are mathematically identical — multiplying $L$ Jacobians in a chain — but the diagnostic differs by activation choice.

**The mechanism.** For an $L$-layer MLP with sigmoid activations, the gradient w.r.t. an early-layer weight $w^{(1)}$ has the form

$$
\frac{\partial \mathcal{L}}{\partial w^{(1)}} \;=\; \frac{\partial \mathcal{L}}{\partial a^{(L)}} \cdot \prod_{\ell=1}^{L} \sigma'(z^{(\ell)}) \cdot W^{(\ell)} \cdots,
$$

where every layer contributes one $\sigma'(z^{(\ell)}) \le 0.25$ factor. If pre-activations sit in the saturated region, each factor is much smaller than $0.25$ — the product shrinks to zero in a handful of layers ([[30-Sources/Statistical-Learning/pdf/Lec-06-improving-MLPs(1).pdf#page=58|slides ~55–62]]).

**Sigmoid is the worst offender.** $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ has max value $0.25$ at $z = 0$ and decays exponentially toward zero in both directions. Every hidden layer that uses sigmoid is *guaranteed* to multiply the gradient by ≤ 0.25. **Stack ten such layers and the gradient is attenuated by at least $0.25^{10} \approx 10^{-6}$.**

**$\tanh$ is mildly better.** $\tanh'(z) = 1 - \tanh^2(z)$ peaks at $1$ (vs. $0.25$ for sigmoid), so the worst-case shrinkage per layer is closer to $1$. But $\tanh$ still saturates on both sides for $|z| \gg 1$.

**ReLU is the structural fix.** $\text{ReLU}'(z) = 1$ for $z > 0$ — no shrinkage on the active side. Combined with [[weight-initialization|He initialization]] (which keeps pre-activations near zero with variance ~1 so a typical unit is in the active region), the chain-rule product no longer collapses by depth.

### Why ReLU + good init enables modern deep learning

The pre-2010 conventional wisdom was that "deep networks don't train" — they don't, with sigmoid + naive small-random init. Two changes broke the curse:

1. **ReLU** removed the per-layer shrinkage on the active side.
2. **Variance-controlled initialization** ([[weight-initialization|Xavier / He]]) kept activations and gradients balanced across layers in expectation.

Neither alone is enough; together, they made networks of arbitrary depth trainable.

### Exam framing — SLP angle

| Question | Answer |
| --- | --- |
| Why do gradients vanish in a deep MLP with sigmoid? | $\sigma'(z) \le 0.25$ at every layer; chain rule multiplies these factors → product → 0 exponentially in depth |
| Why is ReLU on the active side a fix? | $\text{ReLU}'(z) = 1$ for $z > 0$, no per-layer shrinkage |
| Why does He init matter for ReLU? | Keeps pre-activations near zero with controlled variance → typical unit fires (in the active region), so the no-shrinkage property holds in practice, not just in theory |
| Connection to the dead-ReLU problem? | Dead ReLU is the *opposite* failure: a unit *always* in the inactive region. He init fixes both — neither too saturated (sigmoid) nor too dead (ReLU). |

## Related

- [[recurrent-neural-network]] — the architecture where this first hits in the NLP course
- [[bptt|BPTT]] — the algorithm that exposes the problem in RNNs
- [[lstm]] — gated mitigation for sequence-length vanishing
- [[gru]] — simpler gated alternative
- [[lecture-06-improving-mlps|SLP L06]] — where the depth-axis framing lives
- [[relu]] — the activation that breaks the per-layer shrinkage on the active side
- [[weight-initialization]] — the second half of the fix (Xavier for tanh, He for ReLU)
