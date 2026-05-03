---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 19 - Transformers-1.pdf
created: 2026-05-02
---

# Scaled dot-product attention

The standard attention formula used in [[transformer|transformers]] ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=14|slide 14]]). Generalizes [[attention|Bahdanau-style attention]] by introducing **separate learned projections** $W_Q, W_K, W_V$ and **rescaling by $\sqrt{d_k}$** to keep softmax inputs in a usable range.

The blueprint flags this as **very high weight**: the formula is on the formula sheet; **mock Exercise 1 (Q26, 10 pts)** computes this end-to-end on small matrices. Quiz IV Q9 tests the rationale for the $\sqrt{d_k}$ scaling.

## The formula (formula sheet + [[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=14|slide 14]])

Given input vectors stacked as a matrix $X$:
$$Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V$$
where $W_Q, W_K, W_V \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$ are **randomly initialized learnable matrices**.

**Scores** — pairwise dot products:
$$S = Q K^\top, \qquad S_{ij} = q_i \cdot k_j$$

**Scaled scores** — divide by $\sqrt{d_k}$:
$$\frac{Q K^\top}{\sqrt{d_k}}$$

**Attention weights** — row-wise softmax (each row sums to 1):
$$\alpha = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right), \qquad \alpha_i = \frac{e^{S_i}}{\sum_j e^{S_j}}$$

**Output** — weighted sum of values:
$$\mathrm{Attention}(Q, K, V) = \alpha V$$

## Why divide by $\sqrt{d_k}$ (Quiz IV Q9)

[[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=14|slide 14]] only states the rule — "with a standard scaling usually taken as the square-root of the dimensionality of the key vector ($d_k$)" — without giving a rationale.

[not in source — supplementary, from Vaswani et al.] For random vectors of dimension $d_k$, the variance of their dot product grows with $d_k$. Without scaling, large $d_k$ produces large dot products → softmax saturates → gradients vanish. **Dividing by $\sqrt{d_k}$ keeps the variance roughly constant** so softmax stays in a usable range across model sizes.

## How to drill the exercise (mock Q26 procedure)

For 2×2 inputs (the exam scale):

| Step | Operation | Result |
|---|---|---|
| 1 | Read $Q, K, V$ from the question | Three small matrices |
| 2 | Compute $S = QK^\top$ | $2 \times 2$ matrix of dot products |
| 3 | Scale: $S' = S / \sqrt{d_k}$ | Same shape, divided element-wise |
| 4 | Row-wise softmax: $\alpha_{ij} = e^{S'_{ij}} / \sum_k e^{S'_{ik}}$ | Rows sum to 1 |
| 5 | Output: $\alpha V$ | $2 \times d_v$ matrix |

The arithmetic is small enough to do by hand; key risk areas:
- **Forget to scale** — drop the $\sqrt{d_k}$ division at your peril
- **Wrong axis softmax** — apply softmax row-wise (each query row's weights sum to 1), not column-wise
- **Matrix multiplication order** — $\alpha V$, not $V \alpha$; output rows correspond to queries

## What separates this from "attention" generally ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=7|slides 7]] vs 321)

| | Bahdanau-style attention ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=7|slide 7]], classical) | Scaled dot-product attention ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=14|slide 14]], transformer) |
|---|---|---|
| Score function | $s_t^\top h_i$ — direct dot product of decoder state and encoder state | $q_i \cdot k_j$ — dot product of *projected* vectors |
| Q/K/V source | $Q$ from decoder; $K, V$ from encoder (and $K = V$) | All three from learned linear projections of input |
| Scaling | None | $1/\sqrt{d_k}$ |
| Where used | RNN encoder + attention decoder | Self-attention inside transformer blocks |

## Exam framing

| Question | Answer |
|---|---|
| Write the full scaled dot-product attention formula | $\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(QK^\top / \sqrt{d_k}) V$ |
| What does $W_Q, W_K, W_V$ refer to? | Randomly initialized **learnable projection matrices** that produce queries, keys, and values from input embeddings |
| Why divide by $\sqrt{d_k}$? | Prevents large dot products from saturating the softmax — keeps gradient flow stable across model sizes (Quiz IV Q9) |
| Compute attention for a 2×2 problem | (1) $QK^\top$, (2) divide by $\sqrt{d_k}$, (3) row-wise softmax, (4) multiply by $V$ — Mock Q26 |
| What gets the weighted sum: keys, queries, or values? | **Values** — keys are only for scoring; values are what's actually retrieved |

## Related

- [[attention]] — the parent concept; also covers Bahdanau-style classical attention
- [[self-attention]] — uses scaled dot-product over the same sequence
- [[multi-head-attention]] — multiple parallel scaled dot-product attentions
- [[softmax]] — the normalization step
