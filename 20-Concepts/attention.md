---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 19 - Transformers-1.pdf
created: 2026-05-02
---

# Attention mechanism

The **attention mechanism** lets a model dynamically focus on different parts of an input sequence when making predictions, replacing the fixed-length context vector of [[encoder-decoder|encoder-decoder]] systems with a context that's **recomputed at every decoder step** as a weighted sum of all encoder hidden states ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=5|slide 5]]).

The blueprint flags this as **very high weight**: mock Q14, Q15 (attention scores, weights, output) and **mock Q26 = Exercise 1, 10 pts: attention computation by hand on 2×2 matrices**. Quiz IV Q6–Q9, Q19 (and B variants) drill the formulas. The full attention machinery is on the formula sheet.

## The core idea ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=7|slide 7]])

> "The attention mechanism creates a single vector of fixed length by taking the **weighted sum of all the encoder hidden states**:
> $$c_t = \sum_{i=1}^{T} \alpha_{ti}\, h_i$$
> where the weights $\alpha_{ti}$ reflect the relevance of each input position for the current output step."

Crucially, the context vector is **generated anew with every decoding step** and takes all of the encoder's hidden states into the computation — no compression bottleneck ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=7|slide 7]]).

## Why it matters ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=4|slides 4–6]])

The basic encoder-decoder squeezes the entire source into a **single fixed-length vector**. As sequences grow longer, this compression becomes inefficient and loses information ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=4|slide 4]]). Attention fixes this by removing the need for a single global summary.

[[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=6|slide 6]]'s translation intuition: translating "The transformers are great!" → "¡Los transformers son geniales!" requires:
- Knowing whether "The" → "el", "la", "las", or "les" (depends on what noun comes next)
- Inserting "¡" at the start (Spanish-specific punctuation rule)
- Translating "transformers" → "transformadores" *would* be wrong — needs context

A word-by-word translator without context fails. **Attention lets the decoder look back at the relevant source positions** to disambiguate per output token.

## Computation ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=10|slide 10]])

Given encoder hidden states $h_1, \ldots, h_T$ (one per source token) and the current decoder state $s_t$:

1. **Score** each encoder state against the current decoder state:
$$\mathrm{score}_{ti} = s_t^\top h_i$$
2. **Normalize** the scores into attention weights via softmax (one distribution over input positions per output step):
$$\alpha_{ti} = \mathrm{softmax}_i(\mathrm{score}_{ti}) = \frac{e^{\mathrm{score}_{ti}}}{\sum_j e^{\mathrm{score}_{tj}}}$$
3. **Context vector** is the weighted sum of values:
$$c_t = \sum_{i=1}^{T} \alpha_{ti}\, h_i$$

The context $c_t$ is then combined with $s_t$ to produce the next-token prediction ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=11|slide 11]]).

## Output layer ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=11|slide 11]])

Combine decoder state and attention context, project to vocabulary, softmax:
$$o_t = W_o [s_t; c_t] + b_o$$
$$p_t = \mathrm{softmax}(o_t)$$

The model picks the most likely token (or samples from $p_t$).

## Q, K, V interpretation ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=12|slide 12]])

In the attention mechanism, parts have **database-like names**:

| Component | What it is | Where it comes from |
|---|---|---|
| **Keys $K$** | Hidden states used to compute attention scores | Encoder hidden states $h_i$ |
| **Values $V$** | Hidden states used in the weighted sum (the actual content retrieved) | Encoder hidden states $h_i$ (often $K = V$ in this Bahdanau-style formulation) |
| **Queries $Q$** | Hidden vectors that issue the lookup | Decoder hidden states $s_t$ |

> "The encoder works like a key-value store, and the attention mechanism looks the query up in its keys and then returns its own value." ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=12|slide 12]])

In **scaled dot-product attention** ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=14|slide 14]]), these are produced by **separate learned projections** $W_Q, W_K, W_V$ rather than just being raw hidden states — that's the [[scaled-dot-product-attention|transformer-era generalization]].

## Worked example (mock Q26, exam-ready drill)

Inputs: $Q, K, V \in \mathbb{R}^{2 \times 2}$ (or larger). Procedure:

1. **Compute scores:** $S = Q K^\top$ — a matrix of dot products
2. **Scale:** $S' = S / \sqrt{d_k}$ — divide every entry by the square root of the key dimension
3. **Row-wise softmax:** $\alpha_i = e^{S'_i} / \sum_j e^{S'_j}$ — each row sums to 1
4. **Context output:** $\mathrm{Attention}(Q, K, V) = \alpha V$

Each row of the output corresponds to one query — a weighted combination of value rows.

## Exam framing

| Question | Answer |
|---|---|
| What does the attention mechanism replace in encoder-decoder? | The **fixed-length context vector** — replaced by a per-step context computed as a weighted sum of all encoder hidden states ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=5|slide 5]], Quiz IV Q5) |
| How are attention weights computed? | $\mathrm{score}_{ti} = s_t^\top h_i$, then $\alpha_{ti} = \mathrm{softmax}(\mathrm{score}_{ti})$ — softmax of dot products, normalizes over input positions ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=10|slide 10]]) |
| How is the context vector built? | $c_t = \sum_i \alpha_{ti} h_i$ — weighted sum of values, where the weights are the attention probabilities ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=7|slide 7]]) |
| What are Keys, Queries, Values? | $K$ = encoder hidden states (used for scoring), $V$ = encoder hidden states (the content retrieved), $Q$ = decoder hidden states (issuing the lookup) — [[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=12|slide 12]] |
| Why "one context vector per decoder step"? | The context is **dynamic**: at each output token, the decoder selects a different mixture of input positions — [[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=7|slide 7]] |

## Related

- [[scaled-dot-product-attention]] — the standard form with $\sqrt{d_k}$ scaling and learned $W_Q/W_K/W_V$
- [[self-attention]] — attention applied within a single sequence
- [[multi-head-attention]] — multiple parallel attention computations
- [[cross-attention]] — decoder Q attends to encoder K/V
- [[encoder-decoder]] — the architecture attention augments / replaces
- [[softmax]] — used to normalize attention scores
