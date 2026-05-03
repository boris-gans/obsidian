---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 19 - Transformers-1.pdf
created: 2026-05-02
---

# Self-attention

**Self-attention** is [[scaled-dot-product-attention|attention]] applied **within a single sequence** — every token computes its own $Q, K, V$ from learned projections, and attends to all other tokens (and itself) in the same sequence ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=13|slide 13]]). This is the key architectural move that lets [[transformer|transformers]] **remove the RNN entirely** and process all positions in parallel.

The blueprint flags self-attention as **high weight**: Quiz IV Q8, Q10, Q18 (and B variants) target the parallelism, V-vector role, and quadratic complexity. Mock Q24's false-statement trap on transformers builds on self-attention's parallel nature.

## The shift ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=13|slide 13]])

> "We can completely remove the RNNs from the architecture and replace them by simpler FeedForward networks if we just use two different attention cells."

In the encoder-decoder + attention setup of [[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=5|slides 5–12]], the encoder was still an RNN — attention only replaced the fixed-context vector. Self-attention takes the next step: the encoder itself becomes a stack of **self-attention + feed-forward** blocks, with no recurrence at all.

## How it works ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=14|slide 14]], encoder side)

**Input:** each token $i$ has an embedding $e_i$ plus a positional encoding $p_i$:
$$h_i^{(0)} = e_i + p_i$$

**Project** with learned matrices $W_q, W_v, W_k \in \mathbb{R}^{d_k \times d_{\mathrm{model}}}$:
$$q_i = W_q h_i, \qquad v_i = W_v h_i, \qquad k_i = W_k h_i$$

**Score** every pair via dot products: $\mathrm{score}_{ij} = q_i \cdot k_j$, scaled by $\sqrt{d_k}$.

**Weights** via row-wise softmax: $\alpha_{ij} = \mathrm{softmax}_j(\mathrm{score}_{ij})$.

**Context** for each token $i$: $c_i = \sum_j \alpha_{ij} v_j$ — a weighted blend of all tokens' values, with weights determined by how much token $i$ "attends to" token $j$.

**Wrap with residual + LayerNorm + FFN:**
$$h_{\mathrm{out}} = \mathrm{LayerNorm}\!\left(\mathrm{LayerNorm}(h_i + c_i) + \mathrm{FFN}(\mathrm{LayerNorm}(h_i + c_i))\right)$$

The output is a **sequence of contextualized hidden states** — not a single vector ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=16|slide 16]]).

## Three core properties ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=16|slide 16]])

| Layer | $Q, K, V$ | Behavior |
|---|---|---|
| **Encoder self-attention** | All from encoder | Each token attends to all other tokens — output is a sequence of contextualized states |
| **Decoder masked self-attention** | All from decoder | Each position attends only to **previous** target tokens — enforces autoregressive generation **without recurrence** |
| **Cross-attention** (decoder) | $Q$ from decoder; $K, V$ from encoder output | Decoder "looks at" the encoder. The "context" is a sequence, not a global fixed-length summary |

## Why self-attention beats RNN

| Property | RNN | Self-attention |
|---|---|---|
| Sequence dependency | Sequential — $h_t$ requires $h_{t-1}$ | **Parallel** — all positions simultaneously |
| Long-range dependencies | Vanishing gradients limit ~5–10 tokens | **Direct** $O(1)$ path between any two positions |
| Token order | Built into recurrence | Requires explicit [[positional-encoding]] |
| Compute complexity per layer | $O(L \cdot d^2)$ | $O(L^2 \cdot d)$ — **quadratic in sequence length** |

The trade-off: self-attention is **quadratic in sequence length** $L$ (slide blueprint, Quiz IV Q10) — for very long sequences, this becomes the bottleneck (motivating sparse / linear attention variants beyond this course).

## Self-attention vs cross-attention ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=15|slide 15]])

In a transformer **decoder block**, both appear:
- **Masked self-attention** — $Q, K, V$ all from the decoder; mask prevents attending to future positions
- **Cross-attention** — $Q$ from the decoder; $K, V$ from the encoder output

The masking is what turns self-attention into an autoregressive operation: position $i$ can only attend to positions $\le i$ during training, so the same architecture can generate one token at a time at inference (see [[causal-masking]]).

## Exam framing

| Question | Answer |
|---|---|
| What does "self" in self-attention mean? | $Q$, $K$, $V$ all come from **the same sequence** — each token attends to every token in its own sequence |
| What's the complexity of self-attention in sequence length $L$? | **$O(L^2)$** — every pair of positions computes a score (Quiz IV Q10) |
| Why can self-attention be parallelized? | All token positions are processed simultaneously — no sequential dependency on previous hidden states (Quiz IV Q8) |
| Why does self-attention need positional encoding? | It's permutation-invariant — without $p_i$, "dog bites man" and "man bites dog" produce the same output (mock Q16) |
| What's the difference between encoder and decoder self-attention? | Encoder: each token attends to all others (bidirectional). Decoder: **masked** — each position attends only to previous tokens (causal) |
| Where do the $V$ vectors come from in encoder self-attention? | From learned linear projections $W_v$ of the same input embeddings ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=14|slide 14]], Quiz IV Q8) |

## Related

- [[scaled-dot-product-attention]] — the formula self-attention uses
- [[multi-head-attention]] — multiple parallel self-attention heads with separate $W_Q/W_K/W_V$
- [[causal-masking]] — what makes decoder self-attention autoregressive
- [[positional-encoding]] — required to give self-attention order information
- [[transformer]] — the architecture built on stacks of self-attention layers
- [[cross-attention]] — when $Q$ and $K, V$ come from different sequences
