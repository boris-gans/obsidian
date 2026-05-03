---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 19 - Transformers-1.pdf
created: 2026-05-02
---

# Positional encoding

A vector $p_i$ added to each token's embedding to inject **order information** into a [[transformer]] — needed because [[self-attention]] is **permutation-invariant** and would otherwise treat the input as a *set* of tokens, not a sequence.

The blueprint flags this as **medium weight**: mock Q16 (false-statement trap on transformer sequentiality) and Quiz IV Q14, Q14.B test the rationale.

## The input pipeline ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=14|slide 14]], [[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=17|slide 17]])

$$h_i^{(0)} = e_i + p_i$$
- $e_i$ — token embedding (looked up from the [[embedding-matrix]])
- $p_i$ — positional encoding for position $i$
- Sum produces the **first-layer input** to the transformer

The token embedding tells the model **which** token is at this position; the positional encoding tells the model **where** in the sequence this position is.

## Why it's necessary

[[self-attention]] computes attention over all pairs of positions. The dot product $q_i \cdot k_j$ doesn't depend on $i$ or $j$ as integers — only on the **content** of the vectors $q_i, k_j$. Without a positional encoding, swapping any two tokens' positions would produce the same output: the model could not tell "dog bites man" from "man bites dog".

> Mock Q24's false-statement trap on transformers says "transformers process tokens **strictly sequentially**" — that's **FALSE**. They process in parallel; **positional encoding restores order**.

## What positional encodings look like

The deck states only that $p_i$ is **added to the token embedding** — it doesn't specify the form ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=14|slides 14]], 324 show "Positional embeds" in the input pipeline diagram). Two common choices [not in source — supplementary]:

- **Sinusoidal** (Vaswani et al. 2017 original transformer): each dimension uses a sine or cosine of a different frequency, $p_{i, 2k} = \sin(i / 10000^{2k/d})$, $p_{i, 2k+1} = \cos(\ldots)$. Allows the model to extrapolate to unseen sequence lengths.
- **Learned** (BERT, GPT): a learned embedding per position, treated as another parameter table.

For exam purposes, the **rationale** matters far more than the formula: positional encoding exists because self-attention is permutation-invariant.

## Exam framing

| Question | Answer |
|---|---|
| Why do transformers need positional encoding? | Because **self-attention is permutation-invariant** — without $p_i$, the model treats the input as a set, not a sequence (mock Q16, Quiz IV Q14) |
| What's the input to the first transformer layer? | $h_i^{(0)} = e_i + p_i$ — token embedding **plus** positional encoding ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=14|slide 14]]) |
| What does positional encoding restore? | **Token order** — which gives the model the information that "dog bites man" ≠ "man bites dog" |
| Is positional encoding learned or fixed? | Both options exist — sinusoidal (fixed) or learned. The deck doesn't specify. |

## Related

- [[self-attention]] — what makes positional encoding necessary
- [[transformer]] — the architecture that uses it
- [[word-embeddings]] / [[embedding-matrix]] — what positional encoding is added to
