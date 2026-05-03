---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 19 - Transformers-1.pdf
created: 2026-05-02
---

# Multi-head attention

Instead of a single [[scaled-dot-product-attention|attention]] computation, **multi-head attention** runs **multiple attention heads in parallel**, each with its own learned $W_Q, W_K, W_V$ projections. Each head captures a different relational subspace; their outputs are concatenated and projected back to the model dimension ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=17|slide 17]]).

The blueprint flags this as **medium weight**: mock Q17 (multi-head as separate Q/K/V projections) and Quiz IV Q11, Q11.B test the rationale and the projections.

## The idea

A single attention head produces one weighted blend of $V$ vectors per query — one "view" of the sequence. **Different linguistic relationships** (subject-verb agreement, coreference, syntactic dependence, semantic similarity) might be best captured by different attention patterns. Running **multiple heads** with **independent projections** lets each head specialize.

## Mechanics

For $h$ heads, each with head dimension $d_k = d_{\mathrm{model}} / h$:

1. **Per head:** project input $X$ with **separate** $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$
2. **Per head:** compute $\mathrm{Attention}(Q^{(i)}, K^{(i)}, V^{(i)})$ via [[scaled-dot-product-attention]]
3. **Concatenate** all heads' outputs along the feature dimension
4. **Project** back via $W_O \in \mathbb{R}^{(h \cdot d_k) \times d_{\mathrm{model}}}$:
$$\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) \, W_O$$

The total computation is roughly the same as a single $d_{\mathrm{model}}$-dimensional attention, but the result is **richer** because each head learns to attend differently.

## Why multiple heads

> "Multi-head attention: each head uses **separate** Q/K/V projections, captures different relational subspaces; outputs concatenated then projected." (blueprint, "what the formula sheet does NOT provide")

Empirically, different heads pick up different roles:
- One head might track positional adjacency
- Another might track subject-verb relationships
- Another might track entity coreference

The projections $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ project the input into different *subspaces* of $\mathbb{R}^{d_k}$, so each head sees a different "version" of every token.

## Architecture sizes ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=20|slide 20]])

| Model | Heads | Hidden size | Per-head dim |
|---|---|---|---|
| BERT base | 12 | 768 | 64 |
| GPT-3 | 96 | 12,288 | 128 |
| ViT-Huge | 16 | 1,280 | 80 |

## Exam framing

| Question | Answer |
|---|---|
| What's the key property of multi-head attention compared to single-head? | Each head uses **separate** $W_Q, W_K, W_V$ projections (mock Q17, Quiz IV Q11) |
| Why use multiple heads instead of one larger head? | Each head captures a **different relational subspace** of the input — diverse attention patterns instead of one |
| What happens to the outputs of the different heads? | **Concatenated** along the feature dimension and projected through $W_O$ to return to model dimension |
| What's the total parameter count for $h$ heads with per-head dim $d_k$? | Roughly the same as a single $d_{\mathrm{model}}$-dim attention, since $d_{\mathrm{model}} = h \cdot d_k$ |

## Related

- [[scaled-dot-product-attention]] — what each head computes individually
- [[self-attention]] — the typical context for multi-head attention
- [[transformer]] — the architecture that uses it everywhere
- [[cross-attention]] — also multi-head in transformers (queries from decoder, K/V from encoder)
