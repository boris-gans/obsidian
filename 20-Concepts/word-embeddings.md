---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 13 - Word Embeddings.pdf
  - course: NLP
    file: 08_Word_Embeddings.ipynb
created: 2026-05-02
---

# Word embeddings

A **word embedding** represents each word as a **dense vector** $v_w \in \mathbb{R}^d$ in a relatively low-dimensional space ($d \ll V$), where $V$ is the vocabulary size ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=10|slide 10]]). Unlike [[one-hot-encoding|one-hot]] or [[bag-of-words|BoW]] vectors, embeddings are **learned parameters** optimized to minimize prediction error — semantic similarity becomes **closeness in a learned continuous space**.

The blueprint flags this as **high weight**: mock Q23 / Q25 contrast one-hot vs dense; Quiz III Q6, Q6.B, Quiz IV Q13 ask for the embedding-matrix size $V \times d$.

## Sparse → dense: what changes

| Property | One-hot / BoW (in $\mathbb{R}^V$) | Embedding (in $\mathbb{R}^d$, $d \ll V$) |
|---|---|---|
| Dimensionality | $V$ — tens of thousands | $d$ — typically 50–1000 |
| Sparsity | mostly zero | dense (every coordinate matters) |
| Similarity | exact-match only — distinct words are orthogonal | graded — proximity in space (cosine, dot product) |
| Synonymy | "car" ⊥ "automobile" | similar vectors |
| Polysemy | one vector per word (still a problem) | one vector per word *type* — averaged across senses (still a problem) |
| Source | counts | gradient descent on prediction loss |
| Generalization | poor — unseen words have no vector | better — geometry encodes structure |

Mock Q23: dense > one-hot because dense vectors **encode semantic similarity geometrically**. Mock Q25: dense vectors **generalize better** to unseen contexts.

## The embedding matrix

All word vectors are stacked into a single matrix:
$$W \in \mathbb{R}^{V \times d}$$
- $V$ rows, one per vocabulary entry
- $d$ columns, the embedding dimension
- **Total parameters = $V \times d$** ([[embedding-matrix]] is the dedicated note for this MCQ pattern)

Looking up a word's vector is just selecting the corresponding row: $x_w = E[w]$ (formula sheet).

In Word2Vec, two such matrices are typically learned: $W$ for **centre words** and $U$ for **context words**, both $\in \mathbb{R}^{V \times d}$, randomly initialized then updated by gradient descent ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=10|slide 10]]). The final embedding is usually $W$ alone (or the average $\frac{1}{2}(W + U)$).

## Why learned vectors beat counts

The [[distributional-hypothesis]] says words in similar contexts have similar meanings. **Count-based methods** ([[latent-semantic-analysis|LSA]], [[hyperspace-analogue-to-language|HAL]]) operationalize this by collecting statistics first and then decomposing; **embedding methods** ([[word2vec|Word2Vec]]) operationalize it by **predicting context** and letting representations emerge from the prediction loss.

> "If words that appear in similar contexts must produce similar predictions, then their internal representations will converge." ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=11|slide 11]])

Crucially:
- **Geometry is intrinsic** — distances and directions encode statistical regularities of usage
- **Axes have no semantic interpretation** ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=18|slide 18]]) — meaning is distributed across all $d$ dimensions, unlike LSA where axes correspond to variance directions

## What embeddings capture ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=19|slide 19]])

- **Semantic similarity** — words used in similar contexts are pulled together
- **Linear relational structure** — analogies like $v_{king} - v_{man} + v_{woman} \approx v_{queen}$ emerge as approximate vector translations
- **Geometric similarity** computed via [[cosine-similarity]] or dot product

## What embeddings still miss

- **Polysemy** — a single vector per word type. "bank" gets a weighted average of its finance and river senses
- **Context-dependent meaning** — the representation is **static**, fixed at training time

This is the conceptual gap that **transformer architectures** (Session 19) close: contextual embeddings depend on the surrounding sentence.

## Exam framing

| Question | Answer |
|---|---|
| Number of parameters in an embedding matrix for vocabulary $V$, dimension $d$? | **$V \times d$** (Quiz III Q6, Q6.B; Quiz IV Q13) |
| What's the main advantage of dense over sparse representations? | Encodes **semantic similarity** geometrically; better generalization (mock Q23, Q25) |
| Why do similar words end up with similar vectors? | They are **trained to predict similar contexts** — gradient updates pull them together |
| What problem do static embeddings *not* solve? | **Polysemy** — one vector per type, averaged across senses |

## Canonical Skip-Gram-with-Negative-Sampling skeleton (PyTorch)

The notebook builds a minimal Skip-Gram NS model from scratch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramNS(nn.Module):
    """Two embedding tables: in_emb (centre) and out_emb (context)."""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, dim)   # centre word vectors W
        self.out_emb = nn.Embedding(vocab_size, dim)  # context word vectors U

    def forward(self, center, pos_ctx, neg_ctx):
        v = self.in_emb(center)            # [B, d]
        u_pos = self.out_emb(pos_ctx)      # [B, d]
        u_neg = self.out_emb(neg_ctx)      # [B, k, d]

        # log σ(v·u_pos)  +  Σ log σ(-v·u_neg_i)
        pos_loss = F.logsigmoid((v * u_pos).sum(-1))
        neg_loss = F.logsigmoid(-(u_neg @ v.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return -(pos_loss + neg_loss).mean()
```

Reference: `[Skip-Gram NS from scratch (cells 8–11)](30-Sources/NLP/notebooks/08_Word_Embeddings.ipynb)`.

The exam-critical pieces:
- **Two embedding tables** — `in_emb` for centre words ($W$), `out_emb` for context ($U$) — both $V \times d$
- **Loss is `log σ(v_c·v_w) + Σ log σ(-v_n·v_w)`** — exactly the formula sheet form
- The model file *is* the embedding matrix once trained; everything else is training scaffolding
