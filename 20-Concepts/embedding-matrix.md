---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 13 - Word Embeddings.pdf
created: 2026-05-02
---

# Embedding matrix

The **embedding matrix** is the $V \times d$ parameter table that holds one dense vector per vocabulary word ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=10|slide 10]], 208).
$$W \in \mathbb{R}^{V \times d}$$
- $V$ = vocabulary size
- $d$ = embedding dimension (chosen as a hyperparameter, $d \ll V$)
- **Total parameters = $V \cdot d$**

Each **row** is the dense vector for one word. Looking up a word's embedding is just selecting the corresponding row:
$$x_w = E[w]$$
(formula sheet — "embedding lookup").

The blueprint flags the **$V \times d$ count** as a recurring MCQ pattern: Quiz III Q6, Q6.B; Quiz IV Q13, Q13.B.

## Two matrices in Word2Vec

[[word2vec|Word2Vec]] actually trains **two** embedding matrices simultaneously ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=10|slide 10]]):
- $W \in \mathbb{R}^{V \times d}$ — vectors for **centre** words
- $U \in \mathbb{R}^{V \times d}$ — vectors for **context** words

Both are randomly initialized and updated by gradient descent. The Skip-gram softmax uses the dot product $v_c^\top v_w$ where $v_c$ is a row of $U$ (context-side) and $v_w$ a row of $W$ (centre-side). The final exported embedding is usually $W$ alone, or sometimes $\frac{1}{2}(W + U)$.

## Worked example

For $V = 10{,}000$ and $d = 300$:
$$|W| = 10{,}000 \times 300 = 3 \times 10^6 \text{ parameters}$$

This is a ~30× reduction from the [[one-hot-encoding|one-hot]] $\mathbb{R}^V$ representation per word (one-hot is $V$ values per word but with structural redundancy; embeddings are $d$ values per word with semantic content).

## Why it's "the model"

In Word2Vec, the embedding matrix **is** the learned model. There's no deeper neural architecture above it — Word2Vec is a **shallow neural network with one hidden layer** ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=19|slide 19]]), and that hidden layer is exactly the embedding lookup. Once $W$ is trained, you discard the rest of the training apparatus and just use the matrix.

In transformer models (Session 19+), the embedding matrix is just the **input layer** — token IDs are mapped to $\mathbb{R}^d$ vectors that are then transformed by attention layers.

## Exam framing

| Question | Answer |
|---|---|
| Vocabulary $V$, embedding dim $d$ — how many parameters? | $V \cdot d$ (Quiz III Q6, Q6.B; Quiz IV Q13, Q13.B) |
| What does each row of the embedding matrix represent? | The dense vector for one vocabulary word |
| What does the formula sheet entry $x_w = E[w]$ describe? | **Embedding lookup** — selecting the row of $E$ corresponding to word $w$ |
| Why does Word2Vec maintain two embedding matrices? | One for centre words ($W$), one for context words ($U$) — both used in the softmax dot product ([[30-Sources/NLP/pdf/Session 13 - Word Embeddings.pdf#page=10|slide 10]]) |

## Related

- [[word-embeddings]] — the higher-level concept the matrix instantiates
- [[word2vec|Word2Vec]] — the algorithm that learns $W$ via prediction
- [[one-hot-encoding]] — the $\mathbb{R}^V$ baseline embeddings replace
