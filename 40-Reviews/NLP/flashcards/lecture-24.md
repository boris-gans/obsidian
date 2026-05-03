---
tags: [flashcards, NLP]
---

# Lecture 24 — Challenges and Trends (flashcards)

What does a transformer compute, formally?
?
$p_\theta(w_1, \ldots, w_T) = \prod_{t=1}^{T} p_\theta(w_t \mid w_1, \ldots, w_{t-1})$ — a joint distribution over sequences, factorized as a product of next-token conditionals.

In what sense is a transformer's mechanism "invariant across tasks"?
?
The model only **produces the next token** according to a learned distribution. Classification, QA, summarization etc. are not built into the model — they emerge from how we **interpret the output** and **constrain the input/output format**.

How does classification work as generation?
?
Define a finite label set $\mathcal{Y} = \{y_1, \ldots, y_k\}$ and pick $\hat{y} = \arg\max_{y_i \in \mathcal{Y}} p_\theta(y_i \mid x)$. The model generates a token; we restrict the admissible output set to labels.

How does question answering work as conditional generation?
?
Input is concatenated context $c$ + question $q$; the model produces $\hat{a} = \arg\max_a p_\theta(a \mid q, c)$. **No explicit retrieval or reasoning** — the model just continues the sequence in a way statistically aligned with training patterns.

How does summarization work as a sequence transformation?
?
$\hat{s} = \arg\max_s p_\theta(s \mid x)$ where $s$ is a shorter sequence. **Compression isn't explicitly enforced** — it emerges from training data where summaries are shorter than inputs.

What are the structural limits of internal-only language models?
?
(1) Knowledge **frozen in parameters** — updating requires retraining. (2) **No access to external sources** at inference. (3) Cannot distinguish updated/outdated or certain/uncertain. (4) Outputs only reflect training patterns — no structured / verifiable fact storage.

What does RAG stand for and what does it do?
?
**Retrieval-Augmented Generation** — pairs a generative transformer with a retrieval step. Given a query, retrieve relevant documents from an external corpus, then condition generation on both the query and the retrieved documents (mock Q22).

What are the two components of a RAG system?
?
(1) **Retrieval** — geometric: embed the query, find top-$k$ documents in an external corpus by similarity. (2) **Generation** — probabilistic: transformer language model conditioned on the query plus retrieved docs.

How is the retrieval function typically implemented?
?
Embed query $x$ as $\phi(x) \in \mathbb{R}^n$, embed each doc $d$ as $\phi(d)$, and rank by similarity (usually cosine): $d_i = \arg\max_{d \in \mathcal{D}} \mathrm{sim}(\phi(x), \phi(d))$. Top-$k$ are returned.

How are retrieved documents fed into the transformer in RAG?
?
**Concatenated** with the query as the model's input: $x' = [x; d_1; d_2; \ldots; d_k]$. The model processes $x'$ as a single extended sequence.

Write the RAG prediction equation.
?
$\hat{y} = \arg\max_y p_\theta(y \mid x, R(x))$ where $R(x) = \{d_1, \ldots, d_k\}$ is the set of retrieved documents.

What problem does RAG NOT solve?
?
**Finite context length.** Retrieved documents must still fit inside the model's context window. Self-attention is $O(T^2)$, so very long retrieved contexts dilute attention and increase compute.

What's the complexity of self-attention in context length $T$?
?
**$O(T^2)$** — every pair of positions computes a score. Doubles when $T$ doubles, but memory/compute grows with $T^2$.

What are LCMs (Large Context Models)?
?
Models that extend the context window from $T$ to $T' \gg T$ (e.g. 100K, 1M tokens). Capacity for long-range dependencies grows, but the problem **shifts from access to allocation of attention** — relevance must be selected within a much larger pool.

In a RAG system, where does the "knowledge" live?
?
**In the external corpus**, not in the model parameters. The model becomes a "reader" that synthesizes retrieved text. This is what makes RAG **updatable without retraining** — just add documents to the corpus.

What's an "agentic system" ([[30-Sources/NLP/pdf/Session 24 - Challenges and futures.pdf#page=17|slide 17]], beyond exam scope)?
?
An iterative loop where the model is **applied repeatedly** while interacting with an evolving state — possibly with memory, external tools, or intermediate reasoning. Output emerges through successive transformations rather than a single forward pass.

In the unified view of transformers, what determines whether a model is doing "classification" vs "summarization"?
?
**The input prompt format and the output decoding rule.** The model's mechanism is identical — both produce next tokens according to a learned distribution. Classification restricts admissible outputs to a label set; summarization expects a shorter sequence aligned with summary patterns from training data.

What's the unifying idea behind treating QA, classification, and summarization as the "same" task?
?
All are **next-token generation under different input/output constraints**. The model never knows what "task" it's doing — the task lives in the data format, not the architecture. This is why a single pretrained transformer can serve many tasks.
