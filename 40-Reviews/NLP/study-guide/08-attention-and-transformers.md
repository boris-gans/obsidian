---
type: study-guide-cluster
course: NLP
cluster: "08-attention-and-transformers"
theme: "Attention, self-attention, multi-head, transformers"
prerequisites: [05-word-meaning-in-vector-space, 07-neural-sequence-models]
covers-concepts:
  - attention
  - scaled-dot-product-attention
  - self-attention
  - multi-head-attention
  - positional-encoding
  - causal-masking
  - cross-attention
  - transformer
covers-lectures:
  - lecture-19-transformers
exam-weight: high
---

# Cluster 8: Attention and transformers

> **The story of this cluster in one sentence.** Cluster 7's encoder-decoder failed because a single fixed-length context vector can't carry an entire source sentence — attention replaces that bottleneck with a **dynamic, per-step weighted sum over all encoder positions**, and once we have attention we can throw away recurrence entirely (transformers).

## Why this cluster exists

The encoder-decoder bottleneck (Cluster 7's cliffhanger) is an information-flow problem: long sources get compressed past their carrying capacity. Attention's fix is conceptually simple — **let the decoder look at all encoder hidden states and decide for itself which ones matter at this step**. Once that mechanism exists, two further moves become possible: (a) apply attention *within* a single sequence (self-attention) so every token can directly query every other token, and (b) drop the recurrent backbone entirely, leaving an architecture (the transformer) where every layer is just self-attention + feedforward — and every position is computed in parallel.

The blueprint flags this as **very high weight**: **Exercise 1 of the mock is attention by hand on 2×2 matrices** (10 pts); MCQ block covers attention scores / weights / output (mock Q14, Q15), multi-head (mock Q17), positional encoding (mock Q16), the false-statement trap "transformers process tokens strictly sequentially" (mock Q24), Quiz IV Q6–Q11, Q14, Q18–Q20 + B variants. The formula sheet provides **everything mechanical**: `Q = XW_Q`, `K = XW_K`, `V = XW_V`, scores `S = QK^T`, scaled `S/√d_k`, weights `α = softmax(S)`, output `Attention(Q,K,V) = αV`. What's *not* on the sheet: *why* `√d_k`, *why* multi-head beats single-head, *why* positional encoding is needed, *why* transformers parallelize. That's the memorization burden.

**Prerequisites you should feel solid on:**

- [[encoder-decoder]] — the bottleneck this cluster fixes
- [[softmax]] — turns scores into a probability distribution over input positions
- [[cosine-similarity]] — the un-normalized cousin (`Q · K^T`) is the attention score; same template
- [[word-embeddings]] / [[embedding-matrix]] — the input layer is unchanged from Cluster 5; transformers operate on embedding sequences

## The arc

### 1. [[attention]] — the weighted-sum trick

Replace the encoder-decoder's single context vector with a **per-decoder-step context** computed as a weighted sum of *all* encoder hidden states:

`cₜ = Σᵢ αₜᵢ · hᵢ`

where `αₜᵢ = softmax(score(sₜ, hᵢ))` is the attention weight from decoder step `t` to encoder position `i`. The weights are *recomputed at every decoder step*, so the model focuses on different source positions for different output tokens. The original (Bahdanau, 2014) score is `sₜᵀ hᵢ` — a learned dot product. This single mechanism removes the bottleneck without touching the rest of the architecture. **Mock Q14, Q15 + Exercise 1** (10 pts) all live here.

### 2. [[scaled-dot-product-attention]] — the formula-sheet workhorse

The standard transformer-era form **generalizes** Bahdanau's idea by applying **separate learned projections** to produce queries, keys, and values:

`Q = X W_Q,    K = X W_K,    V = X W_V`
`Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V`

The four steps for the exam:

1. **Scores**: `S = Q Kᵀ` (dot products between every query-key pair)
2. **Scale**: divide by `√d_k` — prevents large dot products from saturating softmax
3. **Weights**: row-wise softmax — each row sums to 1
4. **Output**: `α V` — weighted combination of value rows

This is **exactly the procedure for Exercise 1** on small matrices. Drill: given 2×2 `Q, K, V` and `d_k = 2`, produce the 2×2 output. *Why scale by `√d_k`?* Without it, dot products grow linearly with `d_k` and softmax saturates → gradients vanish. With `√d_k` scaling, the variance stays roughly constant across model sizes.

### 3. [[self-attention]] — attention within a single sequence

In Cluster 7's encoder-decoder attention, queries came from the decoder, keys/values from the encoder. **Self-attention** uses the *same* sequence for all three: each token computes its own `Q, K, V` from learned projections of itself, and attends to all tokens (including itself). Result: every position gets a contextually-weighted representation that depends on the *entire sequence*. Two huge consequences:

- **Parallelism**: every position is computed independently — no `hₜ → hₜ₊₁` dependency. Transformers train *much* faster than RNNs.
- **Quadratic complexity**: every pair of positions computes a score → `O(L²)` time and memory in sequence length. The fundamental scaling bottleneck of vanilla transformers.

Quiz IV Q8, Q10, Q18 (and B) test these two facts head-on.

### 4. [[multi-head-attention]] — multiple parallel attentions

A single attention head learns one notion of relatedness. **Multi-head attention** runs `h` parallel attention computations with **separate `W_Q, W_K, W_V` projections** in each head, then concatenates the outputs and projects back to the model dim. Each head can specialize: one captures syntactic dependencies, another captures coreference, etc. Mock Q17 tests this: "why multi-head over single big head?" → *separate projection subspaces capture different relational patterns*. Quiz IV Q11 (and B) hits the same.

### 5. [[positional-encoding]] — restoring word order

Self-attention is **permutation-invariant**: shuffle the input sequence and the output is the same shuffle. *"Dog bites man"* and *"Man bites dog"* would be identical to a positional-blind transformer. **Positional encoding** adds a position-dependent vector `pᵢ` to each token embedding, restoring order information. Two flavors: **sinusoidal** (the original transformer's hand-designed encoding) and **learned** (modern variants). Mock Q16 / Quiz IV Q14 (and B) ask "why are positional encodings necessary?" — answer: self-attention has no built-in notion of position, so order has to be injected through the input.

### 6. [[causal-masking]] — preventing attention to the future

Decoders generate tokens left-to-right. To train them in parallel, you compute attention over the *whole* output sequence at once — but each position must only see *earlier* positions, never future ones. **Causal masking** sets the attention scores `Sᵢⱼ` to `-∞` for `j > i` *before* the softmax, so the future tokens contribute zero weight. This is what lets GPT-style decoders be trained as a single forward pass and still generate autoregressively at inference. Quiz IV Q12 / Q20 + B.

### 7. [[cross-attention]] — decoder queries, encoder keys/values

In an encoder-decoder transformer, the decoder uses **two** attention mechanisms per layer: (a) **causal self-attention** over the partial output (so far), and (b) **cross-attention** where queries come from the decoder and keys/values come from the encoder output. Cross-attention is the *direct generalization* of Cluster 7's encoder-decoder attention to the multi-head, scaled-dot-product form. Quiz IV Q20 / B asks for the directional setup: "queries from decoder, keys+values from encoder."

### 8. [[transformer]] — putting it all together

The transformer (Vaswani 2017) stacks **N identical encoder layers** (each = self-attention + feedforward + residual + layer-norm) and, in encoder-decoder variants, **N identical decoder layers** (each = causal self-attention + cross-attention + feedforward + residuals + layer-norms). Inputs go through an embedding lookup + positional encoding before entering layer 1. Output is logits over the vocabulary, softmaxed for the next token (in generation) or the answer span (in QA). The classic false-statement trap (mock Q24): "transformers process tokens strictly sequentially" → **FALSE**, they process in parallel; sequential is what they replaced. Quiz IV Q19 tests "what attention computes between tokens."

## Connections worth seeing

- **Attention scores `Q · Kᵀ` are dot products — the same operation as cosine similarity (Cluster 3).** The difference: cosine *normalizes* by vector magnitudes; attention does not (it's then softmax-normalized). The same template — *dot product as similarity score* — runs from TF-IDF retrieval (Cluster 3) through Word2Vec's training objective (Cluster 5) to attention here.
- **Attention is the continuous, soft cousin of HMM Viterbi.** Viterbi (Cluster 6) does dynamic programming over discrete states with `max` — a hard pick of one predecessor. Attention does a `softmax`-weighted blend — a soft pick over all positions. *Same question* (which previous position matters now?), *different reduction operator* (max vs. softmax). And the max-vs-sum swap from Cluster 6 (forward → Viterbi) is the same kind of move as softmax-vs-argmax here.
- **Self-attention is parallel by design; RNN is sequential by design.** Same end goal — produce a contextual representation per position — but RNN threads `hₜ → hₜ₊₁` and self-attention computes all positions independently. This single architectural difference is *why* transformers can train on the entire internet and RNNs can't.
- **Multi-head attention is doing what convolutional filters do for vision.** Each head/filter learns to detect a different pattern in the input; their outputs are combined for the next layer. The pattern — *learn diverse features in parallel, then merge* — is one of the great recurring designs in deep learning.
- **The `softmax(QKᵀ / √d_k)` step is exactly Cluster 4's softmax classifier**, but applied row-wise to attention scores instead of class logits. *"Which one is most relevant?"* is structurally the same question as *"which class is this?"*.

## Common confusions

- **Q, K, V in encoder-decoder attention vs. self-attention** — encoder-decoder: Q from decoder, K/V from encoder. Self-attention: Q, K, V all from the same input via *separate* learned projections.
- **Why scale by `√d_k` and not `d_k`?** Variance of `Q · K` grows linearly with `d_k`; std grows with `√d_k`. Dividing by `√d_k` keeps variance constant.
- **Self-attention is NOT permutation-invariant once positional encodings are added** — without them it is, which is why they're mandatory.
- **Causal masking vs. padding masking** — causal masking blocks attention to *future* positions (decoder); padding masking blocks attention to padding tokens (any input with variable length). Both are MCQ-tested.
- **Multi-head vs. multi-layer** — multi-head: parallel attentions in *one* layer. Multi-layer: stacked attention layers. A transformer typically has both: `N = 12` layers, each with `h = 12` heads.
- **Transformer = parallel.** The mock Q24 false-statement trap is exactly this. Memorize: **transformers process tokens in parallel; positional encoding gives them order without recurrence.**

## Self-check (synthesis, not recall)

1. **Attention computes `softmax(Q · Kᵀ / √d_k) · V`.** Identify the *exact* parts of this formula that fix Cluster 7's encoder-decoder bottleneck. Which step *removes* the fixed-context vector, and which step replaces it?
2. **Why does multi-head attention work better than a single attention with the same total parameter count?** Answer in terms of *what each head can specialize in* vs. what a single bigger head is forced to entangle.
3. **Self-attention is `O(L²)` in sequence length.** Pick one downstream consequence: training cost on long documents, deployment latency, model architecture choices for long-context language models. What's the trade-off researchers make?
4. **Without positional encoding, "the dog bit the man" and "the man bit the dog" produce identical transformer outputs.** Construct a sentence pair where the loss of position would *not* matter much — and one where it changes the meaning entirely. What does this say about *which tasks* most need positional information?
5. **Causal masking lets a decoder be trained in parallel** despite being autoregressive at inference. What would training look like *without* causal masking? Why would that be incompatible with how the model is used at inference time?
6. **Looking back to Cluster 6:** Viterbi picks one predecessor with `max`; attention picks a soft blend with `softmax`. Construct a task where the discrete pick is fundamentally what you want (Viterbi is right), and a task where the soft blend is what you want (attention is right).
7. **Looking forward to Cluster 9:** the transformer is now the substrate for almost every modern NLP application. What architectural property — parallelism, long context, contextual embeddings, or pre-training scalability — best explains *why*?

## If you have 10 minutes

1. [[scaled-dot-product-attention]] — the four-step procedure: scores → scale → softmax → weighted sum. Drill on a 2×2 by hand.
2. [[self-attention]] — parallelism + `O(L²)` complexity + permutation-invariance (hence positional encoding)
3. [[multi-head-attention]] — separate Q/K/V projections per head; concatenated then projected
4. The false-statement trap: **transformers do NOT process tokens sequentially** (mock Q24)
5. [[positional-encoding]] — *why* it exists (self-attention is permutation-invariant)

## Next cluster

→ [[09-modern-applications]] — having built the transformer, the rest of NLP becomes "what do you fine-tune it to do?" Cluster 9 walks through the canonical applications — sentiment classification with HF, generation (temperature / beam / top-k), summarization, extractive QA — including **the Code 2 fill-in-blanks question on the HuggingFace QA pipeline** (10 pts). It also names what's still hard (RAG, hallucinations, long context).
