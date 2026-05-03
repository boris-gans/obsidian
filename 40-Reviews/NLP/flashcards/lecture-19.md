---
tags: [flashcards, NLP]
---

# Lecture 19 — Transformers (flashcards)

What problem does the attention mechanism solve in encoder-decoder?
?
The **fixed-length context vector bottleneck** — replacing it with a context that's recomputed at every decoder step as a weighted sum of all encoder hidden states. No information is lost in compression.

Write the attention context formula.
?
$c_t = \sum_{i=1}^{T} \alpha_{ti}\, h_i$ — weighted sum of encoder hidden states. Weights $\alpha_{ti}$ measure relevance of input position $i$ for output step $t$.

How are attention weights computed (classical Bahdanau)?
?
$\mathrm{score}_{ti} = s_t^\top h_i$, then $\alpha_{ti} = \mathrm{softmax}_i(\mathrm{score}_{ti})$ — softmax over input positions for each output step.

What are Q, K, V in the database analogy?
?
**Keys $K$**: encoder hidden states used to compute scores. **Values $V$**: encoder hidden states used in the weighted sum. **Queries $Q$**: decoder hidden states issuing the lookup. The encoder is the key-value store; the decoder queries it.

Write the scaled dot-product attention formula (formula sheet).
?
$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right) V$ where $Q = XW_Q$, $K = XW_K$, $V = XW_V$.

Why scale by $\sqrt{d_k}$?
?
Without scaling, dot products grow with $d_k$ → softmax saturates → gradients vanish. Dividing by $\sqrt{d_k}$ keeps the variance roughly constant so softmax stays well-behaved across model sizes.

What's the procedure for computing attention by hand on small matrices?
?
(1) $S = QK^\top$. (2) Scale: $S' = S / \sqrt{d_k}$. (3) Row-wise softmax to get $\alpha$. (4) Output: $\alpha V$.

What is self-attention?
?
Attention applied **within the same sequence** — every token computes its own $Q, K, V$ from learned projections of itself, and attends to all tokens (including itself). Lets transformers process all positions in parallel without recurrence.

Why are transformers parallelizable but RNNs not?
?
Self-attention computes all positions **simultaneously** — there's no $h_t$ that depends on $h_{t-1}$. RNNs are strictly sequential.

What's the time complexity of self-attention in sequence length $L$?
?
**$O(L^2)$** — every pair of positions computes a score. Quadratic in $L$, the main scaling bottleneck of transformers.

Why do transformers need positional encoding?
?
Self-attention is **permutation-invariant** — without explicit position information, the model treats input as a set, not a sequence. "Dog bites man" would equal "Man bites dog". Positional encoding $p_i$ added to token embeddings restores order.

Common false-statement trap: "transformers process tokens strictly sequentially" — true or false?
?
**FALSE.** Transformers process tokens in **parallel**. Sequential processing is what they replaced (RNNs). Positional encoding gives them order without recurrence.

What is multi-head attention?
?
Multiple parallel attention computations, each with **separate learned $W_Q, W_K, W_V$ projections**. Each head captures a different relational subspace; outputs are concatenated and projected back to the model dimension.

Why use multi-head instead of a single big attention head?
?
Different heads can specialize in different relationships (subject-verb, coreference, syntactic dependence). Splitting the model dimension across $h$ heads gives diverse views without increasing total compute much.

What does causal masking do?
?
In the decoder's masked self-attention, prevents each position from attending to **future tokens** — enforces autoregressive generation without recurrence. Implemented by setting score to $-\infty$ for forbidden positions before softmax.

Why is causal masking needed during training?
?
Transformers process all positions in parallel. Without masking, the decoder would see future tokens during training and learn to copy them. The mask forces each position to predict its next token using only the past.

What is cross-attention?
?
A decoder layer where **queries come from the decoder** and **keys/values come from the encoder output**. It's how the decoder accesses the source while generating each target token.

Where does cross-attention sit in a transformer decoder block?
?
Between the **masked self-attention** sublayer and the **feed-forward** sublayer. Each decoder block: masked self-attn → cross-attn → FFN.

Which transformer family models use cross-attention?
?
**Encoder-decoder** ones: BART, T5, the original transformer. **NOT** BERT (encoder-only) or GPT (decoder-only).

Name the three transformer family branches.
?
**BERT** (encoder-only, 2018, understanding tasks). **GPT** (decoder-only, 2018, generation). **BART** (encoder-decoder, 2020, both).

What's BERT good at, and what's its limitation?
?
**Good at**: classification, sentence similarity, **extractive QA** — bidirectional attention gives rich contextual embeddings. **Limitation**: cannot directly generate text without modifying the architecture.

What's GPT good at, and what's its limitation?
?
**Good at**: chat, creative writing, code completion — autoregressive generation. **Limitation**: can't access future context, may miss dependencies crucial for deeper understanding. Its understanding is **statistical, not semantic** ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=18|slide 18]]).

What is extractive QA?
?
The task of selecting an **answer span from the context** that answers a question. Output is a (start, end) pair of token indices; the answer is always a contiguous chunk of the original text (mock Q19).

What's the canonical HuggingFace QA pipeline (memorize cold)?
?
```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
model_ckpt = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
inputs = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits) + 1
answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx])
```

What does `return_tensors="pt"` do?
?
Tells the tokenizer to return inputs as **PyTorch tensors** (not Python lists or NumPy arrays) — required to feed directly into `model(**inputs)`.

Why `with torch.no_grad():` for inference?
?
Disables autograd graph construction — saves memory and speeds up the forward pass. Always use for inference where backprop isn't needed.

What are `outputs.start_logits` and `outputs.end_logits`?
?
**Per-token scores** predicted by the QA head — one for the position being the **start** of the answer span, one for being the **end**. Take argmax of each to get span endpoints.

Why `+ 1` on the end index in extractive QA?
?
Python slicing is **end-exclusive**. To **include** the predicted end token in the answer, we slice up to `end_idx + 1`.

What's the input format for BERT extractive QA?
?
`[CLS] question [SEP] context [SEP]` — question and context concatenated with special tokens, tokenized together.

What does the attention mask encode?
?
**1 for real tokens, 0 for padding tokens.** Tells the model to ignore padded positions when computing attention.

What does increasing temperature do to the softmax distribution?
?
**Flattens** it — probabilities spread out, sampling becomes more diverse / random. Lower temperature **sharpens** the distribution toward argmax.

What's the relationship between beam search and greedy decoding?
?
**Greedy = beam search with beam size 1.** Larger beams maintain top-$k$ partial sequences ranked by joint probability — better quality at higher compute cost.

What does top-$k$ sampling do?
?
Keeps only the **$k$ most probable tokens**, renormalizes their probabilities, and samples from that truncated distribution — filters out the unlikely long tail while preserving diversity.

What's the embedding-matrix size for vocabulary $V$ and embedding dimension $d$?
?
$V \times d$ — one row per vocabulary word, $d$ columns. This is on the formula sheet implicitly via the lookup $x_w = E[w]$.

For GPT-3 with 96 layers, 96 attention heads, hidden size 12,288, what's the rough parameter count?
?
**~175 billion parameters.** (BERT base: 12 layers, 12 heads, hidden 768 → 110M. GPT-3 is much larger.)

Mock Exercise 1 procedure: given Q, K, V as 2×2 matrices, compute attention end-to-end.
?
(1) $S = QK^\top$ (2×2). (2) $S' = S / \sqrt{d_k}$. (3) Row-wise softmax → $\alpha$. (4) Output: $\alpha V$ (2×2). Watch out for: forgetting $\sqrt{d_k}$, wrong softmax axis (must be row-wise), wrong matmul order ($\alpha V$, not $V \alpha$).
