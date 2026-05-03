---
tags: [flashcards, NLP]
---

# Lecture 17 — Recurrent Neural Networks (flashcards)

What is the central design feature of an RNN that distinguishes it from a feedforward network?
?
A **hidden state** that carries summarized information about all previous tokens forward to the next step. The same parametric cell is applied at every step; the hidden state acts as a learned, fixed-size memory of past context.

Write the Elman RNN update equation (formula sheet form).
?
$h_t = \tanh(W x_t + U h_{t-1})$ — current input $x_t$ combined with previous hidden state $h_{t-1}$, blended through a $\tanh$ nonlinearity.

What does the hidden state $h_t$ encode?
?
A **fixed-size, learned representation of the entire sequence up to time $t$**: $h_t = f(x_1, x_2, \ldots, x_t)$ — a compressed memory of the preceding context.

How many "cells" with distinct parameters are there in an unrolled RNN over $T$ tokens?
?
**One cell**, with one set of weights $U$, $W$, $V$ (and biases). The unrolled picture shows the same cell repeated at each time step — parameters do **not** grow with sequence length.

What's the role of $W$ in an Elman RNN?
?
The **recurrent (hidden-to-hidden) weight matrix** that propagates the hidden state from $h_{t-1}$ to $h_t$. (Note: notation varies — some sources use $W$ for input-to-hidden and $U$ for hidden-to-hidden.)

What does an RNN language model train to maximize?
?
$P(w_{t+1} \mid w_1, \ldots, w_t)$ — probability of the next token given previous tokens. Loss is the negative log-likelihood summed over the sequence.

What's BPTT?
?
**Backpropagation Through Time** — apply standard backpropagation to the unrolled RNN graph. Same parameters appear at every time step, so gradients are summed across time.

Why do simple RNNs suffer from vanishing or exploding gradients?
?
BPTT propagates gradients through the **recurrent weight matrix at every intermediate step**. If its largest singular value is < 1, gradients shrink exponentially (vanishing); if > 1, they grow exponentially (exploding).

What's the practical effective context length of a vanilla Elman RNN?
?
About **5–10 tokens**. Vanishing gradients mean the update of the hidden state depends mostly on the most recent tokens; distant tokens contribute negligibly.

What architectural innovation mitigates vanishing gradients in RNNs?
?
**Gating mechanisms** — LSTM (forget/input/output gates with cell state) and GRU (reset/update gates) protect long-term information through additive paths in the cell state.

Can an RNN be parallelized across the sequence?
?
**No** — each $h_t$ depends on $h_{t-1}$, forcing strictly sequential computation. This is the structural bottleneck that transformers eliminate.

What's a "many-to-one" RNN task?
?
A sequence input → single label output. **Sentiment classification** is the canonical example: process token by token, use the final hidden state to predict the class.

What's a "many-to-many" RNN task with aligned input/output?
?
Per-token output, same length as input. **POS tagging** and **NER** work this way: each input word produces one output label.

What's a "many-to-many encoder-decoder" task?
?
Variable-length input → variable-length output. **Machine translation**: an encoder RNN reads the source sentence into a context vector; a decoder RNN generates the target sentence.

How does a trained RNN generate text?
?
Repeatedly: (1) compute $P(w \mid \text{context so far})$, (2) pick a token (greedy = argmax, sampling = draw from the distribution), (3) append + update hidden state. Continue until **EOS** token.

Why is each token in an Elman RNN treated identically?
?
The same input weights $U$ are applied to every token, and the only path through the model is via the hidden state. There's **no mechanism to weight relevant tokens more than irrelevant ones** — this motivates **attention** in Session 19.

What's the role of $V$ in the Elman RNN?
?
The **output projection matrix**: $o_t = V h_t + b_o$, $\hat{y}_t = \mathrm{softmax}(o_t)$. Maps the hidden state into vocabulary logits for next-token prediction.

What's the standard remedy for exploding gradients in RNN training?
?
**Gradient clipping** — cap the global gradient norm at some threshold (typically 1–5) before the parameter update.

Why does BPTT make training large RNNs computationally expensive?
?
Strictly sequential forward and backward passes — each token requires the previous hidden state. Within a sequence, no parallelism. Transformers fix this by computing all positions simultaneously via self-attention.

In the worked example "I like natural language processing", what does $h_3$ encode?
?
A learned representation of the prefix "I like natural" — the hidden state after the model has read the first three tokens.

What's the training objective for an RNN language model?
?
Minimize the **negative log-likelihood** of the actual next token, summed over the sequence: $L = -\sum_t \log P(w_{t+1} \mid w_{1\ldots t})$.

Why are RNNs called "shared-weight" models when unrolled?
?
The same weight matrices $U$, $W$, $V$ are reused at every time step in the unrolled view. There's only one cell; "unrolled" is just a visualization showing the cell applied $T$ times.
