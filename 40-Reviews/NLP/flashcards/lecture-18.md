---
tags: [flashcards, NLP]
---

# Lecture 18 — Beyond Vanilla RNNs (flashcards)

What's the core idea of "gating" in an RNN?
?
Introduce learnable coefficients in $[0, 1]$ (computed via sigmoid) that act as **element-wise switches** controlling how much information flows through. They turn the recurrence into a controlled dynamical system — the model learns **what to forget, retain, or update**.

How many gates does an LSTM have?
?
**Three**: forget gate $f_t$, input gate $i_t$, output gate $o_t$.

What does the LSTM forget gate do?
?
$f_t = \sigma(W_f x_t + U_f h_{t-1})$ — element-wise multiplies $c_{t-1}$ to control **what to erase** from the cell state. $f_t \approx 0$ → forget; $f_t \approx 1$ → keep.

What does the LSTM input gate do?
?
$i_t = \sigma(W_i x_t + U_i h_{t-1})$ — element-wise multiplies the candidate $\tilde{c}_t$ to control **what new information to write** into the cell state.

What does the LSTM output gate do?
?
$o_t = \sigma(W_o x_t + U_o h_{t-1})$ — element-wise multiplies $\tanh(c_t)$ to control **what to expose** as the next hidden state $h_t$.

Write the LSTM cell-state update.
?
$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ — gated copy of the previous cell state plus a gated candidate update. The additive path is what protects gradients across long sequences.

Write the LSTM hidden-state update.
?
$h_t = o_t \odot \tanh(c_t)$ — the output gate selects what part of the cell state to expose.

Why do LSTMs handle long-range dependencies better than vanilla RNNs?
?
The cell state update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ has an **additive path** through time. When $f_t \approx 1$, $c_t \approx c_{t-1}$ — gradients flow back across many steps without exponential decay (no vanishing gradient).

How many gates does a GRU have?
?
**Two**: reset gate $r_t$ and update gate $z_t$.

What's the role of the GRU update gate $z_t$?
?
Mixes old and new hidden state via $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$. $z_t \to 0$ keeps $h_{t-1}$; $z_t \to 1$ replaces with the candidate. It plays the combined role of LSTM's forget + input gates.

What's the role of the GRU reset gate $r_t$?
?
Controls how much of the previous hidden state enters the candidate computation: $\tilde{h}_t = \tanh(W x_t + U(r_t \odot h_{t-1}))$. $r_t \to 0$ ignores past info when forming the candidate.

Does GRU have a separate cell state?
?
**No** — the hidden state $h_t$ itself carries the information. This is the main structural simplification compared to LSTM.

GRU vs LSTM: which has more parameters and stronger long-range modelling?
?
**LSTM** — more parameters (3 gates + cell state) and usually stronger long-range modelling. **GRU** is faster, simpler, and often comparable but slightly weaker on very long sequences.

What's the main remaining limitation of LSTM and GRU?
?
**Sequential computation** — $h_t$ still depends on $h_{t-1}$, so no parallelism across time. Also, all past info is still compressed into a single hidden state — a bottleneck that **attention** ultimately fixes.

What is an encoder-decoder architecture?
?
Two RNNs: an **encoder** compresses a source sequence into a hidden representation; a **decoder** generates a target sequence as a **conditional language model** conditioned on that representation.

What's the difference between encoder and decoder training-wise?
?
**Encoder** has **no direct loss** — it's a parametric transformation trained from the decoder's loss. **Decoder** is trained as a conditional language model with cross-entropy on next-token prediction. They're trained jointly.

What's the structural problem with the basic encoder-decoder?
?
The entire source sequence must be compressed into a **fixed-size context vector** (the encoder's final hidden state). For long inputs this becomes a bottleneck — information is lost in compression.

What does the encoder-decoder bottleneck motivate?
?
**Attention** — letting the decoder attend to all encoder hidden states directly, with learned weights per output position, instead of routing all information through a single fixed-size vector (Quiz IV Q5).

What's the typical use case for an encoder-decoder?
?
**Sequence-to-sequence** tasks: machine translation, text summarization, speech-to-text, question answering — input and output have potentially different lengths.

What's the decoder modelling, formally?
?
$P(y_{1:U} \mid x_{1:T}) = \prod_{u=1}^{U} P(y_u \mid y_{<u}, c)$ — autoregressive generation conditioned on previously generated tokens AND the encoder's context $c$.

Why does the encoder's hidden state have no predefined meaning?
?
Encoder and decoder are **trained jointly**. The encoder's representation emerges as a **shared communication space** optimized for the decoder's task — it's not interpretable on its own, only in conjunction with the decoder it was trained with.

Pros of gated RNNs over vanilla RNNs?
?
**Learn what to retain, update, or discard** — directly addresses vanishing/exploding gradients. Preserve relevant signals across longer sequences. Structured memory (especially in LSTM).

Cons of gated RNNs (still shared with vanilla)?
?
Strictly **sequential** — no parallelism across time. All past info must still be encoded into a single hidden state at each step — compressed bottleneck. These are the limits transformers eliminate.

In an LSTM, which symbol denotes the long-term memory?
?
The **cell state $c_t$**. The hidden state $h_t = o_t \odot \tanh(c_t)$ is the part exposed to downstream computation; $c_t$ is the underlying memory channel.
