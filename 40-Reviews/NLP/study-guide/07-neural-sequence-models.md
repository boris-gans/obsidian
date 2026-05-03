---
type: study-guide-cluster
course: NLP
cluster: "07-neural-sequence-models"
theme: "Neural networks: from feedforward to sequence"
prerequisites: [04-classical-classifiers, 05-word-meaning-in-vector-space, 06-sequence-labeling]
covers-concepts:
  - perceptron
  - multilayer-perceptron
  - activation-function
  - backpropagation
  - softmax
  - cross-entropy
  - recurrent-neural-network
  - bptt
  - vanishing-exploding-gradients
  - lstm
  - gru
  - encoder-decoder
covers-lectures:
  - lecture-16-feedforward-neural-networks
  - lecture-17-recurrent-neural-networks
  - lecture-18-beyond-rnns
exam-weight: high
---

# Cluster 7: Neural networks — from feedforward to sequence

> **The story of this cluster in one sentence.** HMMs were stuck with discrete tag-transition tables; this cluster replaces them with **learned continuous functions** — first as a single multi-layer net (FFN), then unrolled in time (RNN), then with gates that protect long-range memory (LSTM/GRU), and finally glued together as encoder-decoder for sequence-to-sequence tasks.

## Why this cluster exists

Two limitations meet here. From Cluster 4: linear classifiers are linear, period — they can't learn non-linear decision surfaces. From Cluster 6: HMMs limit transition probabilities to a finite table over discrete tags. The feedforward neural network removes the first limit (multi-layer + non-linearity = universal approximator). The recurrent neural network removes the second limit (the "transition" is now a learned function of an arbitrarily long history compressed into a hidden vector). LSTMs and GRUs then patch RNNs' empirical failure to remember long-range information. Encoder-decoder is the sequence-to-sequence wrapper that lets these models do translation, summarization, and (foreshadowing Cluster 8) the *fixed-context bottleneck* whose failure motivates attention.

Blueprint flags: **high weight**. Mock Q13 (RNN role); Quiz IV Q1–Q5 (RNN, BPTT, LSTM, encoder-decoder bottleneck) and Model B mirrors. The formula sheet provides the **RNN update** `hₜ = tanh(W xₜ + U hₜ₋₁)`, **softmax**, and **cross-entropy**. What's *not* on the sheet: the *interpretation* (what the hidden state encodes, why gradients vanish, what the gates do, why fixed-context fails).

**Prerequisites you should feel solid on:**

- [[logistic-regression]] — LR is conceptually a one-neuron neural network; the perceptron is its NN-shaped twin
- [[sigmoid]] — reused as the LSTM gate activation
- [[hidden-markov-model]] — the discrete-state ancestor of RNN; understand "state evolves over time"
- [[word-embeddings]] — the input layer of every neural model in this cluster

## The arc

### 1. [[perceptron]] — the single neuron

A neuron computes `y = f(w · x + b)` — exactly logistic regression's `z = w · x + b`, followed by a non-linear activation `f`. With `f = sign`, this is the original Rosenblatt perceptron; with `f = sigmoid`, it's logistic regression in NN clothing. The perceptron is the **conceptual bridge** from Cluster 4 to this cluster: every neural model is built out of these units. Fact worth carrying: a **single neuron can only learn linearly-separable decisions** (XOR is impossible) — that's the failure that motivates stacking neurons into multiple layers.

### 2. [[multilayer-perceptron]] — depth + non-linearity = universal approximator

A multilayer perceptron (MLP) stacks neurons into hidden layers, each layer applying `h^(ℓ) = f(W^(ℓ) h^(ℓ-1) + b^(ℓ))`. With at least one hidden layer and a non-linear activation, an MLP can approximate any continuous function (Universal Approximation Theorem). This is *the* reason neural nets matter: linear classifiers (Cluster 4) hit a wall, and adding depth + non-linearity climbs over it. For NLP, the input is typically a sequence of word embeddings flattened (or pooled), the hidden layers learn task-specific features, and the output layer is softmax over classes.

### 3. [[activation-function]] — what makes "non-linear" possible

Without a non-linear activation between layers, stacking is pointless: `W₂(W₁ x) = (W₂ W₁) x` is still linear. Common choices:

- **sigmoid** `σ(z)` — squashes to `(0, 1)`, used as gates
- **tanh** `tanh(z)` — squashes to `(-1, 1)`, the canonical RNN activation
- **ReLU** `max(0, z)` — the modern default; cheap, doesn't saturate for `z > 0`
- **softmax** — the multi-class generalization (next concept)

For NLP, you'll see **tanh** in RNN updates, **sigmoid** in LSTM gates, **ReLU** in transformer feedforward layers, and **softmax** at every output that produces a categorical distribution. Same toolbox, different roles per architecture.

### 4. [[softmax]] — the activation that produces a probability distribution

`softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)`. Maps a vector of real-valued logits to a probability distribution over the same support. Three roles in this course: (a) **multi-class output layer** (which class?), (b) **language modeling** (which next token?), (c) **attention weights** (which input position?). On the formula sheet. Numerical MCQs (Quiz IV Q6) ask you to compute softmax on a length-3 score vector by hand. Practice: `softmax([1, 2, 3])` — exponentiate, sum, divide.

### 5. [[backpropagation]] — the algorithm that trains everything

Backprop computes `∂L/∂θ` for every parameter `θ` in a neural network by **applying the chain rule layer-by-layer in reverse**. It's the engine behind every gradient-descent step taken on every neural model in this course. You don't need to memorize derivations; you need to know (a) it's reverse-mode automatic differentiation, (b) it's `O(network size)` per training example, (c) it requires the activations to be differentiable (this rules out hard `sign`, motivates sigmoid/tanh/ReLU). Backprop is the *implementation* of the training process; gradient descent is the *update rule*.

### 6. [[cross-entropy]] — the loss that makes softmax outputs trainable

`L = -Σᵢ yᵢ log(ŷᵢ)`. For a one-hot target, this collapses to `-log(ŷ_correct)`: penalize the negative log-probability assigned to the right answer. Cross-entropy is the **canonical loss** for any softmax output: language modeling, classification, attention-output decoding. It's also the LR loss from Cluster 4 generalized to multi-class — same math, more dimensions. On the formula sheet. *Why not 0–1 error?* Because gradient descent needs a smooth loss with informative gradients far from the optimum.

### 7. [[recurrent-neural-network]] — neural net unrolled in time

An RNN applies the **same parametric cell** at every time step, threading a hidden state vector `hₜ` forward:

`hₜ = tanh(W xₜ + U hₜ₋₁)`

The hidden state `hₜ` is a **fixed-size representation of the entire prefix `x₁ … xₜ`**. This is the architectural moment: instead of HMM's discrete tag transitions, the "state" is a dense `d`-dim vector, and the "transition" is a learned matrix multiplication. Same weights `W, U` are reused at every step — the *parameters do not grow with sequence length*. The RNN can do many-to-one (sentiment), many-to-many (POS tagging), or one-to-many (text generation) just by varying *where* you read out from.

### 8. [[bptt]] — backprop through time

To train an RNN, you unroll it across the sequence and run backprop through the unrolled graph. This is **backpropagation through time (BPTT)**. The catch: the gradient of the loss at time `t` w.r.t. parameters that produced `h₁` involves **multiplying `t-1` Jacobians** (one per time step). For long sequences, those products blow up or collapse to zero — that's the next concept.

### 9. [[vanishing-exploding-gradients]] — why vanilla RNNs forget

If the recurrent weight matrix's spectral norm is `< 1`, repeated multiplications make gradients **vanish** (model can't learn long-range dependencies). If `> 1`, gradients **explode** (training becomes unstable). In practice, vanishing dominates and limits effective context to **5–10 tokens** for vanilla Elman RNNs. This is the *empirical failure* that motivates LSTMs. Quiz IV Q2 (and B) targets this — be ready to say "vanilla RNNs fail at long-range dependencies because gradients shrink/explode through repeated multiplications across time steps."

### 10. [[lstm]] — gates that protect a long-term cell state

The LSTM (Hochreiter & Schmidhuber, 1997) introduces a **separate cell state `cₜ`** that flows along the sequence with **multiplicative gates** controlling what to keep, add, or expose:

- **Forget gate** `fₜ`: how much of `cₜ₋₁` to keep
- **Input gate** `iₜ`: how much of the new candidate to add
- **Output gate** `oₜ`: how much of `cₜ` to expose as `hₜ`

The cell update `cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ` keeps the gradient path through `cₜ` close to identity when the gates are open — this is the **constant error carousel** that solves vanishing gradients in practice. Quiz IV Q3 (and B) drills the gate semantics. Memorize *what each gate does*, not the equations (the formula sheet would only give the RNN update if anything).

### 11. [[gru]] — LSTM lite

The Gated Recurrent Unit (Cho 2014) merges forget + input into a single **update gate** and adds a **reset gate**, dropping the cell-state vs. hidden-state split. Roughly the same expressive power as LSTM with fewer parameters — often the practical default for medium-size models. Quiz IV Q4 (and B) tests the reset/update gate semantics. Be able to say "reset = how much past to ignore in the candidate; update = how much past to carry vs. how much candidate to admit."

### 12. [[encoder-decoder]] — sequence-in, sequence-out

Many NLP tasks (translation, summarization, QA generation) need **variable-length input → variable-length output**. The encoder-decoder pattern: an **encoder RNN** consumes the source sequence and produces a final hidden state; a **decoder RNN** is initialized from that state and generates the output token by token. The fatal flaw: the entire source must compress into a **single fixed-length context vector** — for long inputs, information is lost. This is the **bottleneck whose failure motivates the attention mechanism in Cluster 8**. Quiz IV Q5 (and B) names this directly.

## Connections worth seeing

- **The perceptron is logistic regression with a different activation.** A perceptron with `f = sigmoid` is exactly LR (Cluster 4); with `f = ReLU` it's a modern hidden unit. The "neural" in neural network is *just stacking these units and training end-to-end with backprop*.
- **Cross-entropy here generalizes the binary cross-entropy of LR (Cluster 4).** The form `-Σ yᵢ log ŷᵢ` reduces to LR's binary cross-entropy when there are exactly two classes and `ŷ = σ(z)`. Same loss, multi-class.
- **The RNN's hidden state is the continuous-vector cousin of HMM's discrete tag.** HMM has finite state space `T = {NN, VB, …}`; RNN has continuous state space `ℝ^d`. The "transition function" goes from a fixed table to a learned matrix multiply. *Same architectural slot, infinitely richer occupant.*
- **LSTM gates use sigmoid for the gates and tanh for the candidate.** This is *exactly* the same pair of activations from earlier in the cluster — sigmoid acts as a soft `[0, 1]` switch, tanh provides a centered `[-1, 1]` content vector. Recognizing this lets you *derive* gate behavior rather than memorize it.
- **The encoder-decoder bottleneck is the same problem as the BoW bottleneck.** BoW compresses a document into a single vector and loses order; encoder-decoder compresses a sequence into a single vector and loses position-specific detail. *Fixed-size summaries always lose detail*; the response in both cases is to keep more information around (TF-IDF + n-grams in Cluster 2; attention in Cluster 8).

## Common confusions

- **Perceptron vs. logistic regression** — same `z = w · x + b`; different activation (`sign` vs. `sigmoid`). Perceptron in NN context = the unit; LR = a particular choice of unit + loss.
- **RNN cell count** — the "unrolled" diagram makes it look like there are `n` cells; there is **one cell** with one weight set, applied `n` times. Parameters do not grow with sequence length.
- **Vanishing vs. exploding gradients** — both come from BPTT through long sequences; vanishing means the model can't learn long-range dependencies; exploding means training crashes.
- **LSTM cell state vs. hidden state** — `cₜ` is the long-term memory (gated), `hₜ` is the *exposed* output of the cell (also used by gates at the next step). Different vectors.
- **Encoder-decoder context vector** — a *single* fixed-dim vector summarizing the entire source. Its inadequacy is why attention exists.
- **Cross-entropy as loss vs. entropy as concept** — *cross-entropy* is the loss `-Σ y log ŷ`; *entropy* alone is `-Σ p log p`. Different formulas, related ideas.

## Self-check (synthesis, not recall)

1. **The RNN update `hₜ = tanh(W xₜ + U hₜ₋₁)` looks like one matrix multiplication.** Why does running this for 50 time steps cause gradient problems that running a 50-layer feedforward net does not (or causes much less)? Tie to BPTT and weight sharing.
2. **LSTMs were specifically designed to fix vanilla RNN's vanishing gradient problem.** Pick one of the three gates and describe the mechanism by which it protects gradient flow through the cell state. Why doesn't the same trick work in a vanilla RNN?
3. **The encoder-decoder forces the entire source sentence to compress into a single fixed-dim vector.** Construct a translation example where this clearly loses information. What's the *minimum* architectural change that fixes it? (Hint: Cluster 8.)
4. **Backprop in a feedforward net vs. BPTT in an RNN** — the same chain rule, different unrolled graph. What's the practical training implication that follows? (Think about parallelism and sequential computation.)
5. **Sigmoid (Cluster 4) and softmax (here) are related — softmax is the multi-class sigmoid.** Show that softmax on 2 classes recovers the sigmoid. Why does this matter for understanding LR as a special case of multi-class neural classification?
6. **Looking back to Cluster 6:** an HMM tag has finite cardinality `|T|`; an RNN hidden state has `d` real-valued dimensions. What does this gain in expressiveness, and what does it cost in interpretability?
7. **Looking forward to Cluster 8:** RNNs are strictly sequential — `hₜ` requires `hₜ₋₁`. If we want to process all positions of a sequence in parallel (for GPU efficiency), what kind of architecture would let us compute all `hₜ` at once?

## If you have 10 minutes

1. [[recurrent-neural-network]] — the RNN update + folded vs. unfolded view; the "many-to-one / many-to-many / encoder-decoder" task taxonomy
2. [[vanishing-exploding-gradients]] — one paragraph on why this happens and what context length it limits a vanilla RNN to
3. [[lstm]] — the three gates and what each does (forget / input / output)
4. [[encoder-decoder]] — the fixed-context-bottleneck framing — this *is* the cliff-hanger for Cluster 8

## Next cluster

→ [[08-attention-and-transformers]] — the encoder-decoder bottleneck (this cluster's last unsolved problem) is fixed by **letting the decoder attend to all encoder positions** at every step, with weights that depend on what's being decoded. That move opens the door to dropping recurrence entirely (transformers), processing tokens in parallel, and learning much longer-range dependencies. It's the architectural shift that defines modern NLP.
