---
tags: [flashcards, NLP]
---

# Lecture 16 — Feedforward Neural Networks (flashcards)

What does a single neuron compute?
?
$y = \sigma(w^\top x + b)$ — a weighted sum of the inputs followed by a nonlinear activation. A parametric nonlinear function of the input vector.

Write the formula for a one-hidden-layer feedforward network.
?
$h = \sigma(W_1 x + b_1)$, $y = W_2 h + b_2$. Combined: $y = W_2\,\sigma(W_1 x + b_1) + b_2$.

What happens if you remove the activation function from a deep network?
?
The whole network collapses to a single linear transformation: $y = W_2 W_1 x$. Stacking layers adds no expressive power without nonlinearity.

What's the formula for the softmax of $x_i$?
?
$\mathrm{softmax}(x_i) = \dfrac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$ — exponentiate each score and normalize so they sum to 1.

For input $\mathbf{z} = (1, 2, 3)$, what's $\mathrm{softmax}(\mathbf{z})$?
?
$(e^1/30.2, e^2/30.2, e^3/30.2) \approx (0.090, 0.245, 0.665)$ — total = 1; the largest input gets the largest probability.

What's the relationship between sigmoid and softmax?
?
**Sigmoid is the binary special case of softmax** ($K=2$). Both convert raw scores to probabilities.

What's the cross-entropy loss for $K$-class classification?
?
$L = -\sum_{k=1}^{K} y_k \log p_k$ where $y$ is one-hot ground truth and $p$ is predicted softmax probabilities. Used as the standard loss for neural classifiers.

What's the gradient of softmax + cross-entropy at the output?
?
$\partial_z L = p - y$ — clean closed form: predicted probabilities minus target one-hot. The combination is numerically convenient because of this simplification.

What does backpropagation compute?
?
The **gradients of the loss with respect to all model parameters**, by recursive application of the chain rule through the layers (Rumelhart et al. 1986).

What's an epoch in training?
?
**One full pass** of forward + backward + parameter update over the training data.

State the gradient-descent parameter update.
?
$\theta_{\text{new}} = \theta_{\text{old}} - \varepsilon \cdot \partial_\theta L$ — step in the direction opposite the gradient by a factor $\varepsilon$ (the learning rate).

What's the role of the learning rate $\varepsilon$?
?
Controls step size. **Too small** → very slow convergence. **Too large** → loss oscillates around the minimum or even increases.

What's the difference between Batch / SGD / Mini-batch?
?
**Batch**: gradient over all training data per update (slow, accurate). **SGD**: one example per update (noisy, fast). **Mini-batch**: a few dozen examples per update (the practical default).

What's the default optimizer in modern NLP?
?
**Adam** — adaptive learning rate with momentum. Robust across most setups.

When to use which loss function?
?
Regression → **MSE**. Binary classification → **Binary Cross-Entropy**. Multi-class classification → **Cross-Entropy**. Distribution matching → **KL Divergence**.

What's the size of the embedding matrix for vocabulary V and embedding dim d?
?
$V \times d$ — one row per vocabulary word, $d$ columns. Total parameters = $V \cdot d$.

What does $e = E^\top x$ compute when $x$ is one-hot?
?
**Embedding lookup** — selects the row of $E$ corresponding to the nonzero entry of $x$. Since $x$ has exactly one 1, the matrix-vector product is just that row.

How does a simple neural model classify a sentence using word embeddings?
?
**word sequence → embeddings → sentence vector (e.g. average) → feedforward classifier → softmax over classes**. The simplest sentence vector: $s = \frac{1}{T} \sum_t e_t$ (average of word embeddings), which is fixed-length regardless of sentence length.

Why is one-hot encoding insufficient for neural models?
?
(1) Extremely **high-dimensional and sparse** when $V$ is large; (2) **No notion of similarity** between words — every pair is orthogonal. Dense learned embeddings address both.

What's a "dead ReLU" and why does Leaky ReLU exist?
?
A ReLU unit whose output is always 0 (input always $\le 0$) — its gradient is also 0, so it stops learning. **Leaky ReLU** ($z$ if $z>0$ else $0.02z$) and **PReLU** (learned slope) keep a small gradient on the negative side.

Why does ReLU have nicer gradients than sigmoid?
?
Sigmoid **saturates** at large $|z|$ — derivative shrinks to ~0, killing gradients during backprop. ReLU has derivative **1 for all $z > 0$** — gradients flow freely as long as the unit is active.

What's the conceptual shift from classical to neural NLP?
?
Classical: **separate** representation (BoW / TF-IDF) and prediction (Naïve Bayes / LR). Neural: **the model itself learns a representation** of the input that's useful for the prediction task — representation and prediction are jointly optimized.

For the linear-perceptron + softmax + cross-entropy setup, what are the parameter gradients?
?
$\partial_b L = p - y$ and $\partial_W L = (p - y) h^\top$ where $p$ is the softmax output, $y$ the one-hot label, and $h$ the previous-layer activation ([[30-Sources/NLP/pdf/Session 16 - Neural Networks-1.pdf#page=13|slide 13]]).
