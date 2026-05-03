---
tags: [flashcards, NLP]
---

# Lecture 03 — Fundamental Concepts (flashcards)

What is the main purpose of tokenization in an NLP pipeline?
?
**Split text into smaller units such as words or subwords** — the basic units assumed by downstream NLP algorithms.

What is the output of `word_tokenize('NLP is fun')`?
?
A **list of tokens**: `['NLP', 'is', 'fun']`.

Which step must occur before counting word frequencies?
?
**Tokenization** — you cannot count units that have not been segmented.

Which preprocessing step reduces vocabulary size by merging different word forms such as *running* and *runs*?
?
**Stemming or lemmatization.**

What's the trade-off between stemming and lemmatization?
?
Stemming prioritizes **statistical simplicity** (rule-based truncation, often non-word stems); lemmatization preserves **linguistic structure** (real dictionary lemmas, depends on grammatical role).

Why is "running" / "runs" not a tokenization issue but a normalization one?
?
Tokenization decides where to split; stemming/lemmatization decides whether to *merge* surface variants of the same root after splitting.

Define an n-gram.
?
A **contiguous sequence of n tokens** extracted from text.

What does an n-gram model define?
?
A **probability distribution over sequences**: $P(w_t \mid w_{t-n+1}, \ldots, w_{t-1})$ — given context, what token is likely to come next.

How are conditional probabilities computed in a trigram model?
?
**By normalizing trigram counts by bigram counts**: $P(w_t \mid w_{t-2}, w_{t-1}) = c(w_{t-2}, w_{t-1}, w_t) / c(w_{t-2}, w_{t-1})$.

Trigram counts: `(I, like, NLP) = 4`, `(I, like, pizza) = 2`. Compute `P(NLP | I like)`.
?
$4 / (4+2) = 2/3$.

If `c(the, model) = 10` and `c(the, model, works) = 4`, what is `P(works | the model)`?
?
$4/10 = 0.4$.

What's a key limitation of n-gram models?
?
They **cannot represent long-range dependencies** — only local context — and produce no hierarchical or semantic structure.

What's the central characterization of n-gram models in one phrase?
?
**Predictive rather than interpretive** — they estimate what is likely to follow, not what is meant.

Why might removing stop-words be harmful for some tasks?
?
Stop-words like *the* and *on* may be **crucial** for syntax analysis or question answering — removing them strips structure.

What is the McCulloh-Pitt neuron?
?
A 1943 logical model where the neuron fires when the sum of inputs exceeds a threshold: $y = H(\sum_i x_i - U)$, with $H$ = Heaviside step.

What does the perceptron equation look like in modern form?
?
$y = f(\mathbf{w}\!\cdot\!\mathbf{x} + b)$ — the composition of an **affine transformation** and a **non-linear activation**.

What does the bias term $b$ do in the perceptron?
?
**Shifts the activation function** so the perceptron can cover the whole function space (move the decision boundary off the origin).

Name three modern activation functions and why they're preferred over Heaviside / sigmoid for deep networks.
?
**ReLU, Leaky ReLU, Parametric ReLU.** They keep derivatives large when the unit is active, avoiding the gradient-vanishing problem of saturating sigmoids.

What is a multi-layer perceptron (MLP)?
?
A **deep feedforward neural network** built by stacking many perceptrons in layers — input, hidden(s), output. The number of hidden layers = **depth**; layer dimensionality = **width**.

What is the universal approximation theorem (informally)?
?
Neural networks (Cybenko 1989; Hornik et al. 1989) can approximate certain classes of functions to **arbitrary precision** given enough hidden units.

How does the MLP differ conceptually from an n-gram model?
?
N-gram models use **fixed frequency counts**; MLPs have **internal parameters** modified through experience, so they generalize across inputs rather than relying on lookup tables.
