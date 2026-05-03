---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 03 - Fundamental Concepts.pdf
created: 2026-05-02
---

# N-gram model

A **probabilistic language model** that estimates the likelihood of a token given its local context — the most influential early statistical model of language. An n-gram is a **contiguous sequence of n tokens** extracted from text.

## Definition

$$P(w_t \mid w_{t-n+1}, \ldots, w_{t-1})$$

"Given this context, what token is likely to come next?" The model is a **probability distribution over sequences** — predictive, not classification or regression.

By estimating **conditional probabilities from observed frequencies**, the model assigns likelihoods to possible continuations. Prediction = pick most likely next token, or sample from the distribution.

## On the formula sheet

| Quantity | Formula |
|---|---|
| Unigram model | $P(w_1, \ldots, w_T) = \prod_t P(w_t)$ |
| Bigram model | $P(w_1, \ldots, w_T) = \prod_t P(w_t \mid w_{t-1})$ |
| MLE estimate | $P(w_t \mid w_{t-1}) = \dfrac{c(w_{t-1}, w_t)}{c(w_{t-1})}$ |
| Number of n-grams | $T - n + 1$ for a sequence of length $T$ |

Trigram MLE (the recurring quiz shape):
$$P(w_t \mid w_{t-2}, w_{t-1}) = \frac{c(w_{t-2}, w_{t-1}, w_t)}{c(w_{t-2}, w_{t-1})} = \frac{\text{trigram count}}{\text{bigram count}}$$

This is what Quiz I Q33 calls "**normalizing trigram counts by bigram counts**".

## Worked exam shapes

- `c(I, like, NLP) = 4`, `c(I, like, pizza) = 2`. Then `P(NLP | I like) = 4/(4+2) = 2/3` (Quiz I Q30)
- `c(we, study, language) = 3`, `c(we, study, models) = 1`. Then `P(language | we study) = 3/4` (Quiz I Q31)
- `c(the, model) = 10`, `c(the, model, works) = 4`. Then `P(works | the, model) = 4/10 = 0.4` (Quiz I Q32)

## Token-agnostic property

N-gram models operate on sequences of tokens **regardless of what the tokens represent** — words, morphemes, or subwords. The behaviour depends critically on the tokenization scheme: change the tokens, change the probabilities. See [[tokenization]] and [[morpheme-and-subword-tokenization]].

## Pros and cons

| Pros | Cons |
|---|---|
| Capture local regularities (word order, collocations) | Cannot represent **long-range dependencies** |
| Generate fluent-looking sequences | No hierarchical / semantic structure |
| Surface-level statistics suffice for plausibility | Echo of [[eliza|ELIZA]]: behaviour can appear coherent without understanding |

## What n-grams are not

> N-gram models are **predictive rather than interpretive**. They estimate what is likely to follow, not what is meant.

Their success **reinforces the distinction between generating language and understanding it** — same conceptual point as [[eliza|ELIZA]] and [[turing-test]]. They rely on local context (Quiz I Q34) and do not capture deep meaning.
