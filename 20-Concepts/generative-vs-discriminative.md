---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 09 - Naive Bayes-1.pdf
created: 2026-05-02
---

# Generative vs discriminative classifiers

Two fundamentally different ways to build a probabilistic classifier — they answer different questions and use different machinery.

## The distinction

| | **Generative** | **Discriminative** |
|---|---|---|
| Models | $P(x, y) = P(x|y)P(y)$ | $P(y|x)$ directly |
| Question answered | "How could this data and label have been jointly generated?" | "What is the label probability for this data?" |
| Examples | [[naive-bayes|Naïve Bayes]], HMMs, LDA | [[logistic-regression]], SVM, MLP, transformers |
| Decision rule | $\hat{y} = \arg\max_y P(x|y) P(y)$ via Bayes' rule | $\hat{y} = \arg\max_y P(y|x)$ directly |

## Exam-ready phrasings

**"In text classification, which statement correctly describes a generative model?"**
> "It specifies how documents and labels are **jointly generated**" (Quiz II Q1).

A generative model specifies a joint story (typically via $p(x|y)p(y)$).

**"Which statement best captures the goal of a discriminative text classifier?"**
> "To **model label probabilities given document text**" (Quiz II.M2 Q1) — i.e. $p(y|x)$.

**"Which statement about Naïve Bayes and logistic regression is NOT correct?"**
> "Naïve Bayes directly models $p(y|x)$ using a softmax decision rule." (Quiz II Q18 — the wrong answer to flag) Naïve Bayes is specified via $p(x|y)$ and $p(y)$; $p(y|x)$ is *obtained* by Bayes' rule, not parameterized as softmax.

## Why both matter

- **Generative** models can generate synthetic data, work well with little data, expose interpretable likelihoods per class
- **Discriminative** models often achieve better classification accuracy because they don't waste capacity modelling $p(x)$, focus directly on the decision boundary

## Naïve Bayes vs logistic regression — a famous parallel

NB and LR are sometimes called a "generative-discriminative pair":

- NB models `P(words|class) P(class)` and assumes word independence given class
- LR models `P(class|words)` directly with a sigmoid/softmax over a linear function of word features

For binary classification, **logistic regression does NOT assume word independence** (Quiz II.M2 Q18: "It assumes word independence" is the FALSE statement about LR). NB does.
