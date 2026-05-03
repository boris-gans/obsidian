---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 09 - Naive Bayes-1.pdf
created: 2026-05-02
---

# Text classification

A **supervised learning task** in which the response variable is **categorical**: the classifier assigns a label to each document/observation from a predefined set of categories. There is no reference to understanding or meaning — the classifier finds a **decision boundary** and assigns probabilities.

## Examples (slide 130)

- Spam vs not-spam (the canonical classical NLP classification task — short texts, binary labels, **strong class imbalance**, operational decisions)
- Sentiment classification (positive / negative / neutral)
- Topic categorization
- Customer feedback labelling

## What classifiers learn

The classifier estimates **conditional probabilities** that observations belong to a particular level of the categorical variable. Two paradigms (see [[generative-vs-discriminative]]):

- **Generative**: model how data and labels are jointly generated — `P(x|y) P(y)`. Example: [[naive-bayes]].
- **Discriminative**: directly model `P(y|x)` — Example: [[logistic-regression]].

## Common classifiers in NLP

- [[naive-bayes]] — fast, generative, conditional-independence assumption
- [[logistic-regression]] — discriminative, linear in features, log-odds interpretation
- SVM — margin-based discriminative
- Neural networks — flexible discriminative, learn representations end-to-end

## What it is NOT

Mock Q21: Naïve Bayes assumption is **features are independent given the class label** — not joint independence, not features being sequential.

Quiz I Q1 distinguishes a **generative** from a **discriminative** model: a generative model **specifies how documents and labels are jointly generated**; discriminative directly estimates `P(y|x)` for documents.

Quiz II.M2 Q1: a discriminative classifier's goal is **to model label probabilities given document text** — that's `P(y|x)`.
