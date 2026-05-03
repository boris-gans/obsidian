---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 09 - Naive Bayes-1.pdf
created: 2026-05-02
---

# Confusion matrix

A 2×2 (or larger) table tabulating the four possible prediction outcomes of a binary classifier. Every binary [[evaluation-metrics|evaluation metric]] is derived from this table.

|   | Predicted Negative | Predicted Positive |
|---|---|---|
| **True Negative** | TN (True Negative) | FP (False Positive) |
| **True Positive** | FN (False Negative) | TP (True Positive) |

## Why we need it

Different errors have **different practical consequences** — relying on a single metric can be misleading. The confusion matrix exposes the four error/success types so we can reason about which matter most for the task.

For **spam classification**, a false positive (legitimate email flagged as spam) is much costlier than a false negative — different metrics weight these differently.

## Derived quantities (all on the formula sheet)

| Metric | Formula | What it measures |
|---|---|---|
| **Precision** | $\dfrac{TP}{TP + FP}$ | Reliability of positive predictions |
| **Recall** (Sensitivity) | $\dfrac{TP}{TP + FN}$ | Proportion of positives recovered |
| **Specificity** | $\dfrac{TN}{TN + FP}$ | How well negatives are detected |
| **NPV** | $\dfrac{TN}{TN + FN}$ | Reliability of negative predictions |
| **Accuracy** | $\dfrac{TP+TN}{N}$ | Overall correctness |
| **F1** | $\dfrac{2 PR}{P+R}$ | Harmonic mean of precision and recall |

See [[evaluation-metrics]] for usage trade-offs and [[accuracy-trap]] for when accuracy lies.

## Worked exam shapes

- "100 entities predicted, 80 correct" → precision = 80/100 = **0.80** (Quiz III Q15)
- "50 entities predicted, 40 correct" → precision = 40/50 = **0.80** (Quiz III Q15 Model B)
