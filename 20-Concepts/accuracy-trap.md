---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 09 - Naive Bayes-1.pdf
created: 2026-05-02
---

# Accuracy trap

The phenomenon where a classifier achieves **high accuracy without doing useful work**, by always predicting the majority class on **imbalanced data**.

## The mechanic

If the dataset is 95% negatives and 5% positives (e.g. spam detection), a classifier that always predicts "negative" gets **95% accuracy** — and zero recall on the positive class. It cannot detect any positives at all. The high accuracy is a property of the data distribution, not the model.

## Symptoms

- **High accuracy, near-zero recall** — the canonical pattern (Quiz II.M3 Q4)
- High accuracy, AUC near 0.5 — model ranks no better than random
- Precision undefined or extremely poor on the minority class
- Constant prediction across the entire test set

## Diagnostic questions

| Question | Answer |
|---|---|
| Which metric is most likely **misleadingly high** when an imbalanced classifier always predicts the majority class? | **Accuracy** (Quiz II Q6) |
| Accuracy is misleading mainly when… | …**classes are imbalanced** (Quiz II.M3 Q17) |
| Why prefer AUC over accuracy in some cases? | Because **thresholds may vary** and accuracy is sensitive to the chosen cutoff (Quiz II Q17) |

## Why this matters for NLP

Many real NLP tasks are **highly imbalanced**: spam filtering, fraud detection, rare-entity NER, hate-speech detection. Defaulting to accuracy as the success metric in these settings produces misleading results — the system appears to work well while completely failing on the class that matters.

## Mitigations

- Report **precision, recall, F1** instead of (or alongside) accuracy
- Use **AUC** for threshold-independent comparison
- Compare against [[baseline-models|baseline models]] — random and prevalence (majority-class) classifiers — to set a floor that any useful model must beat
- Examine the confusion matrix directly, not just summary scalars
