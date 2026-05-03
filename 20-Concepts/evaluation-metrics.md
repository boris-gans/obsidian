---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 09 - Naive Bayes-1.pdf
created: 2026-05-02
---

# Evaluation metrics

The toolkit for assessing classifier performance. All derive from the [[confusion-matrix]]. Choosing among them is a modelling decision: different errors have different costs.

## Definitions (formula sheet)

| Metric | Formula | What it measures |
|---|---|---|
| **Accuracy** | $(TP+TN)/N$ | Overall correctness |
| **Precision** | $TP/(TP+FP)$ | Reliability of positive predictions |
| **Recall** | $TP/(TP+FN)$ | Proportion of positives recovered |
| **Specificity** | $TN/(TN+FP)$ | Correct rejection of negatives |
| **NPV** | $TN/(TN+FN)$ | Reliability of negative predictions |
| **F1** | $2PR/(P+R)$ | Harmonic mean of precision and recall |
| **AUC** | Area under ROC | Ranking quality across thresholds |

## When each one matters

- **Imbalanced classes** (e.g. spam = 5%): use **precision / recall / F1**, not accuracy. See [[accuracy-trap]].
- **Threshold-free comparison**: use **AUC** — measures ranking quality independent of any specific cutoff.
- **Need to trade off recall and precision**: lower the decision threshold to **increase recall, decrease precision** (Quiz II.M2 Q6); raise it to do the opposite (Quiz II.M2 Q10).

## ROC / AUC

ROC curves are useful when **decision thresholds vary** (Quiz II.M2 Q17, II Q17). AUC summarizes ranking quality across all thresholds.

> AUC measures **ranking quality** of documents across thresholds (Quiz II Q14, II.M3 Q9).

A classifier with **high AUC but low accuracy** usually indicates **poor threshold choice** (Quiz II.M3 Q14) — the model ranks documents well but the chosen cutoff misclassifies. A classifier with **high accuracy and AUC ≈ 0.52** is one that classifies the majority class but cannot rank — an [[accuracy-trap]] symptom.

## Worked exam shapes

| Scenario | Implication |
|---|---|
| Classifier always predicts negative on imbalanced data | High accuracy, near-zero recall (Quiz II.M3 Q4: "high accuracy and near-zero recall" is the accuracy trap pattern) |
| Classifier A: acc=0.90, AUC=0.52; Classifier B: acc=0.85, AUC=0.93 | **B ranks documents better** (Quiz II.M2 Q14) |
| Lower decision threshold | **Increase recall, decrease precision** |
| Raise decision threshold to 0.9 | **Increase precision** (Quiz II.M2 Q10) |

## Trade-off intuition

Precision is reliability of positives ("when I say spam, am I right?"); recall is coverage of positives ("did I catch all the spam?"). They trade off via the threshold. F1 balances them with a harmonic mean — sensitive to the smaller of the two.
