---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 10 - Logistic Regression-1.pdf
created: 2026-05-02
---

# Decision threshold

The **probability cutoff** above which a probabilistic classifier predicts the positive class. The default for binary [[logistic-regression]] is $0.5$, but it's a **tunable knob** that trades off precision against recall.

## Default rule

$$\hat{y} = \begin{cases} 1 & \text{if } \sigma(z) \geq \theta \\ 0 & \text{otherwise} \end{cases}$$

with default $\theta = 0.5$ (formula sheet).

## Effect of changing the threshold

| Direction | Effect on recall | Effect on precision |
|---|---|---|
| **Lower** $\theta$ (e.g. 0.5 → 0.3) | **Increases** (predict positive more often → catch more) | **Decreases** (more false positives) |
| **Raise** $\theta$ (e.g. 0.5 → 0.9) | **Decreases** | **Increases** (only confident positives) |

(Quiz II.M2 Q6: lower → recall ↑, precision ↓; Quiz II.M2 Q10: raising to 0.9 → precision ↑.)

## Why it's separate from coefficients

The threshold is **external** to the model — changing it does **not** change the trained weights or the underlying probability estimates. It only changes how those probabilities are converted to labels. Quiz II Q15 makes this clear: changing a coefficient shifts scores; the threshold defines a separate boundary.

## When threshold tuning matters

- **Imbalanced classes** — default 0.5 may produce very low recall on the minority class; lowering rebalances
- **Asymmetric error costs** — if false negatives are much costlier than false positives, lower the threshold
- **Operating point selection** — for ROC/AUC analysis, varying $\theta$ traces out the curve

## Threshold-independent metric

[[evaluation-metrics|AUC]] summarizes ranking quality across **all thresholds** — useful when you don't want to commit to a single $\theta$. ROC curves are especially useful when **decision thresholds vary** (Quiz II.M2 Q17).

## Diagnostic patterns

- **High AUC + low accuracy** ⇒ usually **poor threshold choice** (Quiz II.M3 Q14): the model ranks well but the chosen cutoff misclassifies
- **High accuracy + low AUC** on imbalanced data ⇒ the [[accuracy-trap]]: classifier always predicts the majority class
