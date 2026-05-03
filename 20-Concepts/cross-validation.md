---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Lec1-knn(1).pdf
created: 2026-05-03
---

# Cross-Validation

## Definition

A resampling procedure for estimating the generalisation error of a model — and especially for selecting hyperparameters — when data is too scarce to spend a chunk on a single validation set.

**$k$-fold cross-validation:** partition the available (non-test) data into $k$ equal-sized folds. For each fold $i \in \{1, \dots, k\}$:

1. Train on the union of the other $k - 1$ folds.
2. Evaluate on fold $i$.

Average the $k$ scores. The result is a less variance-prone estimate of out-of-sample performance than a single train/val split.

> *"Idea #4: Cross-Validation: Split data into folds, try each fold as validation and average the results — gives a more robust estimate of the results!"* ([[30-Sources/Statistical-Learning/pdf/SLP-Lec1-knn(1).pdf#page=79|slide 79]])

## Variants

- **$k$-fold** — standard, $k = 5$ or $k = 10$ in practice.
- **Leave-one-out (LOO)** — $k = n$. Maximally data-efficient but expensive; estimate has high variance because successive training sets differ in only one point.
- **Stratified $k$-fold** — preserves the class proportions in each fold. Default for classification with imbalanced labels.
- **Repeated $k$-fold** — re-shuffle and repeat the whole CV scheme; average scores across repetitions.
- **Time-series CV** — train on a prefix, validate on the next chunk; never validate on data older than your train set.

## How to use it for hyperparameter selection

```python
for hp in hyperparameter_grid:
    scores = []
    for fold in k_folds(X_train, y_train, k=5):
        train_idx, val_idx = fold
        model = fit(X_train[train_idx], y_train[train_idx], hp)
        scores.append(score(model, X_train[val_idx], y_train[val_idx]))
    cv_score[hp] = mean(scores)

best_hp = argmax(cv_score)
final_model = fit(X_train, y_train, best_hp)   # refit on all train+val
report(score(final_model, X_test, y_test))     # one-shot on held-out test
```

The **test set is still held out** — CV replaces the train/val split only.

## What it is *not*

- **Not a substitute for a held-out test set.** If you pick hyperparameters by CV, the CV score is biased by selection. Report the test score on data the procedure never saw.
- **Not free from data leakage.** Any preprocessing that uses statistics across rows (PCA, normalization, feature selection) must be re-fit *inside* each fold, not on the full data, to keep the val fold genuinely unseen.

## Related

- [[train-validation-test-split]]
- [[hyperparameter]]
- [[k-nearest-neighbors]] — used to select $k$.
