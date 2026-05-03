---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Lec1-knn(1).pdf
created: 2026-05-03
---

# Train / Validation / Test Split

## Definition

Partition a labelled dataset into three disjoint sets, each with a distinct role:

- **Train** — the data the learning algorithm fits its parameters on.
- **Validation (or "dev")** — used to compare candidate hyperparameter settings or model variants. The model is repeatedly *evaluated* on validation but never *fit* to it.
- **Test** — held out completely until a single, final, unbiased evaluation of the chosen model.

> *"Idea #3: Split data into train, val, and test; choose hyperparameters on val and evaluate on test. Better!"* ([[30-Sources/Statistical-Learning/pdf/SLP-Lec1-knn(1).pdf#page=78|slide 78]])

## Why three sets, not two

L01 walks through the alternatives ([[30-Sources/Statistical-Learning/pdf/SLP-Lec1-knn(1).pdf#page=74|slides 74–78]]):

| Idea | What's wrong |
| --- | --- |
| #1: Pick hyperparameters that work best on **the training data** | $k = 1$ kNN always gets 0 train error → trivially "wins". No information about generalisation. |
| #2: Train / test only — pick hyperparameters by **best test score** | Once you've used test to choose, it's no longer unbiased. You have no idea how the model performs on truly new data. |
| #3: Train / val / test | Hyperparameters chosen on val, final score reported on test (used **once**). |
| #4: Cross-validation | Same logic as #3 but uses every fold of train+val rotation for a more robust val estimate. |

## Common ratios

There is no single right answer; it depends on dataset size:

- Small data: 60/20/20.
- Medium data: 70/15/15 or 80/10/10.
- Big data: 98/1/1 (so that 1% test is still tens of thousands of points).

## Pitfalls

- **Test-set leakage** — touching the test set during model selection invalidates the unbiased-estimate guarantee.
- **Distribution shift between splits** — random shuffling assumes IID. For time series, use a chronological split. For grouped data, split by group (e.g., patients), not by row.
- **Tuning over too many hyperparameters with a small val set** — eventually you overfit to validation. Cross-validation reduces this.

## Related

- [[cross-validation]]
- [[hyperparameter]]
