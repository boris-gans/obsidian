---
tags: [flashcards, NLP]
---

# Lecture 12 — SVM (flashcards)

What does an SVM optimize that distinguishes it from logistic regression?
?
**Margin width** — the distance from the decision boundary to the closest training points. LR minimizes cross-entropy; SVM maximizes the margin via hinge loss.

What is the SVM decision rule for a linear binary classifier?
?
$\hat{y} = \text{sign}(\mathbf{w}\!\cdot\!\mathbf{x} + b)$ — same linear score form as LR / perceptron, different training objective.

What's the loss function used by linear SVM?
?
**Hinge loss**: $\max(0, 1 - y(\mathbf{w}\!\cdot\!\mathbf{x} + b))$ — zero if correctly classified with margin $\geq 1$, otherwise grows linearly.

Does SVM produce a probability?
?
**No** — it produces an uncalibrated margin score. Probabilities require post-hoc calibration (e.g. Platt scaling).

What are support vectors?
?
The training points closest to the decision boundary — the points that define the margin. Removing other points doesn't change the trained classifier.

If a question asks what logistic regression minimizes, what's the wrong answer to flag?
?
**Margin width** — that's SVM-style. LR minimizes **cross-entropy** with observed labels.

What's the kernel trick?
?
Replacing dot products $\mathbf{x}_i \!\cdot\! \mathbf{x}_j$ with a kernel $K(x_i, x_j)$ that implicitly computes a dot product in a higher-dimensional feature space — letting linear SVM solve non-linear problems.

When is linear SVM typically preferred over kernel SVM in NLP?
?
On **high-dimensional sparse text features** (BoW / TF-IDF) — linear SVM is fast, robust, and usually performs comparably to kernel SVM in this regime.

What's a key strength of SVMs that made them prominent in early-2000s NLP?
?
**Strong performance on small, high-dimensional data** — typical text classification with TF-IDF features. They were the standard before LR and neural models took over.
