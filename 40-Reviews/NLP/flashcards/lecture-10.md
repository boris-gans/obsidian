---
tags: [flashcards, NLP]
---

# Lecture 10 — Logistic Regression (flashcards)

What kind of classifier is logistic regression?
?
A **discriminative** binary classifier — directly models $p(y|x)$, no generative story for documents.

State the logistic regression linear-score and prediction (formula sheet).
?
$z = \mathbf{w}\!\cdot\!\mathbf{x} + b$, then $P(y=1|x) = \sigma(z) = 1/(1+e^{-z})$.

What is linear in the word-feature vector in logistic regression?
?
**The log-odds**: $\log\frac{p(y=1|x)}{p(y=0|x)} = \mathbf{w}\!\cdot\!\mathbf{x} + b$. The probability is non-linear in the features.

Compute $\sigma(2)$.
?
$\sigma(2) \approx 0.88$.

Compute $\sigma(-2)$.
?
$\sigma(-2) \approx 0.12$.

For what value of $z$ does $p(y=1|x) = 0.5$?
?
$z = 0$ — the sigmoid equals $0.5$ exactly at zero.

A logistic-regression coefficient $+1.5$ for a word means…?
?
The odds of the positive class are **multiplied by $e^{1.5}$** when the word is present.

A coefficient of $-2$ implies the odds are multiplied by what?
?
$e^{-2}$.

What does a near-zero coefficient mean for a word?
?
**The word has little influence** on the prediction.

What does a negative coefficient mean for a word?
?
It **decreases the log-odds of the positive class** when the word is present.

What is the role of the bias $b$ in logistic regression?
?
**Shifts the decision boundary uniformly** — increasing $b$ makes positives more likely without changing how features compete.

Increasing the absolute value of a weight in a maximum-entropy classifier does what?
?
**Strengthens the influence of that word feature** on the score and log-odds.

What does training logistic regression by maximum likelihood minimize?
?
**Cross-entropy with observed labels** (= negative log-likelihood). Not 0-1 error, not margin.

Why is logistic regression called a maximum-entropy classifier?
?
Because it is **the distribution with maximum entropy that respects observed feature-expectation constraints** — least committed beyond data. The exponential / softmax form follows from the entropy-maximization solution.

Why does max-entropy under linear constraints yield exponential / softmax models?
?
**It follows from the entropy-maximization solution** via Lagrange multipliers — those multipliers become the learned weights.

What happens to $p(y=1|x)$ as $z \to +\infty$?
?
$p(y=1|x) \to 1$ — the sigmoid saturates at 1.

What happens to recall and precision when you lower the decision threshold?
?
**Recall increases, precision decreases** — predicting positive more often catches more true positives but admits more false positives.

What happens to precision when you raise the threshold to 0.9?
?
**Precision increases** — fewer false positives, but recall typically drops.

What's the FALSE statement about logistic regression?
?
"It assumes word independence." LR makes **no** independence assumption — that's Naïve Bayes.

What distinguishes logistic regression from linear regression?
?
The **non-linear link function** — sigmoid (or softmax) maps the linear score to a probability. Linear regression has unbounded outputs and squared loss.

When would you prefer AUC over accuracy?
?
When **decision thresholds may vary** — AUC measures ranking quality across all thresholds; accuracy is tied to one cutoff.

What does it usually mean if a classifier has high AUC but low accuracy?
?
**Poor threshold choice** — the model ranks documents well but the chosen cutoff misclassifies.

Probabilities near 0.5 in a binary classifier indicate what?
?
**High entropy** — maximum uncertainty in binary classification.
