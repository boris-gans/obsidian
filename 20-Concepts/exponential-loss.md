---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Loss-functions-regs.pdf
  - course: Statistical-Learning
    file: SLP-Adaboost(1).pdf
created: 2026-05-03
---

# Exponential loss

A classification loss for binary labels $y \in \{-1, +1\}$:

$$
\ell_{\exp}(z) = e^{-z}, \qquad z = y_i \, h(x_i).
$$

Like hinge and logistic losses, it's a **convex surrogate for the 0/1 loss**, but with a much more aggressive shape.

## The defining shape

$\ell_{\exp}$ as a function of the margin $z$:

| $z$ (margin) | $\ell_{\exp}(z) = e^{-z}$ |
| --- | --- |
| $z \gg 0$ (confidently correct) | $\to 0$ exponentially |
| $z = 0$ (on the boundary) | $1$ |
| $z = -1$ (slightly wrong) | $e \approx 2.72$ |
| $z \ll 0$ (confidently wrong) | grows **exponentially** without bound |

The exponential growth is what makes this loss "aggressive": the loss of a mis-prediction increases **exponentially** with the magnitude of the mistake — and so, therefore, does the gradient signal.

## When it shines: AdaBoost

The slide deck flags this loss as the one **AdaBoost** minimizes (proven via the additive-modeling derivation in L13/L14). The exponential shape gives AdaBoost its three signature behaviors:

- **Aggressive re-weighting**: misclassified examples get exponentially-larger weights in the next round.
- **Fast convergence on clean data**: nice convergence guarantees for gradient-boosting-style algorithms.
- **No probabilistic output**: unlike logistic loss, exponential loss does **not** correspond to a calibrated probability — its output is a margin score, not a $P(y \mid x)$.

## When it fails: noisy labels

The exponential shape is unbounded below at $z \to -\infty$. A single mis-labeled example with margin $-10$ has loss $e^{10} \approx 22\,026$ — and pulls the optimizer toward fitting that example at the expense of everything else. The lecture's exact phrasing:

> *"This function is very aggressive. The loss of a mis-prediction increases exponentially with the value of $-h_w(x_i)y_i$. This can lead to nice convergence results, for example in the case of AdaBoost, but it can also cause problems with noisy data."*

So exponential loss is high-leverage when the data is clean but **brittle** when labels have noise. This is the opposite of hinge loss, which caps influence per example at the margin violation.

## Comparison with the other classification losses

| Loss | Formula | Far-correct ($z \gg 0$) | Far-wrong ($z \ll 0$) | Robust to label noise? |
| --- | --- | --- | --- | --- |
| **0/1** | $\mathbb{1}[z \le 0]$ | $0$ | $1$ | n/a (non-differentiable) |
| **Hinge** | $\max(1-z, 0)$ | exactly $0$ for $z \ge 1$ | linear: $1 - z$ | medium |
| **Logistic** | $\log(1 + e^{-z})$ | $\to 0$ exponentially | linear in $|z|$ asymptotically | medium |
| **Exponential** | $e^{-z}$ | $\to 0$ very fast | grows **exponentially** in $|z|$ | **low** |

## Exam-relevant facts

- $\ell_{\exp}(z) = e^{-z}$ for $z = y\,h(x)$.
- The classification loss minimized (implicitly) by **AdaBoost**.
- **Convex** and **smooth** (differentiable everywhere), but **brittle on noisy data** because errors penalize exponentially.
- Does **not** give calibrated probabilities (compare to logistic loss).

## Related

- [[hinge-loss]] / [[logistic-loss]] — alternative convex surrogates for 0/1.
- [[lecture-10-loss-functions-regularization|SLP L10]] — first appearance.
- [[adaboost]] / [[lecture-14-adaboost|SLP L14]] — the algorithm that minimizes this loss in stages, with closed-form line search yielding $\alpha_t = \tfrac{1}{2}\ln\tfrac{1-\epsilon_t}{\epsilon_t}$.
