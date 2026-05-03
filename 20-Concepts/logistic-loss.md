---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-lec2(1).pdf
  - course: Statistical-Learning
    file: SLP-Loss-functions-regs.pdf
created: 2026-05-03
---

# Logistic loss

The single-point training loss for a sigmoid-output linear classifier ([[logistic-regression]] / "neuron as predictor"), written with the **SVM-style label convention** $y \in \{-1, +1\}$. Algebraically identical to binary [[cross-entropy]] with $y \in \{0, 1\}$ — the *same* loss, just relabelled to make the connection to the [[linear-svm|SVM]] hinge loss visible.

## Definition (SLP L02 form)

For a single training example $(x_i, y_i)$ with $y_i \in \{-1, +1\}$ and **margin** $m_i = y_i(w^T x_i + b)$:

$$
\ell_i = \log\!\big(1 + e^{-y_i(w^T x_i + b)}\big) \;=\; \log(1 + e^{-m_i}).
$$

Equivalently, derived from the sigmoid:

$$
P(y_i \mid x_i) = \frac{1}{1 + e^{-y_i (w^T x_i + b)}}, \qquad \ell_i = -\log P(y_i \mid x_i).
$$

So **logistic loss = $-\log P(y_i \mid x_i)$** with the unified $y \in \{-1, +1\}$ form built in. ([[30-Sources/Statistical-Learning/pdf/SLP-lec2(1).pdf#page=67|slides ~65–68]] derive this form.)

## Behaviour vs. the margin $m = y(w^T x + b)$

| Margin $m$ | Loss $\ell$ | Meaning |
| --- | --- | --- |
| $m \to +\infty$ | $\to 0$ | confidently correct: $\sigma$ outputs ~$1$ for the true class |
| $m = 0$ | $\log 2 \approx 0.693$ | on the boundary: $\hat{p} = 0.5$, max uncertainty |
| $m \to -\infty$ | grows linearly in $|m|$ | confidently wrong: $\ell \approx -m$ |

The **asymptote**: for $m \ll 0$, $\log(1 + e^{-m}) \approx -m$. So logistic loss is *roughly linear* in the margin for confidently-wrong predictions — never blows up super-linearly the way squared loss would.

## Equivalence with binary cross-entropy

For $y_{01} \in \{0, 1\}$ and $\hat{p} = \sigma(z)$:

$$
\ell_{\text{CE}} = -\big[y_{01}\log\hat{p} + (1 - y_{01})\log(1 - \hat{p})\big].
$$

Substituting $y_{\pm} = 2y_{01} - 1$ collapses both branches into a single expression:

$$
\ell_{\text{CE}} = \log\!\big(1 + e^{-y_{\pm}\,z}\big) = \ell_{\text{logistic}}.
$$

Same loss, two parameterizations. The $\{-1, +1\}$ form is preferred whenever the rest of the pipeline uses signed margins (SVMs, AdaBoost, kernel methods).

## Total training loss

Summed over the dataset, this is the SLP L02 objective ([[30-Sources/Statistical-Learning/pdf/SLP-lec2(1).pdf#page=72|final slides]]):

$$
\mathcal{L}(w, b) = \sum_{i=1}^{N} \log\!\big(1 + e^{-y_i(w^T x_i + b)}\big).
$$

L02 stops here — *defining* what we minimize. Optimization (gradient descent / SGD) lands in L03–L05.

## L10's framing — "well-calibrated probabilities"

[[lecture-10-loss-functions-regularization|SLP L10]]'s loss table flags log-loss as **the most popular loss function in ML, since its outputs are well-calibrated probabilities** ([[30-Sources/Statistical-Learning/pdf/SLP-Loss-functions-regs.pdf#page=4|slide 4]]). The "calibrated" claim is what distinguishes it from hinge and exponential loss: minimizing log-loss makes $\hat{p} = \sigma(z)$ converge toward the **true** $P(y = +1 \mid x)$, so $\hat{p} = 0.7$ really means *"7 out of 10 examples like this are positive."* Hinge loss only gives margin scores; exponential loss gives uncalibrated relative scores.

This is also why log-loss is the default for downstream applications that need probabilities — uncertainty quantification, decision thresholds, ensembling — while hinge / exponential are fine when you only need a hard label.

## Why "log-loss" / "binomial deviance" / "logistic loss" / "negative log-likelihood" are the same thing

Different communities, same function:

- **Statisticians**: deviance under a Bernoulli MLE.
- **Information theorists**: cross-entropy of $\hat{p}$ from the true label distribution.
- **Optimization / SVM crowd**: a smooth surrogate to the 0–1 loss that's convex and differentiable everywhere.
- **Deep-learning practitioners**: "log-loss" or "BCE with logits."

When the SLP exam says **"logistic loss"** it means this object specifically — usually in the $y \in \{-1, +1\}$ form, often compared side-by-side with hinge loss in L09–L10.

## Comparison with hinge loss (foreshadow L09–L10)

| Loss | Formula | Convex? | Differentiable? | Behavior at margin $m \gg 0$ | At $m \ll 0$ |
| --- | --- | --- | --- | --- | --- |
| **Logistic** | $\log(1 + e^{-m})$ | ✓ | ✓ everywhere | $\to 0$ smoothly | linear in $|m|$ |
| **Hinge** ([[linear-svm\|SVM]]) | $\max(0, 1 - m)$ | ✓ | piecewise (kink at $m=1$) | exactly $0$ for $m \geq 1$ | linear in $|m|$ |
| **0–1** | $\mathbb{1}[m \leq 0]$ | ✗ | ✗ | $0$ | $1$ |

Both logistic and hinge are convex surrogates of the 0–1 loss. The key difference: hinge **stops penalizing** correctly-classified points beyond the margin (sparse support vectors); logistic **always assigns nonzero gradient** (every example influences $w$, just exponentially less the further inside the correct side it sits).

## Related

- [[cross-entropy]] — same loss in the $y \in \{0, 1\}$ parameterization.
- [[sigmoid]] — the link function whose negative log gives this loss.
- [[logistic-regression]] — the model trained with this loss.
- [[linear-classifier]] — the family this loss is for.
- [[perceptron]] — historical predecessor (mistake-driven, not loss-minimizing).
