---
tags: [concept]
courses: [NLP, Statistical-Learning]
sources:
  - course: NLP
    file: Session 10 - Logistic Regression-1.pdf
  - course: Statistical-Learning
    file: SLP-lec2(1).pdf
created: 2026-05-02
---

# Sigmoid function

The link function in [[logistic-regression]] that maps an unbounded linear score $z \in \mathbb{R}$ to a probability in $(0, 1)$.

## Definition (formula sheet)

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Equivalently, $\sigma(z) = \frac{e^z}{1 + e^z}$.

## Key values to memorize

| $z$ | $\sigma(z)$ | Interpretation |
|---|---|---|
| $-\infty$ | $0$ | Confidently negative |
| $-2$ | $\approx 0.12$ | Strong negative |
| $-1$ | $\approx 0.27$ | Mild negative |
| $0$ | $0.5$ | **Maximum entropy / decision boundary** |
| $1$ | $\approx 0.73$ | Mild positive |
| $2$ | $\approx 0.88$ | Strong positive |
| $+\infty$ | $1$ | Confidently positive |

## Properties

- **Smooth and differentiable**: $\sigma'(z) = \sigma(z)(1-\sigma(z))$
- **Monotonic**: increasing in $z$
- **$z = 0 \Rightarrow \sigma = 0.5$**: the decision boundary lies at zero log-odds (Quiz II.M3 Q6)
- **As $z \to +\infty$**, $\sigma \to 1$; as $z \to -\infty$, $\sigma \to 0$ (Quiz II.M2 Q8)
- **Saturates** at extremes — this is why ReLU-family activations are preferred for deep networks

## Use in NLP

| Use | Where |
|---|---|
| Binary classification probability | [[logistic-regression]] |
| Activation in [[perceptron]] / single-neuron networks | Session 03, 16 |
| **Generalization to multiclass = softmax** | Session 19 (formula sheet) |
| Forget / input gate output in LSTM | Session 18 |

## In Statistical Learning (L02)

The sigmoid is introduced ([[30-Sources/Statistical-Learning/pdf/SLP-lec2(1).pdf#page=42|slide 42]]) as the "special function" that turns the linear score $z = w^T x + b$ into a **probability** — the tool that fixes the perceptron's third failure mode (arbitrary separating hyperplane). The lecture's intuition order:

1. The dot product $w^T x + b$ already gives a *score* — $0$ on the boundary, positive on the $+$ side, negative on the $-$ side.
2. We want points on the boundary to score $0.5$, far points to score near $0$ or $1$.
3. **The sigmoid is the function that does that:** $\sigma(0) = 0.5$, $\sigma(\infty) = 1$, $\sigma(-\infty) = 0$, smooth and monotonic.

The decision rule becomes "is $\sigma(z) > 0.5$?" — equivalently $z > 0$, so the *boundary* is identical to the perceptron's; what changes is the *confidence* expressed by points away from the boundary.

This direct sigmoid → probability story is the conceptual setup for the cross-entropy loss derivation later in the same lecture.

## Interpretation through entropy

When $\sigma(z) \approx 0.5$, the model is maximally uncertain — **high entropy** (Quiz II.M2 Q19). When $\sigma(z) \to 0$ or $1$, **low entropy / high confidence**. The shape of the sigmoid is precisely how uncertainty changes with linear evidence.
