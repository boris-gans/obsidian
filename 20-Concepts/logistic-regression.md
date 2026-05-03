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

# Logistic regression

A **discriminative** binary classifier that maps a linear score of features to a probability via the [[sigmoid]] function. The workhorse text classifier; the same form $z = wx + b$ followed by a non-linearity is the [[perceptron|perceptron unit]], so LR is conceptually a one-neuron neural network.

## Definition (formula sheet)

| Quantity | Formula |
|---|---|
| Linear score | $z = \mathbf{w}\!\cdot\!\mathbf{x} + b$ |
| Sigmoid | $\sigma(z) = \dfrac{1}{1+e^{-z}}$ |
| Prediction | $P(y=1|x) = \sigma(z)$ |
| Decision rule | $\hat{y} = 1 \text{ if } \sigma(z) \geq 0.5$ |

What's **linear** in the features is the **log-odds** (Quiz II Q11), *not* the probability:
$$\log \frac{p(y=1|x)}{p(y=0|x)} = \mathbf{w}\!\cdot\!\mathbf{x} + b$$

See [[log-odds]] for interpretation.

## Sigmoid evaluations (memorize)

| $z$ | $\sigma(z)$ |
|---|---|
| $0$ | $0.5$ |
| $1$ | $\approx 0.73$ |
| $2$ | $\approx 0.88$ |
| $-2$ | $\approx 0.12$ |
| $\to +\infty$ | $\to 1$ |
| $\to -\infty$ | $\to 0$ |

## Coefficient interpretation

A coefficient $w_i$ for word feature $x_i$ adds $w_i$ to the **log-odds** when the word is present. Equivalently, it multiplies the **odds** by $e^{w_i}$:

| Sign / magnitude | Interpretation |
|---|---|
| $w_i = +1.5$ | Strong positive evidence; odds ×$e^{1.5}$ (Quiz II Q15) |
| $w_i = -2$ | Strong negative evidence; odds ×$e^{-2}$ (Quiz II.M3 Q5) |
| $|w_i|$ large | The word has **strong predictive influence** (Quiz II Q16) |
| $w_i \approx 0$ | The word has **little influence** (Quiz II.M2 Q20) |
| $w_i < 0$ | **Decreases the log-odds of the positive class** (Quiz II Q20) |

## Bias term

$b$ shifts the decision boundary uniformly. Increasing $b$ makes positives more likely without changing feature ratios (Quiz II Q7). Useful for class-rebalancing knob.

## Training objective

LR is trained by **minimizing cross-entropy with observed labels** (Quiz II Q9, II.M2 Q9), which is equivalent to **maximum likelihood**:
$$\mathcal{L}(\theta) = -\sum_i \big[y_i \log p_i + (1-y_i) \log(1-p_i)\big]$$

This is the binary cross-entropy on the formula sheet. **Not** 0-1 error, **not** margin width (that's SVM), **not** entropy of predictions.

## Discriminative vs Naïve Bayes

| | Naïve Bayes | Logistic regression |
|---|---|---|
| Type | Generative (models $p(x,y)$) | Discriminative (models $p(y|x)$) |
| Independence assumption | **Yes** — features conditionally independent given label | **No** — no independence assumption |
| Decision rule via softmax? | No, via Bayes' rule | Yes (sigmoid in binary case) |

> "It assumes word independence" is **false** for logistic regression (Quiz II.M2 Q18) — that's NB.

## Derivation: LR is a maximum-entropy classifier

See [[maximum-entropy]]. LR emerges as the unique distribution that:

- Respects observed feature expectations (training data constraints)
- Has **maximum entropy** otherwise (least committed beyond data)

This is why "increasing absolute coefficient strengthens feature influence" (Quiz II Q16) — the coefficient is a **Lagrange multiplier** enforcing the strength of the constraint.

## Decision threshold and metrics

The default threshold is $\sigma(z) \geq 0.5$, but it's a tunable knob. See [[decision-threshold]]. Lowering it increases recall at the cost of precision; raising it does the opposite.

> Probabilities near 0.5 indicate **high entropy** — maximum uncertainty in binary classification (Quiz II.M2 Q19).

## Statistical Learning framing (L02): "the neuron as a predictor"

In [[lecture-02-linear-classifiers-perceptron|SLP L02]], logistic regression is built as the *fix* for the perceptron's three failure modes — specifically the third, "returns any arbitrary separating hyperplane." The construction is identical to the NLP framing but motivated geometrically:

1. The dot product $w^T x + b$ is already a "score" with a useful sign convention (zero on the boundary).
2. The [[sigmoid]] reshapes the score into a probability $\hat{y} = \sigma(w^T x + b) \in (0, 1)$.
3. The **logistic-loss / cross-entropy** is derived as the negative log-likelihood of the data under the Bernoulli model induced by $\hat{y}$.

In SLP terminology this is "the neuron as a predictor" ([[30-Sources/Statistical-Learning/pdf/SLP-lec2(1).pdf#page=42|slide 42]]) — i.e. a sigmoid-activated single-unit perceptron. The same construction generalizes to multi-class via softmax in L03, then to depth via [[multilayer-perceptron|MLPs]] in L04.

The single-point loss, with $y \in \{-1, +1\}$ rather than $\{0, 1\}$, is the [[logistic-loss]] form $\ell_k = \log(1 + e^{-y_k(w^T x_k + b)})$ — the same object, parameterized for the SVM-style label convention used throughout the rest of the SLP course.
