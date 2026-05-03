---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-lec3(1).pdf
created: 2026-05-03
---

# Linear regression

The same single neuron used for [[logistic-regression|binary classification]], with two changes: **drop the activation function** (no sigmoid) and **swap the loss** to mean squared error. Used to predict a real-valued target $y \in \mathbb{R}$ rather than a class.

## The model

$$
\hat{y}_i = w^T x_i + b, \qquad w \in \mathbb{R}^d,\ b \in \mathbb{R},\ \hat{y}_i \in \mathbb{R}.
$$

No squashing — the score $w^T x + b$ *is* the prediction.

## The loss

Sum (or average) of squared errors — see [[mean-squared-error]]:

$$
\ell_i = (y_i - \hat{y}_i)^2, \qquad \mathcal{L}(w, b) = \frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2.
$$

Squared (rather than absolute) so that errors don't cancel by sign and so that the loss is differentiable everywhere.

## SLP L03 motivation: spiciness of a dish

The lecture's running example ([[30-Sources/Statistical-Learning/pdf/SLP-lec3(1).pdf#page=180|slides ~178–185]]): predict a *real-valued* spiciness score ($[0, 10]$) for a dish from a feature vector (chili-pepper count, onion, garlic, …). Each feature has a learned weight; positive weights push the prediction up, negative weights pull it down. Adding yogurt ($w_{\text{yogurt}} < 0$) reduces the score.

This is **the same diagram** as the L02 binary perceptron, just with the activation slot empty and the output interpreted as a real number.

## How to fit it

Two routes, both produce the same answer for OLS linear regression:

1. **Closed form (normal equation)** — minimize $\mathcal{L}$ by setting $\nabla_w \mathcal{L} = 0$ and solving:
   $$\hat{w} = (X^T X)^{-1} X^T y.$$
   Works because $\mathcal{L}$ is *quadratic* in $w$ and *strictly convex*. SLP L03 doesn't derive this — but mention it for context.
2. **[[gradient-descent|Gradient descent]] / [[stochastic-gradient-descent|SGD]]** — same machinery as logistic regression. The loss is convex, so GD converges to the global minimum; the only practical difference from logistic regression is the gradient (no sigmoid in the chain).

## Gradients (for GD)

For a single example:

$$
\frac{\partial \ell_i}{\partial w_j} = -2(y_i - \hat{y}_i)\, x_{i,j}, \qquad \frac{\partial \ell_i}{\partial b} = -2(y_i - \hat{y}_i).
$$

The factor $(y_i - \hat{y}_i)$ is the **residual**; gradient descent moves $w_j$ in the direction that reduces the residual proportionally to the feature's value.

## Why it's "the same neuron" as logistic regression

The lecture frames it as: *"It turns out we can use the same basic neuron model and apply it to a different kind of task, only by changing its loss function. We will also drop its non-linearity."* ([[30-Sources/Statistical-Learning/pdf/SLP-lec3(1).pdf#page=179|slide ~179]]).

| | Logistic regression | Linear regression |
| --- | --- | --- |
| Score | $z = w^T x + b$ | $z = w^T x + b$ |
| Activation | sigmoid | identity (none) |
| Output | probability $\in (0,1)$ | real number $\in \mathbb{R}$ |
| Loss | [[cross-entropy]] / [[logistic-loss]] | [[mean-squared-error\|MSE]] |
| Closed form? | no (logistic is non-linear in $w$) | yes (normal equation) |
| Output target | discrete class | continuous value |

This isomorphism is foundational: the rest of the SLP/deep-learning syllabus is "vary the loss, vary the activation, stack the units."

## Limits of linear regression

- **Linear in features only** — relationships that aren't well-approximated by a hyperplane need feature engineering (polynomial features) or non-linear models (kernels, MLPs).
- **Sensitive to outliers** — the squared term gives huge gradients to far-out examples. Robust losses (Huber, MAE) trade differentiability for outlier resistance.
- **Assumes homoscedastic Gaussian noise** under the MLE interpretation — MSE is the negative log-likelihood under $y_i \sim \mathcal{N}(w^T x_i + b, \sigma^2)$. This connection is what makes MSE the "natural" regression loss.

## Connection to L10 (loss functions, regularization)

L10 generalizes the picture: regression can use squared, absolute (MAE), Huber, or quantile losses; classification can use cross-entropy, hinge, or 0–1 surrogate losses. Linear regression is the *baseline* before regularization (Ridge = $L_2$ on $w$, Lasso = $L_1$, Elastic Net = both — see L10 / [[mock-blueprint§1e|mock §1e]] traps).

## Related

- [[mean-squared-error]] — the loss that defines linear regression.
- [[logistic-regression]] — same neuron, sigmoid activation + cross-entropy.
- [[gradient-descent]] / [[stochastic-gradient-descent]] — how to fit it iteratively.
- [[linear-classifier]] — sibling concept; linear *classifiers* threshold the score, linear *regressors* output it.
