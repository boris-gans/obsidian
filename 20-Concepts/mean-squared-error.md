---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-lec3(1).pdf
  - course: Statistical-Learning
    file: SLP-Loss-functions-regs.pdf
created: 2026-05-03
---

# Mean squared error (MSE)

The default regression loss: average of squared residuals between predictions and targets. **Convex**, **differentiable everywhere**, and the **MLE objective** under Gaussian-noise regression.

## Definition

For predictions $\hat{y}_i$ and targets $y_i$:

$$
\ell_i = (y_i - \hat{y}_i)^2, \qquad \mathrm{MSE}(\hat{y}, y) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2.
$$

Summed (no $1/N$) it's "sum of squared errors" (SSE) — same loss up to a constant rescaling that GD absorbs into the learning rate.

## Why squared, not absolute?

[[lecture-03-intro-neural-nets|SLP L03]] motivates the choice ([[30-Sources/Statistical-Learning/pdf/SLP-lec3(1).pdf#page=185|slide ~185]]):

> *"Simply adding the differences won't work — different signs cancel out. Idea: use squared errors instead."*

Squaring solves the sign-cancellation problem **and** makes the loss differentiable at $\hat{y} = y$ (the absolute-value loss has a kink there). The differentiability matters because [[gradient-descent|GD]] needs $\nabla \mathcal{L}$ to exist everywhere.

## Properties

- **Always non-negative**, $\mathrm{MSE} \geq 0$, equals 0 only when predictions are exact.
- **Convex** in the predictions and (for [[linear-regression|linear regression]]) in the parameters $w$. So GD converges to the global minimum.
- **Differentiable**: $\partial \ell_i / \partial \hat{y}_i = -2(y_i - \hat{y}_i)$. The gradient is proportional to the residual.
- **Sensitive to outliers**: a residual of $10$ contributes $100$; a residual of $1$ contributes $1$. One outlier dominates the loss.

## Statistical interpretation

MSE = negative log-likelihood under $y_i \sim \mathcal{N}(\hat{y}_i, \sigma^2)$:

$$
-\log p(y \mid \hat{y}) = \frac{(y - \hat{y})^2}{2\sigma^2} + \text{const}.
$$

So minimizing MSE is **maximum likelihood** under the Gaussian-residual assumption. This mirrors the relationship between [[cross-entropy]] and Bernoulli MLE in classification — same MLE machinery, different noise model.

## Gradient (for plug-into-GD)

For a single example with $\hat{y}_i = w^T x_i + b$:

$$
\nabla_w \ell_i = -2(y_i - \hat{y}_i)\, x_i, \qquad \frac{\partial \ell_i}{\partial b} = -2(y_i - \hat{y}_i).
$$

The factor $(y_i - \hat{y}_i)$ is the residual — GD pulls $w$ toward features whose residuals are large in magnitude.

## Comparison with other regression losses (preview L10)

| Loss | Formula | Outlier sensitivity | Differentiable everywhere? |
| --- | --- | --- | --- |
| **MSE / $L_2$** | $(y - \hat{y})^2$ | high | ✓ |
| **MAE / $L_1$** | $|y - \hat{y}|$ | low (bounded gradient) | ✗ (kink at 0) |
| **Huber** | quadratic near 0, linear far away | medium | ✓ |
| **Quantile** | asymmetric piecewise linear | medium (skewed) | ✗ (kink) |

L10 will go deeper into when each of these is the right call. The default in L03 is MSE.

## L10's framing — squared loss estimates the MEAN

[[lecture-10-loss-functions-regularization|SLP L10]] reframes the squared/absolute split through what they're estimating ([[30-Sources/Statistical-Learning/pdf/SLP-Loss-functions-regs.pdf#page=22|slide 22]]):

- **Squared loss → estimates the conditional mean** $\mathbb{E}[y \mid x]$. The constant $c$ minimizing $\sum_i (y_i - c)^2$ is the sample mean.
- **Absolute loss → estimates the conditional median**. The constant $c$ minimizing $\sum_i |y_i - c|$ is the sample median.

This is the cleanest one-line answer to "why squared, not absolute?": **what statistic of $y \mid x$ do you actually want?** If your data has heavy-tailed noise (a few examples with extreme $y$), the mean is dominated by them and squared loss inherits the bias — use absolute or [[huber-loss]] instead. If noise is symmetric and well-behaved, squared loss is the maximum-likelihood choice and gives the best estimate.

L10 also names this the **OLS** loss when paired with no regularizer — and the closed-form $w = (XX^\top)^{-1}Xy^\top$ is the solution.

## Why it's the L03 closing case

L03 ends with regression because it shows the unifying claim of the course: **"the same neuron, the same training machinery — just change the loss and the activation."** Linear regression is the simplest demonstration of that unification. The arc of the rest of the course is repeating this trick with different losses and activations to get richer models.

## Related

- [[linear-regression]] — the model MSE is paired with.
- [[gradient-descent]] / [[stochastic-gradient-descent]] — how MSE-trained models are fit.
- [[cross-entropy]] / [[logistic-loss]] — the classification analogue (sister MLE objective).
