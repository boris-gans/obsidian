---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Loss-functions-regs.pdf
  - course: Statistical-Learning
    file: SLP-SVMs-I.pdf
created: 2026-05-03
---

# L2 regularization (Ridge / weight decay)

Penalty $\Omega(w) = \|w\|_2^2 = \sum_j w_j^2$ added to a training loss. The defining property is **shrinkage**: the optimum has all coefficients **smaller in magnitude** than the unregularized one, but **none of them exactly zero**.

When the underlying loss is squared error, the resulting estimator is called **Ridge regression**:

$$
\min_w \; \frac{1}{n}\sum_i (w^\top x_i - y_i)^2 + \lambda \|w\|_2^2,
$$

with closed-form solution

$$
w^* = (XX^\top + \lambda I)^{-1} X y^\top.
$$

The $+\lambda I$ term is the only difference from the OLS normal equations — it makes $XX^\top + \lambda I$ invertible even when $X$ is rank-deficient (highly correlated or duplicated features). This is why Ridge is the default regularizer when you have more features than examples or near-collinear features.

## Convexity and optimization

- **Strictly convex** everywhere. The Hessian of $\|w\|_2^2$ is $2I$ — positive definite. So Ridge always has a **unique** minimizer.
- **Smooth** — differentiable everywhere; the gradient is just $2w$. Compatible with vanilla gradient descent and second-order methods.
- The slide deck phrases this: *"$L_2$ problems are smooth and strongly convex → very nice optimization behavior."*

## Why $L_2$ does NOT induce sparsity (the §1e trap)

A statement that ends wrong: *"L2 regularization adds a penalty proportional to the squared magnitude of weights, which discourages large weights and **leads to sparse models**."* True up to "discourages large weights"; false on the conclusion.

The geometric reason: the gradient of $\|w\|_2^2$ at $w_j = 0$ is exactly zero, so the penalty offers no resistance at exactly $0$ — it only fights *large* magnitudes. A small data signal pushes $w_j$ slightly off zero; the $L_2$ penalty then merely scales it down. The optimum has all components small but nonzero.

By contrast, $\|w\|_1$ has a discontinuous gradient (subgradient $\in [-1, +1]$) at zero, which can flat-out kill any data signal smaller than $\lambda$ in magnitude — that's where $L_1$'s sparsity comes from.

**Memorize**: $L_1$ → sparse. $L_2$ → small but nonzero everywhere.

## In neural networks

Often called **weight decay**: at each gradient-descent step, the regularizer contributes an extra $-\eta \lambda \cdot 2w$ term, so $w \leftarrow (1 - 2\eta\lambda)w + \eta \cdot \text{(data gradient)}$. Each step shrinks $w$ by a constant fraction toward zero before the data gradient pulls it back. The "weight decay" framing is computationally convenient but semantically identical to adding $\lambda \|w\|_2^2$ to the loss.

## In SVMs

Soft-margin SVM is a regularized objective hiding in plain sight:

$$
\min_{w, b, \xi} \; \tfrac{1}{2}\|w\|^2 + C\sum_i \xi_i.
$$

The $\tfrac{1}{2}\|w\|^2$ is $L_2$ regularization with coefficient $\tfrac{1}{2}$; the slack penalty $C\sum_i \xi_i$ plays the role of data loss (hinge loss). $C$ is the **inverse** of the standard $\lambda$ — large $C$ = small $\lambda$ = weak regularization. The geometric "max-margin" intuition is exactly equivalent to "minimize $\|w\|_2$" — a $L_2$-regularized hinge-loss minimizer.

## Empirical equivalence with early stopping

For convex losses, **early-stopping** the training run after $M$ iterations gives a $w$ close to that of $L_2$-regularized training with a corresponding $\lambda$. Both bound how far the optimizer can move from a small-magnitude initialization. This is why the validation curves look the same shape (U-shaped) whether plotted against $\lambda$ or against $M$.

## Exam-relevant facts

- $\Omega(w) = \|w\|_2^2 = \sum_j w_j^2$.
- **Shrinks** coefficients toward zero — does **not** zero them out exactly.
- **Strictly convex** → unique solution, even with correlated features.
- **Smooth** → easy to optimize with GD, closed form for Ridge.
- Closed form: $w = (XX^\top + \lambda I)^{-1}Xy^\top$.
- SVM's $\tfrac{1}{2}\|w\|^2$ is $L_2$ regularization with $C$ = inverse $\lambda$.
- "$L_2$ → sparse" is the §1e trap — **false**.

## Related

- [[regularization]] — umbrella concept.
- [[l1-regularization]] — the sparsity-inducing alternative.
- [[elastic-net]] — combines both.
- [[support-vector-machine]] — uses $L_2$ on $w$ implicitly.
- [[early-stopping]] — empirically equivalent for convex losses.
- [[lecture-10-loss-functions-regularization|SLP L10]] — source.
