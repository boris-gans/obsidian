---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-lec3(1).pdf
created: 2026-05-03
---

# Gradient descent

The default optimization algorithm for training every model in the course that doesn't have a closed-form solution: iteratively update parameters $\theta$ in the direction of $-\nabla \mathcal{L}(\theta)$ until the loss stops decreasing.

## The update rule

In one parameter:

$$
w^{t+1} = w^{t} - \eta \, \frac{d\mathcal{L}}{dw}\bigg|_{w^t}.
$$

In $d$ parameters (the **gradient** is the vector of partial derivatives):

$$
\theta^{t+1} = \theta^{t} - \eta \, \nabla_\theta \mathcal{L}(\theta^t), \qquad \nabla_\theta \mathcal{L} = \begin{bmatrix} \partial \mathcal{L}/\partial \theta_1 \\ \vdots \\ \partial \mathcal{L}/\partial \theta_d \end{bmatrix}.
$$

The **learning rate** $\eta > 0$ controls the step size.

## L03 derivation: why gradient, not random search

[[lecture-03-intro-neural-nets|SLP L03]] motivates GD by rejecting random search ([[30-Sources/Statistical-Learning/pdf/SLP-lec3(1).pdf#page=80|slides ~75–110]]):

- **Idea 1 (random):** sample a few $(w, b)$, compute $\mathcal{L}$ for each, keep the best. Failure modes: how many tries is enough? when do you stop? scales poorly with parameter count.
- **Idea 2 (gradient):** pick an initial guess; *wiggle* each parameter to see whether the loss goes up or down; step in the direction that decreases it; repeat. This is exactly the negative gradient — the multivariate generalization of "follow the slope downhill."

The 1-D case uses the derivative (slope of the tangent line); the 2-D case uses the gradient *vector* (concatenation of two partial derivatives); the high-dim case uses the same idea, just with more partial derivatives. **Calculus computes them directly** — no numerical "wiggling" needed in practice.

## Why step size shrinks automatically near the minimum

A subtle but important property: as you approach the minimum, the *gradient itself* shrinks (the slope flattens), so the *step size* $\eta \, \|\nabla \mathcal{L}\|$ shrinks too — even with a fixed learning rate. SLP slide note: "cleverer than it seems at first glance!" ([[30-Sources/Statistical-Learning/pdf/SLP-lec3(1).pdf#page=145|slides ~145–155]]).

## Failure modes

GD can go wrong in several distinct ways ([[30-Sources/Statistical-Learning/pdf/SLP-lec3(1).pdf#page=150|slides ~150–160]]):

| Failure | Cause | Fix |
| --- | --- | --- |
| **Overshoot / diverge** | $\eta$ too large relative to local curvature | smaller $\eta$, or adaptive LR |
| **Slow convergence** | $\eta$ too small | larger $\eta$, momentum, adaptive LR |
| **Stuck in poor local min** | non-convex loss surface | momentum + SGD noise (escape shallow wells) |
| **Saddle points** | $\nabla \mathcal{L} = 0$ but not a min | momentum / 2nd-order methods |

In high dimensions, **saddle points are far more common than local minima** — most "$\nabla = 0$" critical points have a mix of positive and negative curvature directions, not a strict minimum. This is one of the structural reasons deep nets are easier to optimize than the textbook convex picture suggests.

## When GD has guaranteed convergence

For *convex* $\mathcal{L}$ (logistic regression, linear regression with squared loss, linear-SVM hinge loss with $L_2$ regularization), GD with a sufficiently small step size **converges to the global minimum**. The lecture credits Jensen's inequality for the convexity of cross-entropy under the softmax/sigmoid family ([[30-Sources/Statistical-Learning/pdf/SLP-lec3(1).pdf#page=130|slides ~130–145]]).

For **neural-net losses** (L04+), $\mathcal{L}$ is non-convex — convergence is to a *local* min only, and the loss surface "often looks like this" ([[30-Sources/Statistical-Learning/pdf/SLP-lec3(1).pdf#page=160|slide ~160]]) — bumpy, full of saddles. Practical training relies on noise (SGD), momentum, and adaptive LR to navigate it.

## Two regimes — full-batch vs. stochastic

| Regime | Gradient | Cost per step | Variance |
| --- | --- | --- | --- |
| **Full-batch GD** | $\nabla \mathcal{L} = \frac{1}{N}\sum_i \nabla \ell_i$ | $O(N)$ | zero |
| **Mini-batch GD** | $\frac{1}{|B|}\sum_{i \in B} \nabla \ell_i$ | $O(|B|)$ | moderate |
| **Stochastic GD** | $\nabla \ell_i$ for one $i$ | $O(1)$ | high |

Full-batch is the original definition; in practice always use mini-batch / SGD. See [[stochastic-gradient-descent]] for the practical version.

## Beyond plain GD

Improvements previewed in L03 (covered in detail in later lectures or skipped):

- **Momentum:** $v^{t+1} = \beta v^t + \nabla \mathcal{L}$; $\theta^{t+1} = \theta^t - \eta v^{t+1}$. Velocity carries you across saddle points and accelerates progress in consistently downhill directions.
- **2nd-order methods (Newton, BFGS):** use the Hessian for better step-size adaptation. Expensive per step.
- **Adaptive LR (AdaGrad, RMSProp, Adam):** per-parameter learning rate tuned by history of gradients. Robust to learning-rate misspecification.

L03 explicitly notes "we may look at these in more detail later" — they're not core L03 material, but Adam shows up by name in L06–L07.

## Exam-relevant intuitions

- The negative gradient $-\nabla \mathcal{L}$ is **the direction of steepest descent** in parameter space.
- The **learning rate is a hyperparameter**, not a learned parameter.
- GD by itself is **deterministic** — same initialization + same data = same trajectory. SGD adds randomness.
- For convex problems, GD finds the global optimum. For non-convex (deep nets), only a local one.

## Related

- [[stochastic-gradient-descent]] — the practical mini-batch version.
- [[backpropagation]] — how to *compute* the gradient through a neural network.
- [[mean-squared-error]] / [[cross-entropy]] / [[logistic-loss]] — the loss objects GD minimizes.
- [[learning-rate-schedule]] — strategies for varying $\eta$ over training (L07).
