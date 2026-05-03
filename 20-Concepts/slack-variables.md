---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-SVMs-I.pdf
created: 2026-05-03
---

# Slack variables

The per-example **margin-violation budget** in soft-margin SVM. Each training point $i$ gets its own slack variable $\xi_i \ge 0$ that quantifies how much its margin constraint is being relaxed. The SVM objective minimizes the total slack (weighted by $C$) so that slack is used **only when necessary**.

## The relaxation

Hard-margin requires $y_i(\vec{w}\cdot\vec{x}_i + b) \ge 1$ for all $i$ — infeasible if the data isn't linearly separable. Soft-margin replaces this with

$$
y_i (\vec{w} \cdot \vec{x}_i + b) \ge 1 - \xi_i, \qquad \xi_i \ge 0.
$$

Subtract enough $\xi_i$ and any constraint can be satisfied — even one with $\vec{w}\cdot\vec{x}_i + b$ arbitrarily wrong. So we need to penalize using $\xi_i$.

## The penalty

Add $C \sum_i \xi_i$ to the objective:

$$
\min_{\vec{w}, b, \xi} \tfrac{1}{2}\|\vec{w}\|^2 + C \sum_{i=1}^{n} \xi_i \quad \text{s.t.} \quad y_i(\vec{w}\cdot\vec{x}_i + b) \ge 1 - \xi_i, \; \xi_i \ge 0.
$$

The optimization will set $\xi_i = 0$ whenever it can (no slack needed). When the constraint can't be satisfied with $\xi_i = 0$, it sets $\xi_i$ to the **smallest value that does** satisfy it:

$$
\xi_i^* = \max\!\big(0, 1 - y_i(\vec{w}\cdot\vec{x}_i + b)\big).
$$

This is exactly the **hinge loss**. Substituting back gives the unconstrained hinge-loss form:

$$
\min_{\vec{w}, b} \tfrac{1}{2}\|\vec{w}\|^2 + C \sum_i \max\!\big(0, 1 - y_i(\vec{w}\cdot\vec{x}_i + b)\big).
$$

## Three regimes of slack per point

The value $\xi_i^*$ tells you exactly where point $i$ sits relative to the SVM:

| Range of $\xi_i$ | Meaning |
| --- | --- |
| $\xi_i = 0$ | Margin satisfied: $y_i(\vec{w}\cdot\vec{x}_i + b) \ge 1$. The point sits on or outside its gutter — a "well-classified" point. |
| $\xi_i \in (0, 1]$ | Margin violated, but correctly classified: $0 < y_i(\vec{w}\cdot\vec{x}_i + b) < 1$. The point is inside the margin but still on the right side of the boundary. |
| $\xi_i > 1$ | **Misclassified**: $y_i(\vec{w}\cdot\vec{x}_i + b) < 0$. The point is on the wrong side of the median line. |

Soft-margin support vectors are points with $\xi_i \ge 0$ where the constraint is **active** — i.e., $y_i(\vec{w}\cdot\vec{x}_i + b) = 1 - \xi_i$, which includes both gutter-points ($\xi_i = 0$) and inside-margin / misclassified points ($\xi_i > 0$).

## The C trade-off through the slack lens

| C | Behavior on slack |
| --- | --- |
| Large ($C \to \infty$) | Each $\xi_i$ is very expensive → optimizer drives $\xi_i \to 0$ everywhere → effectively hard margin. |
| Moderate ($C \approx 1$) | Some slack accepted; balance between margin width and training error. |
| Small ($C \to 0$) | Slack is essentially free → $\xi_i$ can be large → margin can be wide and many points can sit inside it. |

The lecture's visual progression $C = 100 \to 10 \to 1 \to 0.1 \to 0.01$ shows exactly this — as $C$ shrinks, more points end up with $\xi_i > 0$ but $\|\vec{w}\|$ shrinks (margin widens).

## Why slack ≥ 0 (not just unbounded)

If $\xi_i$ could be negative, the constraint $y_i(\vec{w}\cdot\vec{x}_i + b) \ge 1 - \xi_i$ would *strengthen* — requiring margin > 1 — which makes no sense. Slack is meant to relax; $\xi_i \ge 0$ enforces "never tighter than the original."

## Slack and the dual

In the SVM dual (covered in [[lecture-15-kernels-i|L15]]), the constraint $\xi_i \ge 0$ produces a Lagrange multiplier with a closed-form upper bound:

$$
0 \le \alpha_i \le C.
$$

Points with $\alpha_i = 0$ are non-support-vectors; with $0 < \alpha_i < C$ are gutter-touching support vectors ($\xi_i = 0$); with $\alpha_i = C$ are inside-margin or misclassified support vectors ($\xi_i > 0$). This **box constraint** $[0, C]$ is the dual fingerprint of slack.

## Exam-relevant facts

- Slack variables $\xi_i \ge 0$ relax the hard-margin constraint to $y_i(\vec{w}\cdot\vec{x}_i + b) \ge 1 - \xi_i$.
- Penalty $C \sum \xi_i$ in the objective makes slack costly — only used when needed.
- $\xi_i = 0$ → margin satisfied; $\xi_i \in (0, 1]$ → inside margin, correctly classified; $\xi_i > 1$ → misclassified.
- Slack-form primal collapses to hinge-loss form when $\xi_i$ is eliminated analytically.
- Mock §6: the slack-form primal is given inline; the test is whether you can reason about $C$'s effect on margin width.

## Related

- [[support-vector-machine]] — where slack lives.
- [[hinge-loss]] — what the slack-and-penalty form collapses to.
- [[margin]] — what slack relaxes.
- [[lecture-09-linear-svms|SLP L09]] — source.
