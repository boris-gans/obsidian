---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: Lec7-slides.pdf
created: 2026-05-03
---

# Learning-rate schedule

A rule for varying the optimizer's learning rate $\eta$ over the course of training. **A constant LR is almost never optimal.** Too low → painfully slow convergence; too high → oscillation, divergence, or settling at a high-loss point. A schedule starts large (fast progress) and decays (precise convergence).

[[lecture-07-training-deep-nets|SLP L07]] frames the motivation as a question: *"in practice, what learning rate should we use if we can't find a particular value that produces the 'good' curve? Answer: all of them. Start with a large learning rate and decay over time."* ([[30-Sources/Statistical-Learning/pdf/Lec7-slides.pdf#page=4|slides ~3–7]])

## The diagnostic curve shapes

| Curve shape | Probable cause |
| --- | --- |
| Loss flat for many epochs, then drops sharply | Bad initialization is the prime suspect ([[30-Sources/Statistical-Learning/pdf/Lec7-slides.pdf#page=18|slide ~18]]) |
| Loss decreases smoothly then plateaus too high | LR too low — never escapes the slow-progress regime |
| Loss oscillates or diverges | LR too high — overshooting |
| Train loss drops, val loss rises | overfitting — stop earlier or regularize harder |
| Smooth fast drop, slow approach to a low value | the "good" curve |

## Concrete schedules

For initial LR $\eta_0$ over total training budget $T$:

**Step decay.** Multiply LR by factor $\gamma < 1$ at fixed milestones $t_1 < t_2 < \dots$:

$$
\eta(t) = \eta_0 \cdot \gamma^{(\text{milestones passed before } t)}.
$$

Canonical example: ResNet on ImageNet uses $\gamma = 0.1$ at epochs 30, 60, 90 ([[30-Sources/Statistical-Learning/pdf/Lec7-slides.pdf#page=8|slide ~8]]). **Trade-off:** more hyperparameters (when to drop, by how much). Practical heuristic: *"look at the training curve and decay once it plateaus."*

**Cosine decay.** Smooth decay from $\eta_0$ to ~0 over $T$ steps:

$$
\eta(t) = \frac{\eta_0}{2}\!\left(1 + \cos\!\frac{\pi t}{T}\right).
$$

**Trade-off:** only two hyperparameters ($\eta_0$ and $T$), no "when to drop" decisions. The slide note: cosine *"is most commonly used; may not be the best, but usually quite reasonable until you're far along."* Potential issue: model spends very little time at the high initial LR.

**Exponential decay.** $\eta(t) = \eta_0 \cdot \alpha^t$ with $\alpha \in (0, 1)$. Smooth like cosine but never reaches zero — trails off slowly.

**Linear decay.** $\eta(t) = \eta_0 \cdot (1 - t/T)$. Simplest non-constant schedule.

**Inverse-square-root decay.** $\eta(t) = \eta_0 / \sqrt{t}$. Common in Transformer training, paired with linear warmup.

**Linear warmup → cosine decay** (the modern default for large models):

$$
\eta(t) = \begin{cases}
\eta_0 \cdot t / T_{\text{warm}} & t \le T_{\text{warm}} \\
\frac{\eta_0}{2}\!\left(1 + \cos\!\frac{\pi (t - T_{\text{warm}})}{T - T_{\text{warm}}}\right) & t > T_{\text{warm}}
\end{cases}.
$$

Warmup avoids early-training instability when adaptive optimizers (Adam) compute their per-parameter step sizes on too few gradient samples; cosine handles the long-tail decay.

## Why decay matches the loss-surface geometry

Early in training, the loss is **dominated by gross errors** — gradients are large and consistent across mini-batches, and a big step is the right step. Late in training, the network is in a **valley near a minimum** — gradients are small and inconsistent, so a big step would overshoot or oscillate. A schedule that starts large and decays matches this transition.

Equivalently: variance of the SGD gradient estimate is roughly constant per mini-batch, but as the *signal* (true gradient) shrinks near the minimum, the *signal-to-noise ratio* drops. A smaller LR damps the noise relative to the signal, which is what you want late in training.

## Pairing with adaptive optimizers (Adam etc.)

Adaptive optimizers (Adam, RMSProp, AdaGrad) already adapt step sizes per parameter using gradient history. Do you still need a schedule on top? **Yes, in practice — but the choice matters less.** The slide note: *"this is especially true if we use advanced SGD techniques (e.g., Adam)."* Adam plus cosine decay is the modern default for most networks.

## Exam-relevant facts

- A constant LR is almost never optimal — start large, decay over time.
- **Step decay**: easy to understand, more hyperparameters (milestones + factor); ResNet uses ×0.1 at epochs 30/60/90.
- **Cosine decay**: only $\eta_0$ and $T$; smooth; the modern default.
- **Loss flat for many epochs then drops** → bad initialization is the prime suspect.
- LR should be searched on a **log scale**, not linear (multiplicative effect).
- "How long to train?" — until validation loss stops improving (see [[early-stopping]]).

## Related

- [[gradient-descent]] / [[stochastic-gradient-descent]] — the optimizer the schedule modulates.
- [[hyperparameter]] — LR is a hyperparameter; schedule = hyperparameter-of-the-hyperparameter.
- [[early-stopping]] — the other side of the "when to stop" question.
- [[weight-initialization]] — bad init shows up as a flat-then-drop curve, often confused with low LR.
- [[lecture-07-training-deep-nets|SLP L07]] — source.
