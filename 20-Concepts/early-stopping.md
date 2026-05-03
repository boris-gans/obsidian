---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: Lec7-slides.pdf
  - course: Statistical-Learning
    file: SLP-Loss-functions-regs.pdf
created: 2026-05-03
---

# Early stopping

A regularizer that limits training duration: **stop training when the validation loss stops improving** (or starts to rise), even if the training loss is still decreasing. Equivalent to picking the model from the epoch with the lowest validation loss, regardless of the epoch budget.

## The mechanism

Track three curves during training ([[30-Sources/Statistical-Learning/pdf/Lec7-slides.pdf#page=18|slides ~16–20]]):

| Curve | Purpose |
| --- | --- |
| **Training loss (raw)** | Per-batch gradient signal. Noisy due to mini-batch variance. |
| **Smoothed training loss** | Moving average of training loss over recent iterations. Reveals the underlying trend without per-batch jitter. |
| **Validation loss** | Computed on a held-out validation set every $k$ epochs. The decision-making curve. |

Stop when validation loss has stopped decreasing for some patience window (e.g., no improvement in 5 consecutive validation evaluations). Restore the best-validation-loss checkpoint as the final model.

## Why early stopping regularizes

Training loss decreases monotonically in expectation (the optimizer is minimizing it). Validation loss typically *also* decreases at first, then **starts to rise** as the model begins fitting noise specific to the training set — overfitting. Early stopping picks the inflection point, which is the bias–variance sweet spot for the current architecture and data.

Geometrically: training-loss minima and validation-loss minima are *not* the same point in parameter space. The validation-loss minimum is closer to the start (less overfit). Early stopping selects this nearer point implicitly by halting before reaching the training-loss minimum.

## Connection to L2 regularization

Empirically and theoretically, early stopping behaves similarly to **$L_2$ weight decay** — both limit how far the optimizer can travel from the (small-magnitude) initialization. With infinite training budget, an unregularized loss can drive $\|w\|$ arbitrarily large; early stopping caps the budget. For convex losses, the equivalence is formal (the early-stopped trajectory traces the same path as $L_2$-regularized solutions for a corresponding $\lambda$).

## Practical recipe

1. Hold out a validation set (~10–20% of training data, never seen during gradient updates).
2. Evaluate validation loss every epoch (or every $k$ batches if epochs are huge).
3. Track the best validation loss seen so far + the epoch at which it occurred + a checkpoint.
4. Stop when validation loss has not improved in $P$ consecutive evaluations (the *patience* parameter, typically 3–10).
5. Return the checkpoint at the best-validation epoch.

## Why "smoothed training loss" matters too

Raw training loss bounces around per-batch — useful for spotting catastrophic divergence (loss → NaN), but not for trend judgment. The smoothed curve (moving average) is what you visually compare against validation loss. **The relationship between smoothed training and validation tells you what's happening:**

- Both decreasing → training as intended.
- Smoothed training decreasing, validation flat → fitting noise; early stopping triggers soon.
- Smoothed training decreasing, validation rising → overfitting; trigger now.
- Smoothed training flat → LR too low or stuck near initialization (orthogonal issue).

## Exam-relevant facts

- Early stopping = pick the model from the epoch with the lowest **validation** loss.
- Track three curves: training (raw), smoothed training (moving average), validation.
- Stops overfitting because training/validation loss diverge once the model starts memorizing.
- Empirically equivalent to $L_2$ regularization — both bound how far from init the weights can move.

## L10's framing — iteration count $M$ as an implicit regularizer

[[lecture-10-loss-functions-regularization|SLP L10]] presents validation curves twice — once vs. the explicit penalty $\lambda$, once vs. the iteration count $M$ — and they have **identical shape**. Both are U-shaped in validation error: small $M$ (or large $\lambda$) underfits, large $M$ (or small $\lambda$) overfits, sweet spot in the middle.

This is the slide-deck way of saying: **iteration count is itself a regularization knob**. You can either:

- Add an explicit penalty $\lambda \Omega(w)$ to the loss and train to convergence, OR
- Train without a penalty and stop early at $M^*$.

For convex losses the two approaches are formally equivalent — early-stopped trajectories trace the same path as $L_2$-regularized solutions for a corresponding $\lambda$.

## Related

- [[hyperparameter]] — patience and validation-eval frequency are themselves hyperparameters.
- [[overfitting-underfitting]] — the failure mode early stopping prevents.
- [[train-validation-test-split]] — early stopping requires a validation set distinct from train and test.
- [[learning-rate-schedule]] — orthogonal "when to do what" question.
- [[lecture-07-training-deep-nets|SLP L07]] — source.
