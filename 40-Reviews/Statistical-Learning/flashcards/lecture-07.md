---
tags: [flashcards, Statistical-Learning]
course: Statistical-Learning
lecture: 07
created: 2026-05-03
---

# Lecture 07 — Training deep nets: flashcards

Why is a constant learning rate almost never optimal?
?
Early in training, the loss is dominated by gross errors and big steps make fast progress. Late in training, the network is in a narrow valley and big steps overshoot or oscillate. The right LR is large at first and small later — a schedule.

What is step decay, and what's the canonical example?
?
Multiply LR by a fixed factor $\gamma < 1$ at fixed epoch milestones. Canonical: ResNet on ImageNet, $\gamma = 0.1$ at epochs 30, 60, 90.

What is cosine decay, in formula and intuition?
?
$\eta(t) = \tfrac{\eta_0}{2}(1 + \cos(\pi t / T))$. Smoothly decreases from $\eta_0$ to $\approx 0$ over $T$ steps. Only two hyperparameters ($\eta_0$, $T$); the modern default.

What's the trade-off between step decay and cosine decay?
?
Step has more hyperparameters (when to drop, by how much) but lets you control timing precisely. Cosine has fewer hyperparameters (just $\eta_0$ and $T$) but spends very little time at the high initial LR.

If the loss curve is flat for many epochs and then drops sharply, what should you suspect first?
?
**Bad initialization.** The flat plateau is the network unable to make any progress until something disrupts the bad init; once disturbed, training proceeds normally.

If training loss decreases smoothly but validation loss is rising, what's happening?
?
Overfitting. The model is fitting noise specific to the training set. Stop training (early stopping) or add regularization.

What are the three useful curves to track during training?
?
(1) Training loss (raw, noisy per-batch). (2) Smoothed training loss — moving average of (1). (3) Validation loss — evaluated periodically on a held-out set. Stopping decisions are made on (3).

What is early stopping?
?
Stop training when the validation loss has stopped improving (or starts to rise), even if training loss is still dropping. Return the model from the epoch with the lowest validation loss.

Why does early stopping work as a regularizer?
?
With infinite training, the optimizer can drive weight magnitudes arbitrarily large to fit training noise. Early stopping caps that budget — the model can't move too far from its (small) initialization. Empirically equivalent to L2 weight decay.

In high-dimensional hyperparameter search with uneven importance, why does random search beat grid search?
?
Grid samples each axis at only $\sqrt[d]{N}$ unique values for $N$ trials. Random samples each axis at almost $N$ unique values. When some axes matter more than others, random's wider coverage on each axis dominates grid's redundant repetition.

Why search learning rate (or regularization strength $C$) on a log scale?
?
These hyperparameters affect the loss multiplicatively, so the meaningful changes are ratios, not differences. Log sampling of $\eta \in [10^{-5}, 10^{-1}]$ gives equal coverage to each decade; linear sampling concentrates ~99% of trials in the largest decade.

What does L07 mean by "non-trivial interaction between learning rate and regularization strength"?
?
Their effects are coupled — tuning them independently can miss the joint optimum. Larger LR sometimes works only with stronger regularization, and vice versa. When budget allows, search the joint space (random search, log scale) rather than fixing one and tuning the other.

What is "linear warmup → cosine decay" and when is it used?
?
Linearly increase LR from 0 to $\eta_0$ over $T_\text{warm}$ steps, then cosine-decay to $\approx 0$. Modern default for large models — the warmup avoids early-training instability when adaptive optimizers (Adam) compute bad step sizes from too-few gradient samples.

Why is the smoothed training loss curve more useful than raw training loss for monitoring?
?
Raw training loss bounces per-batch due to mini-batch variance — visually noisy, hard to read trend. Smoothed (moving average) reveals the underlying decrease without per-batch jitter, making it comparable to validation loss visually.

What is the "patience" parameter of early stopping?
?
Number of consecutive validation evaluations with no improvement before stopping. Typical $P = 3$–$10$. Without patience, a one-epoch fluctuation in validation loss can stop training prematurely.

What is the standard practical workflow for choosing hyperparameters with limited compute?
?
(1) Coarse log-scale random search over a wide range. (2) Narrow to the best region, finer search. (3) Cross-validate only the few finalists. (4) Test-set evaluation only on the final chosen model.

How does L07 connect to mock-exam §1k (SGD updates per example, not per epoch)?
?
L07 frames "one iteration = one mini-batch update." An epoch is just one full pass through the data. Mock §1k tests this directly: SGD updates after each example/mini-batch, not after the full epoch.

Why does Adam still need a learning-rate schedule on top of its per-parameter adaptation?
?
Adam adapts step *direction* (per-parameter scaling by gradient history) but the global magnitude is still set by the schedule. Without decay, Adam in a converged region oscillates around the minimum. Adam + cosine decay is the modern default.

How long should you train a deep network?
?
Until validation loss stops improving (early stopping). The right number of epochs depends on the schedule, data, and model — there's no fixed answer.

Name three concrete LR schedules besides constant.
?
Step decay, cosine decay, exponential decay (also: linear decay, $1/\sqrt{t}$ decay, linear warmup → cosine).
