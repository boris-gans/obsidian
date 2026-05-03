---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-lec3(1).pdf
created: 2026-05-03
---

# Stochastic gradient descent (SGD)

The practical version of [[gradient-descent|gradient descent]]: instead of computing $\nabla \mathcal{L}$ over the full dataset every step, sample a small **mini-batch** and use its gradient as a (noisy) estimate of the true gradient. Cheap per update, parallelizable on GPUs, and noise actually *helps* escape shallow local minima.

## The update rule

For a mini-batch $B \subset \{1, \ldots, N\}$ drawn from the training set:

$$
\theta^{t+1} = \theta^{t} - \eta \, \frac{1}{|B|} \sum_{i \in B} \nabla_\theta \ell_i(\theta^{t}).
$$

Two extreme cases:
- $|B| = N$ — **full-batch GD** (deterministic).
- $|B| = 1$ — **pure SGD** / "online SGD" (one example per update).

Practical training almost always uses **mini-batch SGD** with $|B| \in [32, 512]$ — large enough to use parallel hardware, small enough to inject useful noise.

## SLP L03 framing

[[lecture-03-intro-neural-nets|SLP L03]] introduces SGD as the fix for full-batch GD's two practical problems ([[30-Sources/Statistical-Learning/pdf/SLP-lec3(1).pdf#page=160|slides ~160–175]]):

1. *"Must go thru entire dataset every time to make a single update..."* — wasteful when $N$ is large.
2. *Some examples are redundant* — the model already gets them right; computing their gradient adds nothing.

Mini-batch / SGD is the answer: "instead of using the entire dataset, sample just a few for each update."

## Two ways to sample the mini-batch

The lecture lists two valid strategies ([[30-Sources/Statistical-Learning/pdf/SLP-lec3(1).pdf#page=170|slides ~170]]):

- **Sample uniformly at random every step.** Each batch is drawn independently. Examples can repeat across nearby batches.
- **Shuffle the dataset, then iterate through it linearly** (one *epoch*). Each example is seen exactly once per epoch. Reshuffle between epochs.

Both are unbiased estimators of the full gradient. The second is overwhelmingly the practical default.

## Why "stochastic" — and why noise helps

The mini-batch gradient is a *Monte Carlo* estimate of the true expected gradient over the data distribution:

$$
\frac{1}{|B|} \sum_{i \in B} \nabla \ell_i \approx \mathbb{E}_{x \sim p_{\text{data}}} [\nabla \ell(x; \theta)].
$$

Higher $|B|$ → less variance, closer to the true gradient. Lower $|B|$ → more variance, more "stochastic."

Counter-intuitively, **noise can be good**: full-batch GD steps in the direction of the true gradient *exactly*, so if that direction points into a shallow local well, you're stuck. SGD's noisy steps can "zigzag out" of shallow minima ([[30-Sources/Statistical-Learning/pdf/SLP-lec3(1).pdf#page=170|slides ~170–175]]):

> *"Because SGD calculates gradients based on small, random samples, the updates are not always directed straight toward the minimum. Instead, the updates are noisy and 'zigzag,' which allows the optimizer to jump out of shallow local minima."*

The trade-off is **more total iterations** to converge — each step is smaller in expected progress.

## The mock-§1k trap: per-example, not per-epoch

**SGD updates parameters after each training example (or each mini-batch), not after the full epoch.** Mock blueprint §1k tests this directly. The misconception is treating an "iteration" as one pass through all $N$ examples — that's an *epoch*, not an iteration. One iteration = one parameter update.

Per pass through the full dataset (one epoch), pure SGD makes $N$ updates; mini-batch SGD with $|B|$ makes $\lceil N / |B|\rceil$ updates; full-batch makes 1.

## SGD vs. full-batch GD vs. mini-batch — when each makes sense

| Method | Update cost | Per-step gradient quality | Convergence path |
| --- | --- | --- | --- |
| **Full-batch GD** | $O(N)$ | exact | smooth, deterministic |
| **Mini-batch SGD** | $O(|B|)$ | unbiased estimate | mostly smooth, occasionally jumps |
| **Pure SGD** ($|B|=1$) | $O(1)$ | high-variance estimate | jagged, can escape shallow mins |

In modern deep learning, mini-batch SGD with $|B|$ in the hundreds is universal — the only situation that prefers full-batch is *tiny* datasets where the full pass is cheap.

## SGD in non-convex landscapes (foreshadowing L05)

Once L04 introduces hidden layers, $\mathcal{L}(\theta)$ is non-convex. Plain SGD can:

- Get stuck in a poor local minimum.
- Oscillate in narrow valleys without making progress.
- Crawl across saddle points (where $\nabla = 0$ but the point isn't a minimum).

Fixes that build on SGD:
- **Momentum** — accumulates a velocity term so the optimizer "rolls past" small bumps.
- **Adaptive LR (AdaGrad / RMSProp / Adam)** — per-parameter step size from gradient history.

These are previewed in L03 and detailed in L06–L07.

## Exam-relevant facts

- SGD is **stochastic**: same dataset + same init + different RNG seed → different parameter trajectory.
- Mini-batch size is a **hyperparameter** that trades off per-step cost vs. gradient noise.
- The expected SGD update direction equals the full-batch GD update direction (unbiased).
- Noisy SGD updates **can help escape local minima** but require more total iterations.
- "SGD updates per example, not per epoch" — mock §1k.

## Related

- [[gradient-descent]] — the deterministic full-batch parent.
- [[backpropagation]] — how the per-example gradient is computed in a neural net.
- [[learning-rate-schedule]] — varying $\eta$ over training (L07).
