---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Bagging.pdf
created: 2026-05-03
---

# Bootstrap sampling

**Sampling with replacement** from a fixed dataset to create new datasets of the same size that simulate fresh draws from the underlying distribution. Foundational statistical resampling tool — the "Bootstrap" in [[bagging]].

## The procedure

Given a dataset $D = \{(x_1, y_1), \ldots, (x_N, y_N)\}$:

1. To form one bootstrap sample $D^*$, draw $N$ examples from $D$ **with replacement** — i.e., each draw picks any element of $D$ uniformly at random, then puts it back before the next draw.
2. $D^*$ has the same size as $D$ but typically contains **repeated points** (some chosen multiple times) and **omits others** (some never drawn).
3. Repeat $m$ times to get $D_1^*, \ldots, D_m^*$ — each is one bootstrap sample.

The lecture's emphasis: *"sample with replacement, so total number of points can be the same for all $D_j$! (we will end up with repeated points, but that's okay)"* ([[30-Sources/Statistical-Learning/pdf/SLP-Bagging.pdf#page=10|slide ~10]]).

## How much of $D$ ends up in one bootstrap?

For a single point $x_i$ in $D$, the probability of *not* being drawn in any of $N$ independent draws is $(1 - 1/N)^N \to 1/e \approx 0.368$ as $N \to \infty$. So:

- **~63.2%** of unique original points appear at least once in $D^*$.
- **~36.8%** are **out-of-bag** (OOB) — never drawn.

These OOB points serve as a free held-out set for each bootstrap-trained predictor, the basis of [[bagging|bagging]]'s OOB error estimate.

## Why bootstrap mimics fresh draws from $P$

Without infinite draws from the true distribution $P$, bootstrap is the best simulation of "what would another dataset look like?" Each $D_j^*$ has the same empirical distribution as $D$ in expectation, but with random fluctuations — exactly what we'd see if we collected a fresh sample of $N$ points from $P$. The catch: bootstrap **can only capture variation already present in $D$**, so it underestimates variability that depends on rare events $D$ doesn't contain.

## Use beyond bagging

Bootstrap predates bagging by decades and shows up in many statistical contexts:

- **Confidence intervals** for any statistic $T(D)$: compute $T(D_1^*), \ldots, T(D_m^*)$, take percentiles.
- **Standard error estimates** for an estimator without analytical formulas.
- **Bagging** ([[bagging]]) — train one predictor per bootstrap, average them.
- **OOB error estimation** for ensemble methods.

## Bootstrap vs. cross-validation

Both are resampling-based estimates of test error:

| | **Bootstrap (OOB)** | **Cross-validation** |
| --- | --- | --- |
| Sampling | with replacement, full-size resamples | without replacement, partition into folds |
| Held-out per draw | ~37% (OOB) | $1/K$ (one fold at a time) |
| Computational cost | one model per bootstrap | one model per fold |
| Bias of estimate | slightly pessimistic (OOB sees fewer unique points) | depends on $K$ |

## Exam-relevant facts

- Bootstrap = sampling **with replacement** to form same-size resamples.
- Each bootstrap contains ~63% of unique original points; ~37% are out-of-bag.
- Foundation of [[bagging]] — the "B" in "BAgging".
- Repeat points in a bootstrap are fine — that's the whole point.

## Related

- [[bagging]] — the canonical use.
- [[random-forest]] — combines bootstrap with random feature subsets.
- [[cross-validation]] — alternative resampling strategy.
- [[lecture-12-bagging|SLP L12]] — source.
