---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Boosting.pdf
  - course: Statistical-Learning
    file: SLP-Adaboost(1).pdf
created: 2026-05-03
---

# Weak learner

A predictor whose accuracy on the training distribution is **strictly greater than 50%** for binary classification — i.e., any predictor that does **better than random guessing**, however slightly. The atomic building block of [[boosting]].

For regression, the analogous condition is that the predictor's vector $\vec{h} \in \mathbb{R}^N$ in label space has nonzero, correctly-signed inner product with the residual direction $\vec{y} - \vec{H}_{t-1}$ (i.e., angle $< 90°$).

## Why "weak" is sufficient

The boosting framework iteratively combines many weak learners into a strong predictor. The reason a weak edge over chance is enough:

- Each weak learner moves the ensemble's prediction $\vec{H}$ a small distance toward the target $\vec{y}$ in label space.
- Pythagoras: as long as the move's direction is within 90° of the residual direction, the new distance to $\vec{y}$ is strictly smaller than before (with appropriate step size).
- $T$ such steps compose into a meaningful displacement.

So **iteration count compensates for individual weakness**. Even 50.1%-accurate learners suffice — boosting will eventually produce a strong ensemble, just with more rounds.

## The 50% boundary

The unique pathological case is **exactly 50% accuracy on a binary task** — the weak-learner vector $\vec{h}$ is **perfectly orthogonal** to the residual. No amount of iteration helps; boosting terminates because no progress is possible.

If a candidate weak learner has accuracy $< 50\%$, **flip its sign** (predict the opposite of what it says). The flipped predictor has accuracy $1 - \text{err} > 50\%$, hence is a valid weak learner. The only case where flipping doesn't help is the exact 50% case (orthogonal $\vec{h}$), which is the algorithm-termination condition.

## Canonical weak learners

| Algorithm | Default weak learner |
| --- | --- |
| **AdaBoost** (L14) | **Decision stumps** (depth-1 trees — one feature test, two leaves) |
| **Gradient boosting** (L13) | Shallow regression trees (depth 3–8) |
| Generalized boosting | Any predictor with > 50% accuracy — linear classifiers, naive Bayes, etc. |

In practice, **decision stumps** are the textbook AdaBoost weak learner because they're trivial to train (sweep one threshold per feature) and almost always achieve > 50% accuracy on any nontrivial dataset.

## Weak vs strong learners — the bias-variance angle

| | **Weak learner** | **Strong learner** |
| --- | --- | --- |
| Capacity | low (e.g., depth-1 tree) | high (e.g., fully-grown tree) |
| Bias | **high** (underfits) | low |
| Variance | low (robust to data perturbation) | **high** (overfits) |
| Used by | boosting (combines many to reduce bias) | bagging (averages many to reduce variance) |

The two ensemble paradigms attack opposite halves of the bias-variance trade-off, and use opposite kinds of base learners as a result.

## Historical context: PAC learning

The original 1988 question (M. Kearns): *given a learning algorithm with accuracy guaranteed to be only slightly better than 50%, can we boost it to arbitrarily high accuracy?* The 1990 answer (R. Schapire): yes — and AdaBoost (1995) was the first practical instance. This is now a foundational result in computational learning theory.

## Exam-relevant facts

- Weak learner = predictor with accuracy **strictly > 50%** on binary classification (any edge).
- 50% (orthogonal) is the only pathological case — boosting terminates there.
- A "below-50%" predictor flips into a valid weak learner by sign change.
- Decision stumps are the canonical AdaBoost weak learner; shallow trees for gradient boosting.
- Weak learners have **high bias / low variance** — opposite of bagging's preferred base learners.

## Related

- [[boosting]] — the framework that combines weak learners.
- [[gradient-boosting]] — uses shallow regression trees.
- [[adaboost]] — uses decision stumps and exponential loss; the canonical weak-learner consumer.
- [[decision-tree]] — the base learner family.
- [[bias-variance-decomposition]] — why high-bias / low-variance is the right pairing.
- [[lecture-13-boosting|SLP L13]] / [[lecture-14-adaboost|SLP L14]] — sources.
