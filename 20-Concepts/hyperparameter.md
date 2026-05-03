---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-Lec1-knn(1).pdf
  - course: Statistical-Learning
    file: Lec7-slides.pdf
created: 2026-05-03
---

# Hyperparameter

## Definition

A **hyperparameter** is a choice about the learning algorithm that is *set* by the practitioner rather than *learned* from the training data. Contrast with **parameters**, which are the quantities the algorithm fits to data (e.g., the weights of a linear model, the centroids of a clustering).

> *"What is the best value of k to use? What is the best distance to use? These are hyperparameters: choices about the algorithm that we set rather than learn."* ([[30-Sources/Statistical-Learning/pdf/SLP-Lec1-knn(1).pdf#page=73|slide 73]])

## Examples across the course

| Algorithm | Hyperparameters | Parameters (learned) |
| --- | --- | --- |
| [[k-nearest-neighbors]] | $k$, distance metric, feature weights | None — kNN stores data |
| Decision tree | max-depth, min-samples-split, impurity criterion | The tree structure & split thresholds |
| Linear SVM | regularisation $C$ | weight vector $\mathbf{w}$, bias $b$ |
| Kernel SVM | $C$, kernel, kernel parameters | dual variables $\alpha_i$ |
| Ridge / Lasso | regularisation $\lambda$ | $\mathbf{w}$ |
| Boosting | number of rounds $T$, base learner | per-round stumps and $\alpha_t$ weights |
| MLP | architecture, learning rate, batch size, optimizer, regularisation | weights & biases |
| K-means | number of clusters $K$, initialisation | centroid positions |

## Why the distinction matters

- **You cannot tune hyperparameters on the test set.** That contaminates the test score and gives an over-optimistic estimate of generalisation.
- **You also cannot tune them on the training set.** For [[k-nearest-neighbors|kNN]] the slide deck spells out the trap: *"K = 1 always works perfectly on training data"* — picking $k$ to minimize training error always picks $k = 1$ ([[30-Sources/Statistical-Learning/pdf/SLP-Lec1-knn(1).pdf#page=75|slide 75]]).
- **The standard recipe:** [[train-validation-test-split]] — fit on train, choose hyperparameters on validation, *finally* evaluate the chosen model on test. Or use [[cross-validation]] to make better use of limited data.

## Tuning strategies

- **Grid search** — try every combination on a discrete grid.
- **Random search** — sample combinations randomly; often more efficient when only a few hyperparameters matter.
- **Bayesian / model-based search** — use past trials to guide the next one.
- **Cross-validation** — wrap any of the above in a k-fold loop to get a more robust estimate.

## Why random beats grid (L07)

[[lecture-07-training-deep-nets|SLP L07]] makes the random-vs-grid comparison concrete via the Bergstra & Bengio (JMLR 2012) figure ([[30-Sources/Statistical-Learning/pdf/Lec7-slides.pdf#page=23|slides ~21–27]]):

> *"some hyperparameters might matter more than others! ... here we are 'wasting' effort by repeating the same 3 values for the important parameter ... here we end up learning more information about the distribution for the important parameter."*

**The mechanism.** With 9 evaluations on a $3 \times 3$ grid, each axis is sampled at exactly 3 unique values — even if one axis matters far more than the other. With 9 evaluations of random search, **each axis is sampled at 9 unique values** (almost surely no repeats), so the projection onto the important axis covers more of the meaningful range.

In practice, in deep learning, importance is *almost always* uneven — learning rate matters far more than (say) momentum coefficient. **Random search dominates grid whenever you don't know which axis matters most**, which is essentially always.

## Search on a log scale, not linear (L07)

For hyperparameters that enter the loss multiplicatively — **learning rate $\eta$, regularization strength $C$ or $\lambda$, weight-decay coefficient** — search on a log scale. The L07 quiz: *"Quiz: why log(C)?"* The answer:

- Doubling $C$ (or $\eta$, etc.) has a roughly constant effect in log-loss regardless of the starting value.
- The *meaningful* changes are *ratios*, not differences.
- Linear sampling between $C = 0.01$ and $C = 100$ would put 99% of points above $C = 1$, missing the often-interesting $(0.01, 0.1)$ regime.
- Log sampling covers $(0.01, 0.1, 1, 10, 100)$ in equal steps — meaningful coverage of the meaningful range.

**Concretely**: sample $\log_{10}(\eta)$ uniformly from $[-5, -1]$ rather than $\eta$ uniformly from $[10^{-5}, 10^{-1}]$.

## L07's "non-trivial interaction" caveat

L07 closes with a subtle point ([[30-Sources/Statistical-Learning/pdf/Lec7-slides.pdf#page=29|slide ~29]]): *"non-trivial interaction between learning rate and regularization strength."* Tuning them independently can miss the joint optimum. Practical answer: when budget allows, search the joint $(\eta, \lambda)$ space (random search) rather than fixing one and searching the other.

## Choosing hyperparameters without infinite GPUs

The L07 framing emphasizes practicality:
- Coarse-then-fine: log-scale random search over a wide range first, then narrow.
- Single-fold validation suffices for early exploration; cross-validate only the few finalists.
- Reuse expensive intermediate state (pretrained weights, cached features) across trials.
- **Don't tune on the test set.** Validation set holds the line.

## Related

- [[train-validation-test-split]]
- [[cross-validation]]
- [[overfitting-underfitting]]
- [[learning-rate-schedule]] — schedule choice is itself a hyperparameter.
- [[early-stopping]] — orthogonal regularization that bounds training duration.
- [[lecture-07-training-deep-nets|SLP L07]] — source for the random-vs-grid + log-scale framing.
