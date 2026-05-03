---
type: study-guide-index
course: Statistical-Learning
created: 2026-05-03
---

# Statistical Learning study guide

A narrative layer above the lecture notes, concept notes, and flashcards. Each cluster re-tells one phase of the course as a continuous story — the lecture/concept files give you the *what*, this guide gives you the *why* and the connective tissue.

## How to use this guide

Read the clusters in order on a first pass — each one opens by referencing the previous cluster's unsolved problem, so the spine of the course only emerges if you follow the chain. For review, drop into a single cluster and use the "If you have 10 minutes" block. A few days after reading, try the **synthesis self-checks** without notes; those are the questions the §1 traps and the longer compute exercises actually test.

## The shape of the course in one paragraph

The course is a chain of complexity-control mechanisms, each invented to fix what the previous lecture's "naive" model couldn't handle. We start with **a single neuron and the perceptron's three failure modes** — binary only, doesn't terminate, returns any separating hyperplane — and patch them one at a time until we have a deep MLP we can actually train, then add the things that go wrong when you do (vanishing gradients, dead ReLUs, overfitting) and the patches for those (Cluster 1). With deep nets in hand we *deliberately set them aside* and meet two **classical alternatives** — decision trees and Vapnik's widest-street SVMs — that solve the same problem with very different machinery and seed Clusters 3–5 with raw material (stumps, support vectors, the SVM's $C$) (Cluster 2). The $C$ of L09 raises the question *how do we control model complexity in general?*, which **L10 generalizes** ($L_1$/$L_2$/Elastic Net regularizers, the §1e/§1f traps) and **L11 explains** through the **bias–variance decomposition**: variance, bias², noise, each with its own fix (Cluster 3). The L11 lens then dictates the next cluster's algorithms — bagging averages high-variance trees to kill variance, boosting additively combines high-bias stumps to kill bias, and AdaBoost is the closed-form, exponential-loss instance you'll run by hand on §5 with the **1/2 invariant** as your sanity check (Cluster 4). Boosting combines many simple learners; **kernels** take the orthogonal route — enrich a *single* learner's feature space via the kernel trick, with mock §6's quadratic-kernel SVM as the L09+L15 culmination (Cluster 5). Finally, **the labels disappear** and we turn to unsupervised structure: k-means and single-linkage hierarchical clustering for grouping (§7), PCA for summarizing (§4), and MDS / kernel-PCA / ISOMAP for non-linear extensions, all unified by the **Golden Trio** Position ↔ Similarity ↔ Distance — the unsupervised generalization of the kernel trick (Cluster 6). If you can re-tell that chain in your own words — three issues → classical alternatives → regularization & bias-variance → ensembles → kernels → unsupervised — you can write the exam.

## Clusters

1. [[01-neural-foundation]] — Neural nets foundation — covers L01–L07 — exam weight: high
2. [[02-classical-supervised]] — Classical supervised (trees, SVMs) — covers L08–L09 — exam weight: high
3. [[03-theory-glue]] — Theory glue (regularization, bias–variance) — covers L10–L11 — exam weight: high
4. [[04-ensembles]] — Ensembles (bagging, boosting, AdaBoost) — covers L12–L14 — exam weight: high
5. [[05-kernels]] — Kernel methods — covers L15–L16 — exam weight: medium-high (mock §6)
6. [[06-unsupervised]] — Unsupervised (k-means, hierarchical, PCA, MDS, ISOMAP) — covers L17–L19 — exam weight: medium-high (mock §4 + §7)

## Concept index

Alphabetical map of every SLP-tagged concept note → its home cluster. Use this when you're reviewing one concept and want to know which family it lives in.

- [[activation-function]] → cluster 1
- [[adaboost]] → cluster 4
- [[backpropagation]] → cluster 1
- [[bagging]] → cluster 4
- [[bias-variance-decomposition]] → cluster 3
- [[boosting]] → cluster 4
- [[bootstrap-sampling]] → cluster 4
- [[chain-rule]] → cluster 1
- [[computational-graph]] → cluster 1
- [[covariance-matrix]] → cluster 6
- [[cross-entropy]] → cluster 1
- [[cross-validation]] → cluster 1
- [[curse-of-dimensionality]] → cluster 1
- [[decision-tree]] → cluster 2
- [[dendrogram]] → cluster 6
- [[dropout]] → cluster 1
- [[early-stopping]] → cluster 1
- [[eigendecomposition]] → cluster 6
- [[elastic-net]] → cluster 3
- [[entropy]] → cluster 2
- [[expected-predictor]] → cluster 3
- [[exponential-loss]] → cluster 3
- [[feature-normalization]] → cluster 1
- [[gaussian-kernel]] → cluster 5
- [[generalization-error]] → cluster 3
- [[geodesic-distance]] → cluster 6
- [[gini-impurity]] → cluster 2
- [[gradient-boosting]] → cluster 4
- [[gradient-descent]] → cluster 1
- [[hidden-layer]] → cluster 1
- [[hierarchical-clustering]] → cluster 6
- [[hinge-loss]] → cluster 2
- [[huber-loss]] → cluster 3
- [[hyperparameter]] → cluster 1
- [[information-gain]] → cluster 2
- [[intrinsic-dimension]] → cluster 6
- [[isomap]] → cluster 6
- [[k-means]] → cluster 6
- [[k-nearest-neighbors]] → cluster 1
- [[kernel-pca]] → cluster 6
- [[kernel-trick]] → cluster 5
- [[l1-regularization]] → cluster 3
- [[l2-regularization]] → cluster 3
- [[learning-curve]] → cluster 3
- [[learning-rate-schedule]] → cluster 1
- [[linear-classifier]] → cluster 1
- [[linear-regression]] → cluster 1
- [[logistic-loss]] → cluster 1
- [[logistic-regression]] → cluster 1
- [[manifold-learning]] → cluster 6
- [[margin]] → cluster 2
- [[mean-squared-error]] → cluster 1
- [[mercer-condition]] → cluster 5
- [[minkowski-distance]] → cluster 1
- [[multidimensional-scaling]] → cluster 6
- [[multilayer-perceptron]] → cluster 1
- [[overfitting-underfitting]] → cluster 1
- [[perceptron]] → cluster 1
- [[polynomial-kernel]] → cluster 5
- [[position-distance-similarity]] → cluster 6
- [[principal-component-analysis]] → cluster 6
- [[random-forest]] → cluster 4
- [[regularization]] → cluster 3
- [[relu]] → cluster 1
- [[sigmoid]] → cluster 1
- [[singular-value-decomposition]] → cluster 6
- [[slack-variables]] → cluster 2
- [[softmax]] → cluster 1
- [[softmax-cross-entropy-gradient]] → cluster 1
- [[stochastic-gradient-descent]] → cluster 1
- [[support-vector]] → cluster 2
- [[support-vector-machine]] → cluster 2
- [[train-validation-test-split]] → cluster 1
- [[universal-approximation-theorem]] → cluster 1
- [[vanishing-exploding-gradients]] → cluster 1
- [[voronoi-diagram]] → cluster 1
- [[weak-learner]] → cluster 4
- [[weight-initialization]] → cluster 1

## Gaps found

Topics that the exam blueprint or lectures reference but for which no `20-Concepts/` note currently exists:

- *None.* All 78 SLP-tagged concept notes are placed; every lecture L01–L19 is referenced by exactly one cluster. The blueprint's "memorize-cold" list maps onto existing concept notes; nothing in §1–§7 of the past mock points to a missing concept.

## Coverage check

- **78** SLP-tagged concept notes in vault, **78** placed in clusters, **0** orphans
- **19** lecture notes in vault (L01–L19), **19** referenced by `covers-lectures` in exactly one cluster
- Cluster sizes (concept count): **34 / 9 / 10 / 7 / 4 / 14**. Cluster 1 (L01–L07) is the largest because Phase A is seven lectures of foundational neural-net machinery; cluster 5 (L15–L16) is the smallest because the kernel cluster is two lectures of one big idea (the kernel trick + its construction rules); cluster 6 (L17–L19) is the second-largest because L18 (PCA) and L19 (dim reduction II) introduce a lot of linear-algebra machinery
- Concepts are placed in their **first-introduction** cluster, with cross-cluster development noted via "Connections worth seeing" rather than re-listing them. Examples: [[dropout]] / [[early-stopping]] live in cluster 1 (L06 / L07 first-introduction) but reappear in cluster 3 ([[dropout]] is bagging-in-disguise) and cluster 4 (early-stopping ↔ boosting iterations $T$); [[exponential-loss]] lives in cluster 3 (L10 introduces it as part of the loss family) but is *activated* in cluster 4 (L14 AdaBoost = forward-stagewise additive modelling under exponential loss); [[bias-variance-decomposition]] lives in cluster 3 (L11) but is the lens used throughout cluster 4
