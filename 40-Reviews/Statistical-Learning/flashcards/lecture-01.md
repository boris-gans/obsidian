---
tags: [flashcards, Statistical-Learning]
course: Statistical-Learning
lecture: 01
source: SLP-Lec1-knn(1).pdf
created: 2026-05-03
---

# Lecture 01 — k-NN flashcards

What is the training error of a 1-NN classifier on its training set, and why?
?
Always 0. Each training point is its own nearest neighbour, so it predicts its own label correctly. (This is a §1d mock-exam point.)

In one sentence, what does the kNN classifier do at test time?
?
Compute the distance from the query to every stored training point, take the $k$ closest, and predict by majority vote of their labels.

How is "training" performed for kNN?
?
Trivially — just store the labelled training data. kNN is *lazy* / *instance-based*; no parameters are fit.

Write the Minkowski distance formula and identify $p=1$ and $p=2$.
?
$\mathrm{dist}_p(\mathbf{x}, \mathbf{z}) = \bigl(\sum_{r=1}^{d} |x_r - z_r|^p\bigr)^{1/p}$. $p=1$ is Manhattan ($L_1$), $p=2$ is Euclidean ($L_2$).

Why can you skip the square root in Euclidean distance when running 1-NN?
?
$\sqrt{\cdot}$ is monotonic, so $\arg\min_i \|x - x^{(i)}\|_2 = \arg\min_i \|x - x^{(i)}\|_2^2$. Saves a sqrt per training example.

Define a Voronoi diagram and connect it to nearest-neighbour classification.
?
A partition of the input space into cells, one per training point, where each cell contains the query points whose nearest training point is *that* one. The 1-NN decision boundary is the union of cell-boundary segments between cells of opposite classes.

What's the test-time computational complexity of kNN?
?
$O(kdN)$ — distance to all $N$ training points, each in $\mathbb{R}^d$, plus selecting the $k$ smallest.

How does kNN's training error change as $k$ grows?
?
It increases monotonically (roughly). At $k=1$ training error is 0; at $k=n$ it predicts the majority class for every query.

How does kNN's test (or validation) error change as $k$ grows?
?
U-shaped. High at $k=1$ (overfit / noise sensitive), drops to a minimum at some $k^*$, rises again as $k$ becomes too large (underfit). Pick $k = \arg\min$ validation error.

What goes wrong if you choose $k$ to minimise *training* error?
?
You always pick $k=1$, because 1-NN has zero training error. This is the classic "use train to pick hyperparameter" trap on slide 75.

What goes wrong if you choose $k$ to minimise *test* error?
?
You contaminate the test set: it's no longer an unbiased estimate of generalisation. Use a separate validation set (or cross-validation), and reserve test for one final evaluation.

Describe the train / validation / test split and the role of each.
?
Train: fit model parameters. Validation: pick hyperparameters / compare model variants. Test: held out, used **once** at the end for an unbiased generalisation estimate.

What is $k$-fold cross-validation and what does it estimate more robustly than a single train/val split?
?
Partition the (non-test) data into $k$ folds, train on $k-1$ and validate on the remaining 1, rotating through all folds and averaging. The average score is a lower-variance estimate of out-of-sample performance / a better signal for hyperparameter choice.

State the Cover–Hart 1967 result for 1-NN.
?
As $n \to \infty$, the 1-NN error rate is at most twice the Bayes-optimal error rate: $\epsilon_{1\text{-NN}} \le 2\,\epsilon_{\mathrm{Bayes}}$.

What is the curse of dimensionality, in one sentence?
?
In high dimensions, training data become sparse and pairwise distances concentrate, so the kNN assumption that "similar points share labels" breaks down — neighbours are no longer meaningfully near.

For data uniform in $[0,1]^d$ with $n$ training points, how big is the hypercube needed to capture the $k$ nearest neighbours of a test point?
?
$\ell \approx (k/n)^{1/d}$. For $n=1000, k=10$: $d=2 \to \ell=0.1$; $d=10 \to 0.63$; $d=100 \to 0.955$ (almost the entire space).

Why is feature normalization important for kNN but not for decision trees?
?
kNN distances weight features by their numeric range, so larger-scale features dominate. Decision trees split on per-feature thresholds and are invariant to monotonic per-feature rescalings.

Name two ways to normalize features and write the formula for each.
?
Min-max: $\tilde{x}_j = (x_j - \min_j) / (\max_j - \min_j)$. Z-score: $\tilde{x}_j = (x_j - \mu_j) / \sigma_j$.

Why does increasing $k$ make kNN more robust to label noise?
?
A single mislabelled training point can dominate the 1-NN decision in its neighbourhood. With $k > 1$, the noisy point is one vote out of $k$, so it gets outvoted by correctly-labelled neighbours.

What is a hyperparameter, and how does it differ from a parameter?
?
A hyperparameter is a choice about the algorithm that you set, not learn from data (e.g., $k$ in kNN, $\lambda$ in ridge, depth of a tree). A parameter is fit from training data (e.g., a weight vector $w$).

Name three computational mitigations for kNN's high test-time cost.
?
(1) Pre-sort training points into a fast spatial structure like a kd-tree; (2) compute approximate distances (e.g., locality-sensitive hashing); (3) condense the training set by removing redundant points; or use only a subset of dimensions.

What is Occam's razor in the context of model selection?
?
Among models that fit the training data equally well, prefer the simpler one — it generalises better in expectation. Lecture 1 illustrates this with a clean line vs. a wiggly curve through the same 4 points.

Name the three regimes shown on the standard 2D classification visual in Lecture 1.
?
**Underfitting** (boundary too simple, both errors high), **Optimal** (smooth boundary, lowest test error), and **Overfitting** (wiggly boundary chasing individual points, near-zero train error but high test error).

What's the difference between regression and classification, in one sentence each?
?
Regression: predict a continuous numeric target $y \in \mathbb{R}$. Classification: predict a discrete label $y \in \{c_1, \ldots, c_K\}$.

For a kNN classifier with quantitative features of very different scales, what's the canonical pre-processing step before computing distances?
?
Normalize each feature — e.g., z-score $(x_j - \mu_j) / \sigma_j$ or min-max $(x_j - \min) / (\max - \min)$ — so no feature dominates the distance just because of its numeric range.
