---
tags: [exam-blueprint, Statistical-Learning]
course: Statistical-Learning
created: 2026-05-03
sources:
  - mock-exams/mock_exam_spring26_BCSAI_SLP.pdf
---

# Statistical Learning — Exam Blueprint

> **Caveat from the prof (release email, Spring 2026):** *"Please note that I can't promise the format will be identical since this year there might be a higher number of multiple choice questions."* — Luciano Dyballa.
> **Implication:** treat this past-exam structure as a directional signal about *which topics* he tests and *how he frames* questions, not as a fixed template. Expect more MCQs and possibly fewer / shorter algorithmic-compute exercises. Memorize the underlying mechanics regardless — MCQs on AdaBoost / decision-tree splits / PCA still need the same recall.

## Format

- 8 pages, 7 numbered sections (this past exam).
- Closed book. AI tools forbidden in all quizzes/exams (per syllabus).
- Mix in this past exam: T/F + fill-blank short questions, sketch-the-curve graphs, by-hand algorithmic compute (decision trees, AdaBoost, PCA, k-means), figure annotation with written justification.
- **For the actual Spring 2026 final:** higher MCQ count likely (per prof's release note). Final format details TBD — to be discussed in class.
- No formula sheet provided in the past exam. The SVM slack-formulation primal *is* restated inline on the SVM problem; nothing else is.

## Marks distribution

Only some questions in the past exam show explicit point values. Inferred where possible.

| Section | Topic | Subparts | Points (where stated) |
| --- | --- | --- | --- |
| 1 | Short questions (T/F + fill-blank + critique) | a–l (12) | 2 each = 24 |
| 2 | Training & cross-validation error | a, b, c | 3 + 3 + 5 = 11 |
| 3 | Decision trees (mushroom dataset, by-hand) | a–e | not stated |
| 4 | PCA (3 points in 2-D, by-hand) | a, b, c | not stated |
| 5 | Boosting — full AdaBoost run | a, b | not stated |
| 6 | SVMs with quadratic kernel + slack | a, b, c | not stated |
| 7 | Clustering — k-means + hierarchical (by-hand) | a | not stated |

## Recurring patterns

**§1 — short questions (12 × 2 pts).** Overwhelmingly **True/False**, with one **fill-blank-from-list** (1b: "the model learns ___ representations" → simpler / **hierarchical** / linear / shallow) and one **critique-the-statement** (1e: a true-sounding L2-regularization claim that ends with the wrong conclusion "leads to sparse models" — student must spot it). Topics rotate across the syllabus; many test conceptual nuance, not formulas.

**§2 — qualitative curves + classifier-fits-data.** Sketch (a) train/test error vs. number of training examples, (b) train/test error vs. regularization λ, then (c) circle which classifiers in a list can achieve zero training error on a small XOR-like point cloud.

**§3 — decision-tree algorithmic compute.** Initial entropy or Gini, conditional entropy / Gini on a single split, choose the root attribute by information gain, draw the full unpruned tree, compute training-error fraction.

**§4 — by-hand PCA.** Sample mean, shape of covariance matrix, first principal-component vector — on a 3-point 2-D toy set.

**§5 — AdaBoost full execution.** From a fixed list of decision stumps, draw each stump and circle its misclassified points, then run **3 rounds** starting from uniform weights $1/n$, give the chosen stump and weighted error per round, plus the final ensemble $H$ and its error rate. **Tie-break: prefer low-numbered stumps.**

**§6 — SVM with quadratic kernel under slack.** Slack primal restated inline; $\phi$ defined so that $\phi(x)\cdot\phi(x') = (x\cdot x' + 1)^2$. Draw decision boundary for very large $C$ (low-tolerance margin) vs. very small $C$ (high-tolerance margin). Reason about generalization. **"Justify your answer here"** prompt → partial credit hangs on the written justification.

**§7 — clustering by hand.** K-means (K=2) starting from given centroid initialization — draw clusters after 1st iteration and at convergence. Hierarchical clustering with **single-linkage** — analogous 1st iter and final state. Ignore label distinction (squares vs. dots) for hierarchical.

## Topic coverage map

How each lecture/topic is tested in the past exam.

| Topic / Lecture | Where it appears | Lecture-note anchor |
| --- | --- | --- |
| **L01** k-NN, curse of dim. | §1d (1-NN training error = 0?), §2c (3-NN classifier on XOR) | `[[lecture-01-knn]]` |
| **L02** Linear classifiers / perceptron | §2c (linear SVM on XOR) | `[[lecture-02-linear-classifiers-perceptron]]` |
| **L03** Intro neural nets | §1b (depth → hierarchical reps), §2c (single-hidden-layer MLP on XOR) | `[[lecture-03-intro-neural-nets]]` |
| **L04** MLPs | §1b, §2c | `[[lecture-04-mlps]]` |
| **L05** Backprop | §1j (loss surface depends on training data) | `[[lecture-05-backprop]]` |
| **L06** Improving MLPs (vanishing/exploding, activations) | not directly tested in this mock | `[[lecture-06-improving-mlps]]` |
| **L07** LR schedules / training deep nets | §1k (SGD update semantics — random order, but per-example not per-epoch) | `[[lecture-07-training-deep-nets]]` |
| **L08** Decision trees | §1a (pruning vs. overfit), §1i (missing attributes), §1h (gradient-boosted = linear combo of stumps), §1l (boosted trees grown fully?), §2c, §3 (full compute) | `[[lecture-08-decision-trees]]` |
| **L09** Linear SVMs | §1c (decision boundary determined by SVs), §2c, §6 (quadratic-kernel SVM) | `[[lecture-09-linear-svms]]` |
| **L10** Loss & regularization | §1e (L2 → sparsity? **false**), §1f (Elastic Net & convexity), §2b (curves vs. λ) | `[[lecture-10-loss-functions-regularization]]` |
| **L11** Bias–variance | §2a (curves vs. N) | `[[lecture-11-bias-variance]]` |
| **L12** Bagging | not tested in this mock | `[[lecture-12-bagging]]` |
| **L13** Boosting | §1g (iterations as complexity control), §1h (gradient-boosted = linear combo of stumps) | `[[lecture-13-boosting]]` |
| **L14** AdaBoost | §1l (no pruning of stumps), §5 (full 3-round run) | `[[lecture-14-adaboost]]` |
| **L15** Kernels I | §6 (quadratic kernel mechanics) | `[[lecture-15-kernels-i]]` |
| **L16** Kernels II | §6 | `[[lecture-16-kernels-ii]]` |
| **L17** Clustering (k-means) | §7 (k-means + single-linkage hierarchical) | `[[lecture-17-clustering-kmeans]]` |
| **L18** PCA | §4 (mean, covariance shape, first PC) | `[[lecture-18-pca]]` |
| **L19** Dim reduction II | not tested in this mock | `[[lecture-19-dim-reduction-ii]]` |

Topics **not tested** in this past exam but still on the syllabus, so still fair game (especially under a more-MCQ format): L06 (vanishing/exploding gradients, activation choice), L12 (bagging), L19 (dim-reduction beyond PCA). Don't deprioritize them.

## Prof tells

- **Format may shift toward more MCQs this year** (per release email). Don't assume 12 T/F + 6 algorithmic compute is fixed.
- **AdaBoost tie-break rule:** "Break ties by preferring low-numbered stumps." When two stumps yield equal weighted error, pick the one with the smaller index.
- **Entropy or Gini interchangeably:** "(or Gini)" parenthetical means either impurity measure earns full credit on §3.
- **"Justify your answer here"** appears on figure-annotation questions (§6a, §6b). The drawing alone is not enough — the written reason is where partial credit lives.
- **§1 short questions mix T/F with critique-the-statement** disguised as "innocent" assertion (e.g. 1e). Read each sentence to its end before answering — wrong claims are often planted at the tail.
- **AI tools forbidden in quizzes/exams** (syllabus AI policy).
- **Pass requires ≥ 3.5 / 10 on the final** regardless of midterm/quiz performance.

## What the formula sheet provides

In this past exam: **only** the SVM slack-formulation primal, restated inline on §6:

$$
\arg\min_{w \in \mathbb{R}^p,\, b \in \mathbb{R}} \tfrac{1}{2}\|w\|^2 + C\sum_{i=1}^{n} \xi_i \quad \text{s.t.} \quad y^i\big(\langle w,\, \phi(x^i)\rangle + b\big) \ge 1 - \xi_i,\ \ \xi_i \ge 0
$$

with $\phi$ such that $\phi(x)\cdot\phi(x') = (x\cdot x' + 1)^2$ for the quadratic kernel.

Nothing else is given. Treat *every* formula below as memorize-cold.

## What the formula sheet does NOT provide

The high-leverage memorize list. None of these were given in the past exam — assume same for the final unless the prof says otherwise.

**Decision trees (L08)**
- Entropy: $H(Y) = -\sum_c p_c \log_2 p_c$.
- Gini impurity: $G(Y) = 1 - \sum_c p_c^2$.
- Conditional entropy on split: $H(Y \mid X) = \sum_v \frac{|S_v|}{|S|} H(Y \mid X = v)$.
- Information gain: $\mathrm{IG}(X) = H(Y) - H(Y \mid X)$.
- Stopping criteria + handling missing attributes (surrogate splits, fractional instances).

**AdaBoost (L14)**
- Per-round weighted error: $\epsilon_t = \sum_i w_t^{(i)} \mathbb{1}[h_t(x^i) \ne y^i]$.
- Stump weight: $\alpha_t = \tfrac{1}{2} \ln\!\frac{1 - \epsilon_t}{\epsilon_t}$.
- Weight update: $w_{t+1}^{(i)} \propto w_t^{(i)} \exp(-\alpha_t y^i h_t(x^i))$, then renormalize.
- Final classifier: $H(x) = \mathrm{sign}\!\big(\sum_t \alpha_t h_t(x)\big)$.
- Initial weights uniform: $w_1^{(i)} = 1/n$.

**PCA (L18)**
- Center: $\tilde{x}_i = x_i - \bar{x}$.
- Covariance: $\Sigma = \tfrac{1}{n}\sum_i \tilde{x}_i \tilde{x}_i^\top$ (shape $d \times d$ for $d$-dim data).
- First PC = unit eigenvector of $\Sigma$ with the largest eigenvalue.
- Pipeline mechanics for the by-hand case: mean, covariance, eigendecompose, project.

**K-means + hierarchical (L17)**
- K-means: assign each point to nearest centroid (Euclidean), recompute centroids as cluster means, iterate to convergence.
- Single-linkage cluster distance: $d(A, B) = \min_{a \in A,\, b \in B} \|a - b\|$.
- Each point starts as its own cluster at iteration 0; merge the two closest at each step.

**Loss & regularization (L10)**
- L1: $\Omega(w) = \|w\|_1$ → induces sparsity (corner solutions).
- L2: $\Omega(w) = \tfrac{1}{2}\|w\|_2^2$ → shrinks coefficients toward zero, **does not zero them out** (the §1e trap).
- Elastic Net: $\Omega(w) = \alpha\|w\|_1 + (1-\alpha)\tfrac{1}{2}\|w\|_2^2$ → both effects; **strictly convex** even with correlated features (L1 alone is only convex, not strictly).
- Hinge, logistic, squared loss canonical shapes.

**Bias–variance (L11)**
- Train error decreases with $N$; test error decreases and converges to (bias² + irreducible) — generalization gap closes.
- Vs. λ: very small λ → low train error, high test error (overfit). Very large λ → both errors rise (underfit). Test error has a U-shape in λ.

**SVM (L09 + L15/16)**
- Hard-margin: maximize $2/\|w\|$ s.t. $y^i(\langle w, x^i\rangle + b) \ge 1$.
- Soft-margin slack form (given inline on the past exam, but worth memorizing).
- Support vectors = points with $\xi_i > 0$ or on the margin; **the decision boundary depends only on these.**
- Large $C$ → small margin, fewer / no slack; small $C$ → wide margin, more slack tolerated.
- Kernel trick: replace $\langle x, x'\rangle$ with $K(x, x')$ wherever inner products appear in the dual.
- Polynomial kernel: $K(x, x') = (x\cdot x' + c)^d$. Quadratic = $(x\cdot x' + 1)^2$.

**k-NN (L01)**
- 1-NN training error is **0** (each point is its own nearest neighbor on the training set).
- Curse of dimensionality: in high $d$, all points become roughly equidistant — neighborhoods stop being informative.

**Neural nets (L03–07)**
- "Hierarchical representations" is the canonical depth-justification phrase (matches §1b answer).
- SGD = update parameters after **each** training example (in random order), not after the full epoch (the §1k trap).
- Loss surface shape depends on the training data — change the data, change the surface (§1j is **true**).
- Boosting iterations control model complexity analogously to early stopping (§1g is **true**).
- Gradient-boosted trees = additive (linear) combination of weak learners (often stumps) → boundary can be far more complex than any single stump (§1h is **true**).
- Boosted trees are typically grown **fully** without pruning when they're stumps; for deeper boosted trees the answer can flip — read the question carefully (§1l).

## Phase 2 anchor list (lecture-note slugs to create)

For cross-reference once Phase 2 ingest begins:

`lecture-01-knn`, `lecture-02-linear-classifiers-perceptron`, `lecture-03-intro-neural-nets`, `lecture-04-mlps`, `lecture-05-backprop`, `lecture-06-improving-mlps`, `lecture-07-training-deep-nets`, `lecture-08-decision-trees`, `lecture-09-linear-svms`, `lecture-10-loss-functions-regularization`, `lecture-11-bias-variance`, `lecture-12-bagging`, `lecture-13-boosting`, `lecture-14-adaboost`, `lecture-15-kernels-i`, `lecture-16-kernels-ii`, `lecture-17-clustering-kmeans`, `lecture-18-pca`, `lecture-19-dim-reduction-ii`.
