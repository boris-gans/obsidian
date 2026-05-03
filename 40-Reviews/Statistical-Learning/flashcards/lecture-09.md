---
tags: [flashcards, Statistical-Learning]
course: Statistical-Learning
lecture: 09
created: 2026-05-03
---

# Lecture 09 — Linear SVMs: flashcards

What is the SVM decision rule?
?
$\hat{y}(\vec{u}) = \text{sign}(\vec{w}\cdot\vec{u} + b)$. Same form as the perceptron and logistic regression — what differs is how $\vec{w}$ and $b$ are trained.

What is the SVM "widest street" intuition (Vapnik)?
?
Among all linear separators, pick the one whose perpendicular distance to the closest training point is maximal — equivalently, the one that defines the widest "street" between the two classes. The street's gutters touch the closest points (the support vectors).

What are the SVM margin constraints?
?
$\vec{w}\cdot\vec{x}^+ + b \ge 1$ for positives and $\vec{w}\cdot\vec{x}^- + b \le -1$ for negatives. With $y_i \in \{+1, -1\}$, both unify to $y_i(\vec{w}\cdot\vec{x}_i + b) \ge 1$.

What is the formula for SVM margin width? Derive it briefly.
?
Width $= 2/\|\vec{w}\|$. Project $\vec{x}^+ - \vec{x}^-$ onto $\vec{w}/\|\vec{w}\|$: $\text{width} = (\vec{x}^+ - \vec{x}^-) \cdot \vec{w}/\|\vec{w}\| = ((1-b) + (1+b))/\|\vec{w}\| = 2/\|\vec{w}\|$.

What is the SVM hard-margin primal problem?
?
$\min_{\vec{w}, b} \tfrac{1}{2}\|\vec{w}\|^2$ subject to $y_i(\vec{w}\cdot\vec{x}_i + b) \ge 1$ for all $i$.

Why does minimizing $\|\vec{w}\|$ widen the margin?
?
Margin width is $2/\|\vec{w}\|$ — smaller $\|\vec{w}\|$ literally means wider margin. Geometrically: small $\|\vec{w}\|$ means the level sets $\vec{w}\cdot\vec{x} + b = \pm 1$ are pushed far apart; the optimization pushes them outward until they hit the closest training points.

What is the SVM soft-margin / slack primal?
?
$\min_{\vec{w}, b, \xi} \tfrac{1}{2}\|\vec{w}\|^2 + C\sum_i \xi_i$ s.t. $y_i(\vec{w}\cdot\vec{x}_i + b) \ge 1 - \xi_i$, $\xi_i \ge 0$. The slack $\xi_i$ relaxes each constraint at cost $C$.

What does the slack variable $\xi_i$ tell you about point $i$?
?
$\xi_i = 0$: margin $\ge 1$ (well-classified). $\xi_i \in (0, 1]$: inside the margin but correctly classified. $\xi_i > 1$: misclassified. The optimization sets each $\xi_i$ to the smallest value satisfying its constraint.

What happens in the SVM as $C$ increases?
?
The margin constraints are enforced more strictly — narrow margin, fewer / smaller slacks, tries to classify every point. In the limit $C \to \infty$, this becomes hard-margin (infeasible if the data isn't separable).

What happens in the SVM as $C$ decreases?
?
The margin constraints are loosened — wide margin, more slack tolerated, $\|\vec{w}\|$ shrinks. At $C \to 0$ the SVM ignores the data entirely and just minimizes $\|\vec{w}\|$.

What is the SVM "hinge-loss" formulation, and how does it relate to slack?
?
$\min_{\vec{w}, b} \tfrac{1}{2}\|\vec{w}\|^2 + C\sum_i \max(0, 1 - y_i(\vec{w}\cdot\vec{x}_i + b))$. Eliminating the optimal slack $\xi_i^* = \max(0, 1 - y_i(\vec{w}\cdot\vec{x}_i + b))$ from the slack-form primal gives this exactly.

What is a support vector?
?
A training point whose constraint is **active** at the optimum: it sits exactly on the margin gutter (hard margin), or has $\xi_i > 0$ (soft margin — inside the margin or misclassified). Non-SV points have no effect on the SVM.

What does mock §1c claim about the SVM decision boundary?
?
"The decision boundary depends only on the support vectors." **True.** In the dual, $\alpha_i > 0$ ⟺ support vector; the optimal $\vec{w} = \sum_{i\in\text{SV}} \alpha_i y_i \vec{x}_i$ depends only on SVs.

Can a linear SVM achieve zero training error on XOR-like point cloud (mock §2c)?
?
**No.** XOR is not linearly separable. A linear SVM cannot achieve zero training error in the original space — you need a kernel (L15–L16) or a feature transformation.

What is the difference between hinge loss and logistic loss for far-from-boundary correctly-classified points?
?
**Hinge** is exactly zero for points with margin $\ge 1$ — they contribute nothing to the gradient, giving SVM its sparsity (only SVs matter). **Logistic** is small but positive everywhere — every training point still contributes to the gradient.

How does the SVM regularize?
?
Implicitly, through $\tfrac{1}{2}\|\vec{w}\|^2$ in the primal — this is exactly L2 regularization. The hyperparameter $C$ is the inverse regularization strength: large $C$ prioritizes data fit (less regularization); small $C$ prioritizes large margin (more regularization).

Why is the SVM more robust to outliers than logistic regression?
?
A far-away outlier with $y_i(\vec{w}\cdot\vec{x}_i + b) \gg 1$ contributes zero to the SVM's hinge loss → its presence doesn't move the boundary. The same point in LR contributes a small but nonzero logistic loss, which can drag the boundary toward it.

What is the geometric meaning of the SVM dual variable $\alpha_i$?
?
$\alpha_i = 0$ for non-support-vectors. $0 < \alpha_i < C$ for gutter-touching support vectors with $\xi_i = 0$. $\alpha_i = C$ for inside-margin / misclassified support vectors with $\xi_i > 0$. The box $[0, C]$ is the dual signature of slack.

Where does the slack-form SVM appear in the past mock exam?
?
**§6** — the soft-margin SVM primal is *restated inline* on the past exam, with the quadratic-kernel feature map $\phi(x)\cdot\phi(x') = (x\cdot x' + 1)^2$. The exam tests $C$-trade-off geometry: large $C$ → narrow margin, small $C$ → wide margin.

What's the structural-risk-minimization argument for max-margin classifiers?
?
Generalization bounds depend on the *margin*, not the input dimensionality. For a linearly separable dataset with margin $\gamma$, the VC-dimension of the max-margin classifier scales as $1/\gamma^2$ — independent of dimension. So a wide-margin SVM in 10000 dimensions can still generalize well.

Compare SVM, LR, and the perceptron in one line each.
?
Perceptron: minimize misclassifications via mistake-driven updates. LR: minimize cross-entropy → calibrated probability output. SVM: maximize margin (minimize $\tfrac{1}{2}\|\vec{w}\|^2$ + hinge loss) → sparse, support-vector-determined boundary.
