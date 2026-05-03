---
tags: [concept]
courses: [NLP, Statistical-Learning]
sources:
  - course: NLP
    file: Session 12 - DeepDiveSVM.pdf
  - course: Statistical-Learning
    file: SLP-SVMs-I.pdf
  - course: Statistical-Learning
    file: SLP-Kernels(1).pdf
created: 2026-05-02
---

# Support Vector Machine (SVM)

A linear binary classifier that finds the decision boundary which **maximizes the margin** between the two classes — the distance from the boundary to the closest training points (the **support vectors**). For non-linear boundaries, SVMs use **kernel functions** to map features into a higher-dimensional space where a linear separator exists.

**Two course framings** — both correct, different in emphasis:
- **NLP (text classification framing)**: SVM as a sentiment-classifier baseline, contrasted with logistic regression on the loss/output dimension. Emphasis: hinge loss, $C$ regularizer, kernel trick as a tool for high-dim text features.
- **SLP (geometric framing)**: SVM derived from the "widest street" geometry — Vapnik's max-margin formulation, the $2/\|\vec{w}\|$ derivation, the slack-and-penalty soft-margin extension. Emphasis: convex primal, support vectors, $C$ trade-off through margin width.

The two framings are kept clearly delineated below.

## Decision rule

Same linear score form as [[logistic-regression]] and the [[perceptron]]:
$$\hat{y} = \text{sign}(\mathbf{w}\!\cdot\!\mathbf{x} + b)$$

The differences from LR are in **how the weights are trained** and in **what the score means** (uncalibrated margin, not probability).

## Training objective: margin maximization

Soft-margin SVM minimizes
$$\frac{1}{2}\|\mathbf{w}\|^2 + C \sum_i \max(0, 1 - y_i(\mathbf{w}\!\cdot\!\mathbf{x}_i + b))$$

| Term | Meaning |
|---|---|
| $\frac{1}{2}\|\mathbf{w}\|^2$ | L2 regularizer; large margin = small $\|\mathbf{w}\|$ |
| $\max(0, 1 - y(wx+b))$ | **Hinge loss**: zero penalty if correctly classified with margin $\geq 1$ |
| $C$ | Trade-off parameter between margin width and training-error tolerance |

The margin width is $2/\|\mathbf{w}\|$.

## How it differs from LR and NB

| | NB | LR | SVM |
|---|---|---|---|
| Type | Generative | Discriminative | Discriminative |
| Output | Probability | **Probability** (sigmoid) | **Margin score** (uncalibrated) |
| Training objective | Joint likelihood | **Cross-entropy** | **Margin maximization (hinge loss)** |
| Independence assumption | Yes (features given class) | No | No |

> Quiz II Q9 false-statement bait: "Training logistic regression by maximum likelihood minimizes margin width." That's **SVM-style**, not LR.

## Kernels (briefly)

For non-linear boundaries, SVMs use the **kernel trick**: replace dot products $\mathbf{x}_i \!\cdot\! \mathbf{x}_j$ with a kernel function $K(x_i, x_j)$ that implicitly computes a dot product in a higher-dimensional space. Common kernels: RBF, polynomial, sigmoid.

For NLP text classification with BoW / TF-IDF features (already very high-dimensional), **linear SVM is typically sufficient** — explicit nonlinear features are rarely needed.

## Strengths and limitations

**Strengths**
- Works well on **small, high-dimensional data** (classic NLP setting)
- Robust theoretical foundation (margin maximization → good generalization bounds)
- Strong baseline before neural models

**Limitations**
- **No native probabilistic output** — needs Platt scaling for probabilities
- Doesn't scale as gracefully as LR to very large datasets (kernel methods especially)
- Less interpretable than LR coefficients
- Has been largely superseded by LR (for probability) and neural models (for representation learning)

## Why this matters for the exam (NLP)

SVM is the **contrast class** for distinguishing LR's training objective:

- LR minimizes **cross-entropy**, NOT margin width (Quiz II Q9)
- LR has a **non-linear link function** (sigmoid); SVM does not — its decision is sign of a linear score (Quiz II.M3 Q15)

---

## SLP framing — the "widest street" geometric derivation

[[lecture-09-linear-svms|SLP L09]] derives SVM from scratch via geometry. The pivot question (Vapnik, 1990s): *"We still want to draw a straight line, but **which** straight line?"* Answer: the line that defines the widest "street" between the two classes, where the street's boundaries (the gutters) touch the closest points (the support vectors).

### Step 1: the decision rule

Imagine a vector $\vec{w}$ orthogonal to the median line of the street. For an unknown vector $\vec{u}$, the projection $\vec{w} \cdot \vec{u}$ tells us how far onto the positive side it sits. Setting $b = -C$ (where $C$ is the threshold), the decision rule becomes

$$
\text{If } \vec{w} \cdot \vec{u} + b \ge 0 \text{ then predict } +.
$$

### Step 2: the margin constraints

Free $\vec{w}$ and $b$ have too many degrees of freedom — any rescaling $(\alpha\vec{w}, \alpha b)$ gives the same boundary. **Fix the scale** by requiring positive examples to project to $\ge +1$ and negative ones to $\le -1$:

$$
\vec{w} \cdot \vec{x}^+ + b \ge 1, \qquad \vec{w} \cdot \vec{x}^- + b \le -1.
$$

With targets $y_i \in \{+1, -1\}$, both unify to

$$
y_i (\vec{w} \cdot \vec{x}_i + b) \ge 1, \qquad \forall i.
$$

Equality holds **only at support vectors** (points sitting on the gutters).

### Step 3: width = 2/‖w‖

For any positive support vector $\vec{x}^+$ and negative support vector $\vec{x}^-$, the street's width is the projection of $\vec{x}^+ - \vec{x}^-$ onto the unit normal $\vec{w}/\|\vec{w}\|$:

$$
\text{width} = (\vec{x}^+ - \vec{x}^-) \cdot \frac{\vec{w}}{\|\vec{w}\|} = \frac{(1 - b) + (1 + b)}{\|\vec{w}\|} = \frac{2}{\|\vec{w}\|}.
$$

So **maximize width = minimize $\|\vec{w}\|$ = minimize $\tfrac{1}{2}\|\vec{w}\|^2$** (the last form is convex and differentiable).

### Step 4: the hard-margin primal

$$
\min_{\vec{w}, b} \tfrac{1}{2} \|\vec{w}\|^2 \quad \text{s.t.} \quad y_i (\vec{w} \cdot \vec{x}_i + b) \ge 1, \; \forall i.
$$

Geometric intuition for why minimizing $\|\vec{w}\|$ widens the margin: if $\vec{w}$ is too big, its dot product with points already reaches 1 close to the boundary — the gutters are pulled tight to the median. Shrinking $\vec{w}$ pushes the $\pm 1$ level sets *outward* until they meet the closest support vectors.

### Step 5: soft margin / slack

Real data isn't always separable. The hard-margin constraints become **infeasible**. Introduce slack $\xi_i \ge 0$ allowing each constraint to be violated at a cost:

$$
y_i (\vec{w} \cdot \vec{x}_i + b) \ge 1 - \xi_i, \qquad \xi_i \ge 0.
$$

Subtract enough $\xi_i$ and any constraint can be satisfied — but we want minimal slack. Add penalty $C \sum_i \xi_i$ to the objective:

$$
\arg\min_{\vec{w}, b, \xi} \;\tfrac{1}{2} \|\vec{w}\|^2 + C \sum_{i=1}^{n} \xi_i \quad \text{s.t.} \quad y_i (\vec{w} \cdot \vec{x}_i + b) \ge 1 - \xi_i, \; \xi_i \ge 0.
$$

This is the **soft-margin / slack-form SVM** — the form restated inline on the SLP mock §6. Eliminating $\xi_i$ analytically gives the equivalent **hinge-loss form**:

$$
\min_{\vec{w}, b} \tfrac{1}{2}\|\vec{w}\|^2 + C \sum_i \max\!\big(0, 1 - y_i(\vec{w}\cdot\vec{x}_i + b)\big).
$$

### The C trade-off (visual progression from L09)

| C | Behavior |
| --- | --- |
| $C = 100$ | Strict — tries to classify every point with margin $\ge 1$. Narrow margin if there are outliers. Effectively hard-margin. |
| $C = 10$ | Slightly looser; one outlier may end up inside the margin but still correctly labeled. |
| $C = 1$ | Balanced — moderate margin, moderate slack tolerance. |
| $C = 0.1$ | Loose — wider margin, more slack. Outliers may be misclassified. |
| $C = 0.01$ | "Went too far here." Margin too wide; even clearly-correct points may be sacrificed. |

The lecture's note: *"notice how $\vec{w}$ becomes small as the margin increases."* Small $C$ → small $\|\vec{w}\|$ → wide margin.

### The geometric picture: support vectors are everything

A point is a **support vector** iff it sits on the gutter or inside the margin (in soft-margin: $\xi_i > 0$). **All other points have zero effect on the optimum.** Move them anywhere outside the margin and the boundary doesn't change. This is exactly mock §1c: *"the decision boundary depends only on the support vectors."*

## The dual formulation + kernel trick (L15)

L09 ends with the **soft-margin primal**. L15 derives the **dual** via Lagrange multipliers and KKT conditions, then shows that the dual depends on the data **only through inner products** — which lets us swap them for kernel evaluations and run the whole SVM in a high-dimensional feature space at the cost of an $n \times n$ matrix.

### Step 1 — Lagrangian

Introduce $\alpha_i \ge 0$ for each margin constraint and $\mu_i \ge 0$ for each slack constraint:
$$
\mathcal{L}(\vec{w}, b, \xi, \alpha, \mu) = \tfrac{1}{2}\|\vec{w}\|^2 + C \sum_i \xi_i - \sum_i \alpha_i [y_i(\vec{w} \cdot \vec{x}_i + b) - 1 + \xi_i] - \sum_i \mu_i \xi_i.
$$

### Step 2 — KKT stationarity

Setting partial derivatives to zero:
- $\partial \mathcal{L} / \partial \vec{w} = 0 \Rightarrow \vec{w}^* = \sum_i \alpha_i y_i \vec{x}_i$. **The optimal weights are a linear combination of training points** (this is the "representer-style" insight L15 reaches via GD induction; here it falls out of KKT).
- $\partial \mathcal{L} / \partial b = 0 \Rightarrow \sum_i \alpha_i y_i = 0$.
- $\partial \mathcal{L} / \partial \xi_i = 0 \Rightarrow \alpha_i + \mu_i = C$, hence $0 \le \alpha_i \le C$.

### Step 3 — the dual problem

Substitute $\vec{w}^* = \sum_i \alpha_i y_i \vec{x}_i$ back into $\mathcal{L}$ and simplify. The dual is:
$$
\boxed{\ \max_{\alpha}\ \sum_i \alpha_i - \tfrac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j\, \langle \vec{x}_i, \vec{x}_j \rangle \quad \text{s.t.}\quad 0 \le \alpha_i \le C,\ \sum_i \alpha_i y_i = 0\ }.
$$

This is a **convex QP** in $n$ variables (the dual α's), independent of feature dimension $d$. The data appears **only inside the inner product** $\langle \vec{x}_i, \vec{x}_j \rangle$ — the gateway for the kernel trick.

### Step 4 — kernelize

Replace $\langle \vec{x}_i, \vec{x}_j \rangle$ with a [[kernel-trick|kernel function]] $K(\vec{x}_i, \vec{x}_j) = \langle \phi(\vec{x}_i), \phi(\vec{x}_j) \rangle$ for some feature map $\phi: \mathbb{R}^d \to \mathbb{R}^D$ (often $D \gg d$, sometimes infinite-dimensional). The dual becomes:
$$
\max_{\alpha}\ \sum_i \alpha_i - \tfrac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j\, K(\vec{x}_i, \vec{x}_j) \quad \text{s.t.}\quad 0 \le \alpha_i \le C,\ \sum_i \alpha_i y_i = 0.
$$

We solve the SVM in $\phi$-space **without ever computing $\phi(x)$ explicitly** — only the $n^2$ values $K(x_i, x_j)$.

**Test-time prediction:**
$$
h(\vec{x}_{\text{new}}) = \mathrm{sign}\!\Big(\sum_i \alpha_i y_i\, K(\vec{x}_i, \vec{x}_{\text{new}}) + b\Big).
$$

### Why support vectors matter

KKT complementary slackness: $\alpha_i > 0 \iff$ the constraint $y_i(\vec{w}\cdot\vec{x}_i + b) = 1 - \xi_i$ is **active** — i.e., the point sits on the margin or inside it. **Support vectors** are exactly the points with $\alpha_i > 0$. All other points contribute zero to $\vec{w}^*$ and zero to predictions — you could delete them with no effect on the model. This is the formal version of mock §1c's claim *"decision boundary depends only on support vectors."*

### Common kernels

| Kernel | $K(x, z)$ | Implicit feature space |
| --- | --- | --- |
| Linear | $x^\top z$ | original — no transformation |
| Polynomial degree $p$ | $(x^\top z + c)^p$ | monomials of degree $\le p$ — $\binom{d+p}{p}$-dim |
| **Quadratic** (mock §6) | $(x^\top z + 1)^2$ | monomials of degree $\le 2$ |
| Gaussian / RBF | $\exp(-\|x - z\|^2 / 2\sigma^2)$ | **infinite-dimensional** |

L16 covers when an arbitrary $K$ is a valid kernel ([[mercer-condition]]) and the construction rules for combining kernels.

## SLP exam relevance (different from NLP)

- **Mock §1c** — decision boundary determined by support vectors. Formally: $\alpha_i > 0$ iff the KKT margin constraint is active.
- **Mock §2c** — linear SVM **cannot** achieve zero training error on XOR (not linearly separable in original space). With a quadratic or RBF kernel, it can.
- **Mock §6** — the slack primal is **restated inline on the exam**. The exam tests $C$-trade-off geometry: large $C$ → narrow margin, small $C$ → wide margin. **Quadratic kernel mechanics** $K(x, x') = (x\cdot x' + 1)^2$ embed everything into degree-≤2 polynomial feature space — see [[lecture-15-kernels-i|L15]] for the full dual derivation, [[polynomial-kernel]] for the kernel itself.
- **The "Justify your answer here" prompts** demand written reasoning. Verbalize the $C$ trade-off, the role of support vectors, and what the kernel buys you.

## Cross-course summary

| | NLP framing | SLP framing |
| --- | --- | --- |
| Motivation | Classification baseline; contrast with LR | Widest-street geometry |
| Loss | Hinge loss | $\tfrac{1}{2}\|\vec{w}\|^2$ + slack penalty |
| Primary insight | Margin maximization gives generalization; high-dim sparse features fit linear SVM | Geometry: width $= 2/\|\vec{w}\|$, support vectors determine boundary |
| Kernels | Mentioned briefly; usually linear suffices for text | Will be detailed in L15–L16 |
| $C$ trade-off | Regularization parameter | Margin-width vs. training-error tolerance |
