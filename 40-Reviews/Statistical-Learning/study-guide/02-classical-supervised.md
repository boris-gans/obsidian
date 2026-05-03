---
type: study-guide-cluster
course: Statistical-Learning
cluster: "02-classical-supervised"
theme: "Classical supervised learning"
prerequisites:
  - 01-neural-foundation
covers-concepts:
  - decision-tree
  - entropy
  - gini-impurity
  - information-gain
  - support-vector-machine
  - support-vector
  - margin
  - hinge-loss
  - slack-variables
covers-lectures:
  - lecture-08-decision-trees
  - lecture-09-linear-svms
exam-weight: high
---

# Cluster 2: Classical supervised learning

> **The story of this cluster in one sentence.** *We leave neural networks behind for now* and meet two non-MLP classifiers — decision trees and linear SVMs — that solve the same binary-classification problem with very different machinery, both of which become foundational for everything in Clusters 3–5.

## Why this cluster exists

Phase A built one model class — the deep MLP — and the entire toolkit needed to train it. Phase B *deliberately* sets that toolkit aside. **We leave neural networks behind for now** because the next phases of the course (regularization theory, ensembles, kernels) are easier to teach with two new classifiers whose machinery is more transparent than backprop. Decision trees give us the **stumps** that AdaBoost will later boost (Cluster 4). Linear SVMs introduce the **first concrete regularizer** the student will see (the $C$ parameter), which Cluster 3 will then generalize to the full $L_1$/$L_2$/Elastic Net family. The connection back to neural nets is loose; the connection forward is tight. Two classifiers, two different geometries, both load-bearing for the rest of the course.

**Prerequisites you should feel solid on:**

- [[linear-classifier]] — the unifying object: a classifier whose decision rule is the sign of $w^\top x + b$. SVMs live here; trees do not.
- [[overfitting-underfitting]] — the U-shaped trade-off you'll see in *both* tree depth and SVM $C$.
- [[perceptron]] + [[logistic-regression]] — the prior linear classifiers; SVM is positioned as their max-margin successor.
- [[cross-validation]] — both classifiers have a key hyperparameter ($k$ for trees → max-depth / pre-prune; $C$ for SVMs); validation chooses it.

## The arc

Two lectures, two arcs that meet at "stumps and support vectors are both *the next cluster's raw material*."

### 1. [[decision-tree]] — the model and the algorithm

A [[decision-tree]] is a non-parametric, axis-aligned classifier: each **internal node** tests one attribute, each **branch** is one value of that attribute, each **leaf** predicts the class. A path from root to leaf is a *conjunction* of attribute tests; the whole tree is a *disjunction* of root-leaf paths. Training is **top-down greedy** (ID3 / C4.5 / CART, Quinlan): at the root, pick the best splitting attribute; create one child per value; sort training examples; recurse on each child until perfectly classified. The greedy choice is what the next three concepts make precise.

### 2. [[entropy]] + [[gini-impurity]] — measuring how "mixed" a node is

To pick the **best** splitting attribute we need a numerical measure of how mixed the labels at a node are. Two interchangeable choices (the prof flagged "(or Gini)" earns full credit either way on §3): [[entropy]] $H(Y) = -\sum_c p_c \log_2 p_c$ — Shannon's information-theoretic measure, maximum at the uniform distribution, zero at one-hot — and [[gini-impurity]] $G(Y) = 1 - \sum_c p_c^2$ — the probability that a random example is misclassified by a label sampled from $p$. Same shape in the binary case (peaks at $p=0.5$); Gini is cheaper because there's no logarithm.

### 3. [[information-gain]] — picking the best split

For a candidate split on attribute $X$ that partitions the data $S$ into branches $S_v$, the **conditional entropy** $H(Y \mid X) = \sum_v \frac{|S_v|}{|S|} H(Y \mid X = v)$ is the weighted-average child entropy. [[information-gain]] is the parent-minus-children reduction $\mathrm{IG}(X) = H(Y) - H(Y \mid X)$. The greedy algorithm picks the attribute with **largest IG** at each node (or equivalently, with Gini, the smallest weighted child Gini). This is the §3 mechanic — initial entropy or Gini, conditional entropy / Gini on each candidate split, choose the root attribute by IG, draw the unpruned tree, compute training-error fraction. Practice it cold; it's the most algorithmic compute on the mock for trees.

The L08 closing message — *"CART are very lightweight classifiers; very fast during testing; usually not competitive in accuracy but can become very strong through bagging (Random Forests) and boosting (Gradient Boosted Trees)"* — is the explicit handoff to Cluster 4. The depth-1 trees (**stumps**) it references will become the canonical AdaBoost weak learner.

### 4. [[support-vector-machine]] — Vapnik's widest street

L09 swings to the opposite end of Phase B: a **parametric, linear, max-margin** classifier. Same linear decision form as the perceptron and logistic regression — $\hat{y} = \mathrm{sign}(w^\top x + b)$ — but the question Vapnik asks isn't "find a separating hyperplane" (the perceptron already does that); it's *"we still want to draw a straight line, but **which** straight line?"* The answer is **Vapnik's widest street**: pick the hyperplane that defines the widest "street" between the two classes, where the street's gutters touch the closest training points.

### 5. [[support-vector]] + [[margin]] — the geometry of the optimum

Fix the scale of $(w, b)$ by requiring positives to project to $\ge +1$ and negatives to $\le -1$; with $y_i \in \{+1, -1\}$ this collapses to the canonical constraint $y_i(w^\top x_i + b) \ge 1$. The points where equality holds are the **[[support-vector|support vectors]]** — they sit exactly on the gutter. The **[[margin]]** width works out to $2/\|w\|$ (the elegant cancellation $\text{width} = (x^+ - x^-) \cdot w/\|w\| = 2/\|w\|$). Maximizing the margin = minimizing $\|w\|$ = minimizing $\tfrac{1}{2}\|w\|^2$, the differentiable form. The mock §1c claim "the decision boundary depends only on the support vectors" is the geometric punchline: move any non-SV point anywhere outside the margin and the optimum is unchanged.

### 6. [[slack-variables]] + [[hinge-loss]] + the $C$ trade-off — soft margin

Real data isn't always separable. Hard-margin SVM is *infeasible* in that case. The fix is **slack**: introduce $\xi_i \ge 0$ allowing each margin constraint to be violated at a cost. The new primal is

$$\min_{w,b,\xi}\ \tfrac{1}{2}\|w\|^2 + C\sum_i \xi_i \quad \text{s.t.}\quad y_i(w^\top x_i + b) \ge 1 - \xi_i,\ \ \xi_i \ge 0.$$

This is the form **restated inline on the mock §6** — but worth memorizing because the prof might not restate it next time. Eliminating $\xi_i$ analytically gives the equivalent **hinge-loss form** $\min_w \tfrac{1}{2}\|w\|^2 + C \sum_i \max(0, 1 - y_i(w^\top x_i + b))$. The [[hinge-loss]] $\max(0, 1-y\cdot z)$ is zero whenever the example sits comfortably on the correct side of the margin and grows linearly otherwise — sparse-gradient, which is why SVMs end up with sparse $\alpha$'s and only a small set of support vectors.

The **$C$ trade-off** is the cluster's most important MCQ-bait: large $C$ → tight constraints, narrow margin, few or no slack-violators, risk of overfit; small $C$ → wide margin, many slack-violators tolerated, risk of underfit. Mock §6 asks you to draw both extremes and **justify in writing** — the drawing alone is not enough for full credit. *Notice how $\|\vec{w}\|$ becomes small as the margin increases* (the lecture's own caption): small $C$ → small $\|w\|$ → wide margin.

## Connections worth seeing

- **Trees and SVMs sit at opposite ends of the bias–variance spectrum** — and Cluster 3 will name that spectrum. A deep tree is a high-variance, low-bias learner (memorizes training set, wobbles wildly with resampling). A linear SVM with large $C$ is the opposite (single hyperplane, low variance, possibly high bias on non-linear data). This is exactly why Cluster 4 will *bag* trees (variance-reducing) and *boost* stumps (bias-reducing) — the choice of base learner is dictated by the bias–variance term you want to attack.
- **The $C$ in soft-margin SVM is the *first concrete regularizer* in the course.** It's a knob that trades training-error tolerance for $\|w\|$ — equivalently, model complexity. L10 will generalize it: $C$ becomes $\lambda$, the slack penalty becomes a generic $\Omega(w)$ penalty, and the family expands to $L_1$ / $L_2$ / Elastic Net. Mock §1e's "L2 → sparsity (false)" trap is set up by the conceptual continuity between $C$ here and $\lambda$ there.
- **Decision stumps are AdaBoost's raw material.** This is why L08 comes before L14 in the syllabus despite the lectures being topically unrelated — by the time L14 arrives, the student needs to know what a "depth-1 tree" is, how to enumerate one (sweep one threshold per feature), and how to evaluate its weighted error. Mock §3 (full tree) and §5 (full AdaBoost) are pedagogical siblings, glued through the stump.
- **The kernel trick is implicit here.** L15 will replace the inner product $\langle x_i, x_j \rangle$ in the SVM dual with $K(x_i, x_j)$, but the *machinery that makes this possible* (the dual depends on data only through inner products) is already latent in the L09 primal. Mock §6 — **quadratic-kernel SVM with slack** — is the cross-cluster culmination: L09's slack form + L15's kernel substitution.
- **Support vectors are to SVMs what high-IG attributes are to trees.** Both are the data that *matters* — the rest can be deleted with no effect on the model. The selectivity is what makes both classifiers interpretable in ways MLPs aren't.

## Common confusions

- **Entropy vs. Gini** — interchangeable for splitting (the prof flags this on §3). Both peak at $p = 0.5$ (binary); Gini is cheaper computationally; entropy is information-theoretic. Same answer either way.
- **Hard-margin vs. soft-margin SVM** — hard-margin requires perfect linear separability and is infeasible otherwise; soft-margin adds slack $\xi_i \ge 0$ and a penalty $C \sum_i \xi_i$. The mock §6 form is soft-margin; defaulting to hard-margin will lose you the question.
- **Large $C$ vs. small $C$** — large $C$ = strict, narrow margin, few slack-violators (effectively hard-margin in the limit); small $C$ = lenient, wide margin, more violators tolerated. *Memorize this direction* — it's easy to flip under exam pressure. Mnemonic: $C$ stands for "Cost of slack" — high cost means *don't* use slack, which forces a tight fit.
- **Information gain vs. Gini-of-tree** — IG = entropy parent minus weighted entropy children (so we *maximize* IG); Gini-of-tree is the weighted-child Gini directly (so we *minimize* it). Both pick the same kind of split; sign convention differs.
- **Pre-pruning vs. post-pruning** — pre-pruning stops growth early (e.g., min-examples-per-leaf); post-pruning grows fully then collapses subtrees by validation error. Mock §1a tests pruning vs overfit — the answer hinges on knowing both exist.
- **"Boosted trees grown fully?" (mock §1l)** — for stumps (depth-1, AdaBoost's default) there's no concept of "growing fully" — they're already as small as they can be, no pruning needed. For deeper boosted trees the answer can flip; **read the question wording carefully**.
- **Hinge loss vs. logistic loss** — both are linear-classifier losses, but hinge is **zero past the margin** (sparse gradients, sparse SVs) while logistic decays smoothly to zero (every point contributes to every gradient step). This is why SVM gives you a small set of support vectors and LR doesn't.

## Self-check (synthesis, not recall)

1. **(blueprint, §1c)** "The SVM decision boundary depends only on the support vectors" — restate this in terms of the dual $\alpha_i$'s. Which $\alpha_i$'s are zero, and what does that mean for what happens if you delete a non-SV training point?
2. **(blueprint, §1e preview / §6)** A friend says "soft-margin SVM with very small $C$ is essentially the maximum-margin hyperplane." True or false, and what does small $C$ actually do to the margin width and the slack penalty?
3. **(blueprint, §3-style)** You're handed a binary-label dataset where attribute A perfectly separates class 1 from everything else (when A=Yes, class is always 1; when A=No, the rest are mixed). Compute IG(A) without doing arithmetic by reasoning from the entropy formula. Which side of the split has zero entropy?
4. **(synthesis, back to Cluster 1)** Both [[logistic-regression]] (Cluster 1) and SVM (this cluster) end up as $\hat{y} = \mathrm{sign}(w^\top x + b)$ at decision time. What differs is **how $w$ is chosen**. Name the loss function each minimizes and the geometric quantity each implicitly optimizes (probability fit vs. margin width).
5. **(synthesis, forward to Cluster 4)** Why is a **decision stump** (depth-1 tree) a good weak learner for boosting, but a poor stand-alone classifier? Phrase your answer in terms of bias and variance, even though we haven't formally introduced those words yet.
6. **(synthesis, forward to Cluster 5)** Mock §6 uses a **quadratic kernel** SVM — the polynomial kernel $K(x, x') = (x \cdot x' + 1)^2$. Without using kernels, the same task would require explicitly constructing all degree-≤2 monomial features. What does the kernel buy us computationally, and where in the L09 primal/dual does the inner product $\langle x_i, x_j \rangle$ appear in a form that's "kernel-ready"?

## If you have 10 minutes

The minimum viable review for this cluster:

1. The **§3 worked example** in [[lecture-08-decision-trees]] — practice computing initial entropy, conditional entropy on a candidate split, IG, and the unpruned tree's training error
2. The **widest-street derivation** in [[support-vector-machine]] (the SLP framing section, steps 1–5: decision rule → margin constraints → width = $2/\|w\|$ → hard primal → slack)
3. The **$C$ trade-off table** in [[support-vector-machine]] — memorize the direction (large $C$ → narrow margin) and be able to verbalize **why** for §6's "justify your answer"

## Next cluster

→ [[03-theory-glue]] — Trees and SVMs both raised the same question without quite answering it: *how do we control model complexity?* Tree depth and the SVM $C$ are two ad-hoc instances of the same dial. Cluster 3 generalizes — first by laying out the **full family of regularizers** ($L_1$, $L_2$, Elastic Net, hinge / logistic / squared / Huber / exponential losses) and then by giving the *theoretical lens* that explains what regularization is doing: the **bias–variance decomposition**.
