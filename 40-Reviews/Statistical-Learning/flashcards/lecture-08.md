---
tags: [flashcards, Statistical-Learning]
course: Statistical-Learning
lecture: 08
created: 2026-05-03
---

# Lecture 08 — Decision trees: flashcards

What is the structure of a decision tree (internal nodes, branches, leaves)?
?
Internal nodes test one discrete-valued attribute $X_i$. Branches from a node correspond to one value of $X_i$. Leaves predict the class label $Y$ (or class probabilities $P(Y \mid X \in \text{leaf})$).

What is the formula for entropy $H(Y)$ as label-distribution impurity?
?
$H(Y) = -\sum_c p_c \log_2 p_c$. Pure node: $H = 0$. Uniform binary: $H = 1$. Uniform $C$-way: $H = \log_2 C$.

What is the formula for Gini impurity $G(Y)$?
?
$G(Y) = 1 - \sum_c p_c^2 = \sum_c p_c (1 - p_c)$. Pure: $G = 0$. Uniform binary: $G = 0.5$.

What is the formula for conditional entropy on a candidate split?
?
$H(Y \mid X) = \sum_v \frac{|S_v|}{|S|} H(Y \mid X = v)$ — weighted average of child entropies, weighted by branch fraction.

What is information gain, and what does the ID3/C4.5 algorithm do with it?
?
$\mathrm{IG}(X) = H(Y) - H(Y \mid X)$ — entropy reduction from splitting on $X$. The greedy algorithm picks the attribute with the **largest IG** at each node.

What's the mock-exam shortcut for "Entropy or Gini"?
?
The prof flagged "(or Gini)" parenthetically — either impurity measure earns full credit on §3. Use whichever is faster to compute by hand (Gini avoids logarithms).

What is the top-down induction algorithm (ID3 / C4.5 / CART)?
?
At each node: (1) pick best attribute $A$ by argmax IG, (2) assign $A$ to the node, (3) create one descendant per value of $A$, (4) sort training examples to leaves, (5) if pure, stop; else recurse on each new leaf.

What are decision stumps, and where do they appear?
?
Depth-1 decision trees — one split, two leaves. The canonical weak learner for AdaBoost (L14). When mock §1l asks "are boosted trees grown fully?" the answer for stumps is "yes — depth 1, no pruning needed."

Why do decision trees overfit so aggressively?
?
A fully grown tree can perfectly memorize the training set in $O(n)$ leaves (one leaf per example). Trees have very high variance — small data perturbations produce wildly different trees.

What does pruning do to a decision tree's training and test error?
?
Pruning **increases** training error (the tree fits the training set worse) but **decreases** test error (better generalization). This is the mock §1a trap — pruning trades training fit for generalization.

How do decision trees handle missing attribute values? (mock §1i)
?
Two main strategies: **surrogate splits** (CART) — store backup attributes whose splits agree with the primary; **fractional instances** (C4.5) — send a fractional copy of the example down each branch, weighted by the branch's data proportion. Both are correct; "ignore the example" is wrong.

Can a decision tree achieve zero training error on any consistent dataset? (mock §2c)
?
Yes — given enough depth, a decision tree can shatter any finite set of training examples. (One leaf per example in the worst case.) This is the same property as 1-NN.

What is the IG bias toward many-valued attributes, and what fixes it?
?
An attribute with one distinct value per training example trivially gives $H(Y \mid X) = 0$ — IG is maximized but the tree is useless. Fix: **gain ratio** in C4.5, which divides IG by the entropy of the partition itself, penalizing high-cardinality attributes.

What is a regression tree, and how does it differ from a classification tree?
?
A regression tree handles continuous labels $y \in \mathbb{R}$. Same induction algorithm but the impurity is squared loss $L(S) = \tfrac{1}{|S|} \sum (y - \bar{y}_S)^2$ where $\bar{y}_S$ is the mean label in $S$. Each leaf predicts $\bar{y}_S$.

What does CART stand for, and what does it use as impurity?
?
**Classification and Regression Trees** (Breiman et al., 1984). Uses **Gini** impurity for classification and **squared loss** for regression. Always builds binary trees.

How do you compute training error from an unpruned decision tree? (mock §3 final part)
?
Run every training example through the tree, predict from its leaf (majority class at the leaf), count misclassifications, divide by total. For a tree fit greedily on consistent data with no pruning, this is often $0$ — the tree has memorized the training set.

What is gradient-boosted trees in the mock §1h sense?
?
A linear combination of decision stumps (or shallow trees), each fit to the residuals of the previous one. The aggregated decision boundary can be far more complex than any single stump — making the answer to "is gradient boosting a linear combo of stumps?" **true** structurally, while still being expressive.

What is the difference between Gini impurity and the Gini coefficient?
?
**Gini impurity** is a label-distribution measure used in decision trees: $G = 1 - \sum p_c^2$. **Gini coefficient** is an inequality measure used in economics. Same name, completely different formula and use. Don't confuse them.

Why does CART use Gini instead of entropy?
?
Computational cost. Entropy needs $\log_2$ at every candidate split during training; Gini uses only multiplications and subtractions. The two pick the same root attribute on >95% of standard datasets, so the speedup is essentially free.

What's the right framing for the §1a question "pruning makes the tree more accurate on training data"?
?
**False.** Pruning makes the tree *less* accurate on training data — that's what pruning is. It improves *generalization* (validation/test accuracy) by reducing variance.

How would you compute information gain on a candidate split — give the steps.
?
(1) Compute parent entropy $H(Y)$. (2) Partition data by the attribute's values, getting subsets $S_v$. (3) For each $S_v$, compute child entropy $H(Y \mid X = v)$. (4) Weight: $H(Y \mid X) = \sum_v (|S_v|/|S|) H(Y \mid X = v)$. (5) IG = $H(Y) - H(Y \mid X)$.
