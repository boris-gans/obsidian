---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: dtrees_slides_SLP.pdf
  - course: Statistical-Learning
    file: example-dtree.pdf
  - course: Statistical-Learning
    file: SLP-Bagging.pdf
  - course: Statistical-Learning
    file: SLP-Boosting.pdf
created: 2026-05-03
---

# Decision tree

A non-parametric supervised classifier (or regressor): a tree where each **internal node** tests one attribute, each **branch** corresponds to a value of that attribute, and each **leaf** predicts the class label (or class probability). Inference traverses root → leaf following the tests; training builds the tree top-down by greedily picking the best splitting attribute at each node.

## Structure

| Component | Role |
| --- | --- |
| Internal node | Test one discrete-valued attribute $X_i$ |
| Branch from node | Selects one value for $X_i$ |
| Leaf | Predicts $Y$ (or $P(Y \mid X \in \text{leaf})$) |

A path root → leaf is a **conjunction** of attribute tests; the whole tree is a **disjunction** of root-leaf paths. Decision trees can represent any Boolean function of categorical attributes, given enough depth.

## Top-down induction (ID3 / C4.5 / CART)

Quinlan-style greedy recursive induction ([[30-Sources/Statistical-Learning/pdf/dtrees_slides_SLP.pdf#page=8|slide ~8]]):

```
node = Root
loop:
  1. A ← argmax over attributes of IG(A)   (or min Gini)
  2. Assign A as the decision attribute for node
  3. For each value of A, create a new descendant of node
  4. Sort training examples to leaf nodes
  5. If training examples are perfectly classified, STOP
     Else iterate over new leaf nodes
```

The "best" attribute is whichever **maximizes [[information-gain]]** (or equivalently minimizes weighted child impurity). Greedy: doesn't reconsider earlier splits.

**Algorithm variants:**
- **ID3** (Quinlan, 1986) — discrete attributes only, entropy-based IG.
- **C4.5** (Quinlan, 1993) — continuous attributes via threshold search; uses **gain ratio** instead of raw IG to fix the high-cardinality bias; handles missing values with fractional instances.
- **CART** (Breiman et al., 1984) — always binary splits; uses Gini for classification and squared loss for regression.

## Splitting criteria — entropy or Gini

**Entropy-based.** [[information-gain|IG]] = $H(Y) - H(Y \mid X)$ = parent entropy minus weighted child entropy. Pick the attribute with the largest IG.

**Gini-based.** Pick the split with the smallest weighted child Gini:

$$
G^T(S) = \frac{|S_L|}{|S|} G(S_L) + \frac{|S_R|}{|S|} G(S_R).
$$

The prof flagged "Entropy or Gini interchangeably" — either earns full credit on the mock. In practice they almost always pick the same root attribute. Gini is cheaper (no log).

## Stopping criteria

- All training examples at the node have the same label (pure node — IG = 0 for any further split).
- No attributes left to split on.
- **Pre-pruning** thresholds: max depth, min samples per leaf, min IG to continue.

## Pruning to combat overfitting

A fully grown tree perfectly memorizes the training set — in the worst case, one leaf per training example. **High variance.** Two pruning strategies:

- **Pre-pruning** — stop growing when one of the criteria above fires.
- **Post-pruning** (recommended) — grow the full tree, then collapse subtrees whose removal doesn't hurt validation error. Cost-complexity pruning is the standard recipe.

Pruning trades a worse training fit for a better validation/test fit — the tree-version of bias-variance.

## Handling missing attributes

Standard problem: at training time or test time, an example has a missing value for the attribute being tested.

| Strategy | How it works | When |
| --- | --- | --- |
| **Surrogate splits** (CART) | At each node, store backup attributes whose splits agree with the primary on most training examples. Use the surrogate when the primary is missing. | Default in CART |
| **Fractional instances** (C4.5) | Send a fractional copy of the example down each branch, weighted by the branch's training-data proportion. Predictions are weighted votes. | Default in C4.5 |
| **Imputation** | Replace missing values with the column mean / mode before training. | Generic preprocessing |
| **Ignore the example** | Drop incomplete training examples. | Bad — wastes data |

Mock §1i tests this: the right answer is "surrogate splits" (or "fractional instances"), **not** "ignore the example."

## Regression trees (CART)

Same algorithm, different impurity. With continuous labels $y \in \mathbb{R}$, the split criterion is squared loss:

$$
L(S) = \frac{1}{|S|} \sum_{(x, y) \in S} (y - \bar{y}_S)^2, \qquad \bar{y}_S = \frac{1}{|S|}\sum y.
$$

Each leaf predicts $\bar{y}_S$. A regression tree is a **piecewise-constant function** over the input space — step approximation of the underlying regression function. Best-split-finding is $O(n \log n)$ per attribute (sort, then sweep).

## Strengths and weaknesses

**Strengths:**
- Handle both discrete and continuous features natively.
- Output is highly interpretable (a sequence of human-readable tests).
- No feature scaling needed.
- Fast at test time: $O(\text{depth})$ per example.

**Weaknesses:**
- High variance — tiny data perturbations produce different trees.
- Greedy top-down induction can miss the globally optimal tree.
- Overfit aggressively without pruning.
- Weak alone — typical accuracy is mediocre.

The lecture's framing: *"CART are very lightweight classifiers; very fast during testing; usually not competitive in accuracy but can become very strong through bagging (Random Forests) and boosting (Gradient Boosted Trees)."*

## Why trees power ensembles (L12–L14)

Trees' weakness — high variance, mediocre alone — is exactly what makes them ideal **ensemble building blocks**:

- **Bagging / Random Forests** ([[lecture-12-bagging|L12]]) — average many **fully-grown, unpruned** trees trained on bootstrap samples. Each tree is high-variance / low-bias; aggregation kills variance while preserving low bias. Random Forest adds the "split using $k = \sqrt{d}$ random features" trick to decorrelate the trees.
- **Boosting / Gradient Boosting** ([[lecture-13-boosting|L13]]) — sequentially fit trees to residuals → bias reduction.
- **AdaBoost** ([[lecture-14-adaboost|L14]]) — uses **stumps** (depth-1 trees) reweighted by training error.

By the time you reach those lectures, the entropy / Gini / IG machinery from this lecture is the working language. Note the inversion: bagging wants trees grown **fully** (high-variance baseline to average out); boosting wants **stumps** (high-bias baseline to combine).

## Exam-relevant facts

- Internal nodes test attributes; branches by value; leaves predict.
- Build top-down by greedy IG (or min-Gini) maximization.
- Either entropy or Gini earns full credit on the mock (§3 algorithmic compute).
- Pruning **reduces** training fit but **improves** generalization (mock §1a).
- Missing-attribute handling: surrogate splits or fractional instances (mock §1i).
- Trees can achieve **zero training error** on any consistent dataset (mock §2c) — at the cost of overfitting.

## Related

- [[entropy]] — the impurity measure for ID3 / C4.5.
- [[gini-impurity]] — the impurity measure for CART.
- [[information-gain]] — the splitting criterion.
- [[lecture-08-decision-trees|SLP L08]] — source.
- [[overfitting-underfitting]] — what pruning prevents.
