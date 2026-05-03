---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: dtrees_slides_SLP.pdf
created: 2026-05-03
---

# Gini impurity

A label-distribution impurity measure used in decision trees as an alternative to [[entropy]]. Conceptually: the **probability of misclassifying a randomly drawn example if we labeled it by sampling from the empirical class distribution**. Cheaper to compute than entropy (no logarithm) and almost always picks the same splits.

> **Note:** "Gini impurity" $\ne$ "Gini coefficient" (the inequality measure used in economics). They share the name but are different quantities.

## Definition

For class probabilities $p_1, \dots, p_C$ at a node:

$$
G(Y) = 1 - \sum_{c=1}^{C} p_c^2 = \sum_{c=1}^{C} p_c (1 - p_c).
$$

The two forms are equivalent ($\sum p_c = 1$).

**Range:**

| Distribution | $G$ |
| --- | --- |
| Pure node ($p_c = 1$ for one class) | $0$ |
| Uniform binary ($p = 0.5$) | $0.5$ |
| Uniform $C$-way | $1 - 1/C$ |

In the binary case $G = 2p(1 - p)$, peaks at $0.5$ when $p = 0.5$ — the same shape as binary entropy, just scaled differently.

## Gini of a tree (split criterion)

For a binary split partitioning $S$ into $S_L$ and $S_R$:

$$
G^T(S) = \frac{|S_L|}{|S|} \, G(S_L) + \frac{|S_R|}{|S|} \, G(S_R).
$$

The split criterion: **pick the attribute and threshold that minimize $G^T(S)$.** Equivalent to the [[information-gain]] argmax with entropy — different scale, same outcome (in most cases).

## Why Gini, not entropy, in CART

Computational cost. Entropy needs $\log_2$ at every candidate split during training, which dominates split-search time. Gini uses only multiplications and subtractions. For a tree with $n$ examples and $d$ features, the speedup compounds at every internal node. CART defaults to Gini; ID3/C4.5 default to entropy.

In practice, the two pick the same root attribute on >95% of standard datasets — and where they disagree, neither tree is reliably better than the other. The lecture flagged them as **interchangeable** on the mock exam.

## Intuition: "probability of misclassification"

Pick a random example from the node and label it by drawing from the empirical class distribution. Probability of misclassification:

$$
P(\text{wrong}) = \sum_c p_c \cdot P(\text{label} \ne c) = \sum_c p_c (1 - p_c) = G(Y).
$$

This makes Gini directly interpretable as a Bayes-error proxy at the node.

## Worked computation (mock-§3 style)

A node with 5 positive, 3 negative ($p = 5/8$):

$$
G(Y) = 1 - \left(\tfrac{5}{8}\right)^2 - \left(\tfrac{3}{8}\right)^2 = 1 - \tfrac{25}{64} - \tfrac{9}{64} = \tfrac{30}{64} = 0.469.
$$

Splitting into $S_L$ (3 positive, 0 negative — pure) and $S_R$ (2 positive, 3 negative):

- $G(S_L) = 0$.
- $G(S_R) = 1 - (2/5)^2 - (3/5)^2 = 12/25 = 0.48$.

Weighted child Gini:

$$
G^T(S) = \tfrac{3}{8} \cdot 0 + \tfrac{5}{8} \cdot 0.48 = 0.30.
$$

The split reduces Gini from $0.469$ to $0.30$ — a good split. (Equivalent to a positive Gini "gain" of $0.169$.)

## Exam-relevant facts

- Formula: $G(Y) = 1 - \sum_c p_c^2$. **Memorize.**
- Binary peak: $G = 0.5$ at $p = 0.5$ (compare entropy peak: $H = 1$).
- Pure node: $G = 0$.
- Tree split: minimize the weighted average $G^T(S) = \sum_v (|S_v|/|S|) G(S_v)$.
- "Gini impurity" $\ne$ "Gini coefficient" — different concept.
- Either entropy or Gini earns full credit on the mock.

## Related

- [[entropy]] — the more theoretically motivated alternative.
- [[information-gain]] — the entropy-based version of the same split criterion.
- [[decision-tree]] — where Gini is used (CART default).
- [[lecture-08-decision-trees|SLP L08]] — source.
