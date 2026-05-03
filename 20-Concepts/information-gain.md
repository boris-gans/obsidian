---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: dtrees_slides_SLP.pdf
created: 2026-05-03
---

# Information gain

The **reduction in [[entropy]]** when the data is split on an attribute — the criterion that ID3 / C4.5 maximize at each node when growing a [[decision-tree]]. The CART analogue with Gini works the same way (minimize weighted child Gini).

## Definition

For a candidate split on attribute $X$ at a node with data $S$:

$$
\mathrm{IG}(X) = H(Y) - H(Y \mid X),
$$

where $H(Y)$ is the parent entropy and $H(Y \mid X) = \sum_v \frac{|S_v|}{|S|} H(Y \mid X = v)$ is the weighted child entropy.

**Always non-negative** (Jensen's inequality on $-x \log x$). $\mathrm{IG}(X) = 0$ iff $X$ and $Y$ are independent.

## The greedy algorithm

At each node:

1. For each unused attribute $X_i$, compute $\mathrm{IG}(X_i)$ on the data at that node.
2. Pick $X^* = \arg\max_i \mathrm{IG}(X_i)$.
3. Split the data by the values of $X^*$ and recurse on each child.

The tree is built top-down, greedily — never reconsidering earlier splits. **Locally optimal, globally suboptimal** in general.

## Worked example (mock-§3 flow)

14 examples at the root: 9 positive, 5 negative. $H(Y) \approx 0.940$ bits.

Three candidate attributes:

| Attribute | $H(Y \mid X)$ | IG |
| --- | --- | --- |
| Outlook (3 values) | 0.694 | 0.246 |
| Temperature (3 values) | 0.911 | 0.029 |
| Humidity (2 values) | 0.788 | 0.152 |
| Wind (2 values) | 0.892 | 0.048 |

Pick **Outlook** (highest IG). Split, recurse, repeat.

This is the kind of computation §3 of the mock exam asks you to do **by hand** ([[exam-blueprint#Recurring patterns|blueprint §3]]).

## Bias toward many-valued attributes

IG has a structural problem: an attribute with very many distinct values (in the limit, one distinct value per training example) gives $H(Y \mid X) = 0$ — every child is a single example, hence pure. **IG is maximized**, but the resulting tree is useless (it memorizes the training set in one split).

Two fixes:

**Gain ratio (C4.5).** Normalize IG by the entropy of the partition itself (a measure of how "spread out" the split is):

$$
\mathrm{GainRatio}(X) = \frac{\mathrm{IG}(X)}{-\sum_v \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}}.
$$

The denominator penalizes high-cardinality attributes. C4.5 uses this; ID3 doesn't.

**Cap the cardinality.** Some implementations refuse to split on attributes with too many values, or convert continuous attributes into a single threshold-based binary split (CART's approach).

## Exam-relevant facts

- $\mathrm{IG}(X) = H(Y) - H(Y \mid X)$. **Memorize.**
- Always $\ge 0$; equals $0$ iff $X \perp Y$.
- Greedy ID3/C4.5 picks the attribute with **largest IG** at each node.
- High-cardinality bias: IG favors many-valued attributes spuriously; gain ratio is the fix.
- Equivalent for CART/Gini: pick the split with the **smallest weighted child Gini**.

## Related

- [[entropy]] — the impurity measure IG is computed from.
- [[gini-impurity]] — the Gini-based analogue (minimize weighted child Gini).
- [[decision-tree]] — where IG is the splitting criterion.
- [[lecture-08-decision-trees|SLP L08]] — source.
