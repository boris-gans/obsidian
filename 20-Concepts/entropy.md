---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: dtrees_slides_SLP.pdf
created: 2026-05-03
---

# Entropy (as label-distribution impurity)

Shannon's information-theoretic measure of the **uncertainty** in a distribution — applied here to the class-label distribution at a decision-tree node. Pure nodes (one class only) have entropy $0$; uniform-class nodes have maximum entropy $\log_2 C$. Decision-tree induction (ID3, C4.5) **minimizes weighted child entropy** at each split via [[information-gain]].

## Definition

For a discrete distribution over $C$ classes with probabilities $p_1, \dots, p_C$:

$$
H(Y) = -\sum_{c=1}^{C} p_c \log_2 p_c.
$$

By convention, $0 \log_2 0 \equiv 0$ (the limit of $p \log p$ as $p \to 0^+$).

**Units.** Bits (because of $\log_2$). Sometimes expressed in nats ($\log_e$) or dits ($\log_{10}$); the choice is a constant rescaling that doesn't change argmax-of-IG decisions.

## Range

| Distribution | Entropy |
| --- | --- |
| One class with $p_c = 1$, others $0$ (pure) | $0$ — no uncertainty |
| Uniform over $C$ classes | $\log_2 C$ — maximum uncertainty |
| Binary, $p = 0.5$ | $1$ bit |
| Binary, $p \in \{0, 1\}$ | $0$ bits |

In the **binary case**:

$$
H(Y) = -p \log_2 p - (1-p) \log_2 (1-p),
$$

which peaks at $H = 1$ when $p = 0.5$ and drops to $0$ at $p \in \{0, 1\}$.

## Conditional entropy on a tree split

For a candidate split on attribute $X$ that partitions data $S$ into subsets $S_v$ (one per attribute value $v$):

$$
H(Y \mid X) = \sum_v \frac{|S_v|}{|S|} \, H(Y \mid X = v).
$$

Each term: probability of going down branch $v$, times the entropy of labels in that branch.

**Bound.** $H(Y \mid X) \le H(Y)$ always — knowing $X$ can only reduce uncertainty about $Y$. Equality iff $X$ and $Y$ are independent.

## Why entropy as impurity

Three intuitive properties:
1. **Zero on pure nodes.** A pure node has no label uncertainty; we shouldn't penalize it.
2. **Concave in $p$.** A weighted average of pure children has lower total impurity than the parent — splitting helps even if neither child is fully pure. Convexity of $-x \log x$ is what makes IG non-negative.
3. **Symmetric across classes.** Doesn't privilege any specific class label.

Information-theoretically, $H(Y)$ is the **expected number of yes/no questions** needed to identify $Y$ from its distribution — connecting to Huffman coding.

## Computational cost vs. Gini

Entropy involves logarithms — slower than [[gini-impurity|Gini]]. In practice the two pick the same root attribute most of the time (their level sets are very similar in the binary case). The lecture notes them as interchangeable on the mock; CART defaults to Gini for speed, ID3/C4.5 use entropy.

## Worked computation (mock-§3 style)

Suppose 14 training examples at the root: 9 positive, 5 negative.

$$
H(Y) = -\tfrac{9}{14} \log_2 \tfrac{9}{14} - \tfrac{5}{14} \log_2 \tfrac{5}{14} \approx 0.940 \text{ bits.}
$$

Now consider splitting on `Wind` with values `{Strong, Weak}`, where:
- `Strong`: 6 examples (3 positive, 3 negative). $H = 1.0$ bit.
- `Weak`: 8 examples (6 positive, 2 negative). $H \approx 0.811$ bits.

Conditional entropy:

$$
H(Y \mid \text{Wind}) = \tfrac{6}{14} \cdot 1.0 + \tfrac{8}{14} \cdot 0.811 \approx 0.892 \text{ bits.}
$$

Information gain:

$$
\mathrm{IG}(\text{Wind}) = 0.940 - 0.892 = 0.048 \text{ bits.}
$$

This is the calculation flow you'll do by hand on §3.

## Exam-relevant facts

- Formula: $H(Y) = -\sum_c p_c \log_2 p_c$. **Memorize.**
- Pure node: $H = 0$. Uniform binary: $H = 1$. Uniform $C$-way: $H = \log_2 C$.
- Conditional entropy: weighted average of child entropies.
- IG = parent entropy − conditional entropy (always $\ge 0$).
- Either entropy or Gini earns full credit on the mock.

## Related

- [[gini-impurity]] — the cheaper alternative.
- [[information-gain]] — entropy reduction; the splitting criterion.
- [[decision-tree]] — where entropy is used as impurity.
- [[cross-entropy]] — closely related but distinct measure (entropy of $p$ relative to $q$).
- [[lecture-08-decision-trees|SLP L08]] — source.
