---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 09 - Naive Bayes-1.pdf
created: 2026-05-02
---

# Bayes' formula

The foundational identity of probabilistic reasoning under uncertainty. Relates the **posterior probability** of a hypothesis given evidence to the **likelihood** of the evidence under the hypothesis times the **prior probability** of the hypothesis.

$$\boxed{\,P(A|B) = \frac{P(B|A)\,P(A)}{P(B)}\,}$$

## Components

| Term | Name | NLP meaning (for class $y$ given document $x$) |
|---|---|---|
| $P(A|B)$ | **Posterior** | $P(\text{class} | \text{document})$ — what we want |
| $P(B|A)$ | **Likelihood** | $P(\text{document} | \text{class})$ — how the document looks under each class |
| $P(A)$ | **Prior** | $P(\text{class})$ — base rate of the class |
| $P(B)$ | **Evidence / marginal** | normalization constant; same across classes |

## In NLP

Bayes' formula is the engine behind [[naive-bayes|Naïve Bayes]] classification:
$$P(y|x) = P(y) \cdot \frac{P(x|y)}{P(x)}$$

Since $P(x)$ is constant across classes (we're picking the best $y$ for fixed $x$), we drop it and rank classes by $P(y) P(x|y)$ — the unnormalized posterior. With the [[naive-bayes|naïve independence assumption]], $P(x|y) = \prod_i P(x_i|y)$ and the classification rule is

$$\hat{y} = \arg\max_y P(y) \prod_i P(x_i|y) \quad\text{(formula sheet form)}$$

## Why it's useful

The formula **measures the impact of non-independence** between two events. If $A$ and $B$ are independent, then $P(A|B) = P(A)$ — observing $B$ tells you nothing about $A$. Bayes' formula is what lets us update beliefs: we start with a prior, observe evidence, and rescale the posterior by the likelihood ratio.

This is the same mathematics that drives [[hmm-viterbi|HMM Viterbi]] (priors + transitions + emissions = posterior over hidden state sequence) and many other probabilistic NLP models.
