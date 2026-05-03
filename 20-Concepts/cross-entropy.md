---
tags: [concept]
courses: [NLP, Statistical-Learning]
sources:
  - course: NLP
    file: Session 07 - Information Retrieval.pdf
  - course: NLP
    file: Session 10 - Logistic Regression-1.pdf
  - course: Statistical-Learning
    file: SLP-lec2(1).pdf
  - course: Statistical-Learning
    file: SLP-lec3(1).pdf
created: 2026-05-02
---

# Cross-entropy

A measure of how unexpected observed data are under a probabilistic model. **On the formula sheet.**

## Definition

$$H(p, q) = -\sum_i y_i \log \hat{y}_i$$

Where $y_i$ is the observed (true) distribution and $\hat{y}_i$ the predicted distribution. **Binary case** (also on sheet):
$$H_{\text{bin}} = -\big[y \log p + (1-y)\log(1-p)\big]$$

## Intuition: surprise

Cross-entropy measures **surprise**: when the model assigns *low* probability to observed words, surprise is *high* (high cross-entropy). When the model assigns *high* probability to observed words, surprise is *low* (low cross-entropy). Good models reduce surprise on real data.

This is the same intuition behind [[tf-idf|TF-IDF]]: rare-but-observed terms are statistically surprising and therefore informative; common terms are predictable and carry little information. TF-IDF "anticipates" cross-entropy by rewarding statistically surprising words.

## Why it's the standard training objective

Training [[logistic-regression]] (and most NLP classifiers) **minimizes cross-entropy with observed labels** — equivalently, maximum likelihood (Quiz II Q9, II.M2 Q9):

> **MLE corresponds to minimizing negative log-likelihood (cross-entropy) on observed labels.**

For language modelling, minimizing cross-entropy between predicted and observed token distributions encourages the model to assign high probability to actual next-words.

## Worked exam shapes

**Q: probability $p(y=1) = p(y=0) = 0.5$, what is the entropy (natural log)?**
$$H = -\sum_i p_i \log p_i = -2 \cdot 0.5 \log 0.5 = \log 2$$
(Quiz II Q2 answer: $\log 2$.)

**Q: classifier assigns probability 1 to one label. Entropy?**
$0$ — certainty implies zero entropy (Quiz II.M2 Q16).

**Q: probabilities near 0.5 indicate?**
**High entropy** — maximum uncertainty in binary classification (Quiz II.M2 Q19).

## Bridging to next-word prediction

When framed probabilistically, **next-word prediction = choose the word that minimizes expected surprise given context**. Each level of statistical NLP — TF-IDF weighting, IR ranking, n-gram language models, neural language models — is a refinement of the same underlying principle: **modelling language by controlling surprise**.

## In Statistical Learning (L02): the loss derivation

[[lecture-02-linear-classifiers-perceptron|SLP L02]] derives cross-entropy as the answer to "what is the loss whose curve is high when the prediction is confidently wrong and zero when it's confidently right?" The chain is:

- **Likelihood (Bayes view):** the probability of observing $y^{(k)}$ given parameters is $p^{y^{(k)}}(1-p)^{1-y^{(k)}}$ for a Bernoulli; product over the dataset gives the likelihood $L(w, b)$.
- **Negative log:** $\ell_k = -\log P(y^{(k)} \mid x^{(k)})$ — large when $P$ is small (the prediction was confidently wrong).
- **Cross-entropy (Shannon view):** the same expression equals $-\sum_y p_y \log \hat{p}_y$ between the actual and predicted distributions.
- **Log-loss / logistic-loss (SLP convention):** rewriting with $y \in \{-1, +1\}$ gives $\ell_k = \log(1 + e^{-y_k (w^T x_k + b)})$. See [[logistic-loss]].

Minimizing cross-entropy = maximizing likelihood = the standard objective for [[logistic-regression]] (and every neural-network classifier with a softmax/sigmoid output head).

## Multi-class form (SLP L03)

Once L03 introduces [[softmax]] for multi-class problems, the loss generalizes from the binary form to:

$$
\ell_i = -\log a_{y_i} \;=\; -\log \frac{e^{z_{y_i}}}{\sum_{c'=1}^{C} e^{z_{c'}}},
$$

where $y_i$ is the *index* of the true class and $a = \mathrm{softmax}(z)$ ([[30-Sources/Statistical-Learning/pdf/SLP-lec3(1).pdf#page=60|slides ~60–70]]). Only the correct-class probability appears explicitly — the lecture's question "why don't we need to worry about the wrong classes' probabilities?" has the answer: the softmax constraint $\sum_c a_c = 1$ implicitly couples them. Pushing $a_{y_i} \to 1$ pushes the others toward 0.

This is the form pasted into every neural-net training loop you'll write: scores → softmax → cross-entropy. Binary case (L02) is the $C = 2$ degenerate version; the multi-class case here is the general one.
