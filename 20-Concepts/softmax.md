---
tags: [concept]
courses: [NLP, Statistical-Learning]
sources:
  - course: NLP
    file: Session 16 - Neural Networks-1.pdf
  - course: Statistical-Learning
    file: SLP-lec3(1).pdf
created: 2026-05-02
---

# Softmax

The **softmax** function is the standard way to turn a vector of real-valued scores into a **probability distribution** over $K$ classes. It generalizes the [[sigmoid]] (which handles binary) to multi-class settings.

The blueprint flags it as **medium weight** for MCQs: Quiz IV Q6.M-A and Q6.M-B test softmax computation on a small score vector. The formula is on the formula sheet.

## Formula (formula sheet)

$$\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$

For an input vector $\mathbf{z} = (z_1, \ldots, z_K)$, the output $\mathrm{softmax}(\mathbf{z})$ has entries that:
- are **non-negative** (each $e^{z_i} > 0$)
- **sum to 1** ($\sum_i \mathrm{softmax}(z_i) = 1$)

So the result is a valid probability distribution.

## Where softmax appears in the course

- **Output layer of multi-class neural classifiers** — convert logits to class probabilities (Session 16, [[30-Sources/NLP/pdf/Session 16 - Neural Networks-1.pdf#page=13|slide 13]]: probability is a softmax of $z = Wh + b$)
- **[[word2vec|Word2Vec]] Skip-gram conditional** — $P(c \mid w) = \dfrac{\exp(v_c^\top v_w)}{\sum_{w'} \exp(v_{w'}^\top v_w)}$
- **Attention weights** — $\alpha = \mathrm{softmax}(QK^\top / \sqrt{d_k})$, applied **row-wise** so each query distributes 1.0 of attention across keys
- **Generation: temperature sampling** — divide logits by $T$ before softmax: $\mathrm{softmax}(z / T)$ — high $T$ flattens, low $T$ sharpens

## Worked example

For $\mathbf{z} = (1, 2, 3)$:
$$e^1 = 2.718,\quad e^2 = 7.389,\quad e^3 = 20.086$$
$$\sum = 30.193$$
$$\mathrm{softmax}(\mathbf{z}) = (0.090, 0.245, 0.665)$$
The largest input gets the largest probability; small differences in $z$ become amplified by exponentiation.

## Pairing with cross-entropy

Softmax outputs are paired with **[[cross-entropy]] loss** for classification:
$$L = -\sum_{k=1}^{K} y_k \log p_k, \qquad p_k = \mathrm{softmax}(z_k)$$
Together they form the **softmax + cross-entropy** combination used everywhere in NLP classifiers — including the BoW + MLP text classifier of Session 16 and the HuggingFace `AutoModelForSequenceClassification` head.

The gradient of cross-entropy through softmax simplifies cleanly: $\partial L / \partial z = p - y$ ([[30-Sources/NLP/pdf/Session 16 - Neural Networks-1.pdf#page=13|slide 13]]), which is why this pairing is numerically convenient.

## Softmax vs sigmoid

| | Sigmoid | Softmax |
|---|---|---|
| Output dim | scalar | vector of $K$ |
| Use | binary probability | multi-class distribution |
| Sum to 1? | (with $1-\sigma$) | yes, by construction |
| Parameter | one logit $z$ | vector of logits $\mathbf{z}$ |

Sigmoid is the $K=2$ special case of softmax with one logit fixed at 0.

## Numerical pitfall

For large $z_i$, $e^{z_i}$ overflows. The standard trick is:
$$\mathrm{softmax}(z_i) = \frac{e^{z_i - \max_j z_j}}{\sum_k e^{z_k - \max_j z_j}}$$
Subtracting the max gives the same probability distribution but keeps exponents $\le 0$, avoiding overflow. [not in source — practical note]

## Exam framing

| Question | Answer |
|---|---|
| Compute $\mathrm{softmax}(1, 2)$ | $(e^1/(e^1+e^2), e^2/(e^1+e^2)) \approx (0.269, 0.731)$ |
| Why softmax for classification? | Turns logits into a valid probability distribution that sums to 1 (Quiz IV Q6.M-A/B) |
| What's the relationship to sigmoid? | Sigmoid = softmax for binary classification |
| What does softmax do in attention? | Normalizes raw scores $S$ row-wise into attention weights $\alpha = \mathrm{softmax}(S)$ |

## In Statistical Learning (L03): "what happens if we use many neurons?"

L03 of SLP introduces softmax via a concrete pain-point: stack $C$ binary perceptrons in parallel, each scoring its own class, and the outputs *don't add up to 1* — there's no joint constraint forcing a probability distribution ([[30-Sources/Statistical-Learning/pdf/SLP-lec3(1).pdf#page=50|slides ~45–55]]). Softmax is the fix: exponentiate and normalize so the outputs *are* a distribution. SLP's wording: "$a$ stands for activation; output is always positive."

The course makes the equivalence with binary L02 explicit: **two-unit softmax with one logit fixed at 0 is exactly the sigmoid** ([[30-Sources/Statistical-Learning/pdf/SLP-lec3(1).pdf#page=70|slides ~65–75]]). So binary [[logistic-regression]] is the $C = 2$ corner case of the multi-class story. Using a two-unit softmax for binary is *redundant* — fewer parameters via sigmoid suffice.

In the SLP arc, softmax + multi-class [[cross-entropy]] is the canonical output-and-loss pair for every classification model from L03 (single-layer) through L05–L07 (deep MLPs), and it's the same object SVMs (L09) replace with hinge + max-margin.

## Related

- [[sigmoid]] — the binary case
- [[cross-entropy]] — the standard pairing loss
- [[attention]] — softmax over scaled QK^T
- [[word2vec|Word2Vec]] — softmax over inner products in the Skip-gram conditional
