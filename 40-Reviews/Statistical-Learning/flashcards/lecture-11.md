---
tags: [flashcards, Statistical-Learning]
---

# Lecture 11 — Generalization and the bias–variance tradeoff

What is the difference between **training error** and **generalization error**?
?
Training error is computed on the dataset used to fit the model. Generalization error is $\mathbb{E}_{(x,y)\sim P}[\mathcal{L}(h(x), y)]$ — the expected loss on a fresh sample from the population. Training is the proxy; generalization is the actual goal.

What two sources of randomness underlie the bias-variance decomposition?
?
1) The training set $D \sim P^n$ is random — so the trained predictor $h_D = A(D)$ is random. 2) The label $y \mid x$ may itself be non-deterministic — same $x$, different $y$.

What is the **expected predictor** $\bar{h}(x)$?
?
$\bar{h}(x) = \mathbb{E}_{D \sim P^n}[h_D(x)]$ — the average prediction the algorithm makes at $x$ when re-trained on many independently drawn datasets. Estimable in practice by averaging the predictions of many bootstrap-trained models.

What is the **expected label** $\bar{y}(x)$?
?
$\bar{y}(x) = \mathbb{E}[y \mid x]$ — the conditional mean of $y$ given $x$. Different from a single observed label because $y$ may be non-deterministic given $x$ (same house features, different prices).

Write the bias-variance decomposition for squared loss.
?
$\mathbb{E}_{x,y,D}[(h_D(x) - y)^2] = \mathbb{E}_{x,D}[(h_D(x) - \bar{h}(x))^2] + \mathbb{E}_x[(\bar{h}(x) - \bar{y}(x))^2] + \mathbb{E}_{x,y}[(\bar{y}(x) - y)^2]$ = Variance + Bias² + Noise.

Define the **variance** term in the bias-variance decomposition.
?
$\text{Var} = \mathbb{E}_{x,D}[(h_D(x) - \bar{h}(x))^2]$ — how much the trained predictor wobbles around its average $\bar{h}$ when retrained on a different dataset $D$.

Define the **bias²** term in the bias-variance decomposition.
?
$\text{Bias}^2 = \mathbb{E}_x[(\bar{h}(x) - \bar{y}(x))^2]$ — the squared difference between the *average* predictor and the truth-in-expectation. It's what's left after averaging out variance: even with infinite training data, would the algorithm class match the conditional mean of $y$?

Define the **noise** term in the bias-variance decomposition.
?
$\text{Noise} = \mathbb{E}_{x,y}[(\bar{y}(x) - y)^2]$ — the irreducible variation of $y$ given $x$. Set by the data itself, not the model. Cannot be reduced without changing features or relabeling.

Which term in the bias-variance decomposition is **irreducible**?
?
**Noise.** Bias and variance are both reducible by changing the model or training procedure. Noise depends only on the data.

How does **more training data** affect each of bias, variance, and noise?
?
Variance ↓ (predictor estimate gets less noisy). Bias ≈ unchanged (set by the model class). Noise = unchanged (set by the data). Asymptotic test error → bias² + noise.

How does **a more complex model** affect bias and variance?
?
Bias ↓ (richer class can match $\bar{y}$ more closely). Variance ↑ (more parameters fit noise in any one dataset). Total test error has a U-shape in complexity.

What controls model complexity for **k-NN**?
?
The neighbor count $k$. **Small $k$ = complex** (overfits to nearest noise); **large $k$ = simple** (smooths out structure). $k = 1$ is the most complex, $k = N$ the simplest.

What controls model complexity for **decision trees**?
?
Tree depth and number of leaves (and minimum samples per leaf, plus pruning aggressiveness). Deeper / more leaves = more complex.

What controls model complexity for **linear SVM**?
?
The slack penalty $C$. Large $C$ = strict / complex / narrow margin. Small $C$ = loose / simple / wide margin. $C$ behaves like the inverse of $\lambda$.

What controls model complexity for **MLPs**?
?
Depth, width, and total number of parameters (and to a lesser extent training duration before early stopping).

State the three properties of learning curves (error vs $N$).
?
1) Validation error ≥ training error. 2) Validation error decreases monotonically as $N$ grows. 3) Training error never decreases with $N$ — it stays flat or rises.

Why does **training error never decrease** with more training data?
?
Training error is a minimum over a hypothesis class — adding a new datum can only force this minimum higher (if the new point isn't fittable) or leave it equal. A fixed-capacity model can't perfectly fit progressively more diverse points.

Why does **validation error decrease** with more training data?
?
More data shrinks the variance of the trained predictor $h_D$. As $D$ grows, $h_D$ converges to $\bar{h}$, and the variance term in the bias-variance decomposition shrinks toward zero.

How do you diagnose **high variance** from a learning curve?
?
**Wide gap** between training error (well below the acceptable error $\varepsilon$) and validation error (well above $\varepsilon$). Fix: more data, simpler model, more regularization, fewer features, or ensembling (bagging).

How do you diagnose **high bias** from a learning curve?
?
**Both curves are above $\varepsilon$** with a small gap. Training error itself is high — the model can't even fit the training set. Fix: more complex model, more / richer features, less regularization, train longer.

What does it mean when both training and validation error remain above $\varepsilon$ even after bias and variance fixes?
?
The data is **noise-dominated** — too much intrinsic randomness in $y$ given $x$. Fix by cleaning labels or collecting more discriminative features.

What is the asymptotic floor of validation error as $N \to \infty$?
?
**Bias² + noise.** The variance term shrinks to zero with infinite data, but bias and noise are independent of $N$.

How does **regularization** ($\lambda$) affect bias and variance?
?
More regularization (larger $\lambda$) → simpler effective model → **lower variance, higher bias**. Less regularization → richer effective model → higher variance, lower bias. The L10 U-curve in $\lambda$ is the bias² + variance sum traced out.

What does the slogan "bagging reduces variance, boosting reduces bias" mean in bias-variance terms?
?
Bagging averages many bootstrap-trained predictors, approximating $\bar{h}$ — this drives the variance term toward zero while leaving bias unchanged. Boosting combines many high-bias / low-variance weak learners (e.g., stumps) and reduces bias by composition.
