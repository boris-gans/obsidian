---
tags: [flashcards, Statistical-Learning]
course: Statistical-Learning
lecture: 02
source: SLP-lec2(1).pdf
created: 2026-05-03
---

# Lecture 02 — Linear classifiers and the perceptron

What are the three issues with the perceptron, per SLP L02 slide 2?
?
1) Only works for binary classification. 2) Doesn't terminate unless the data is *perfectly* linearly separable. 3) When separable, returns *any* arbitrary separating hyperplane (some are clearly worse than others).

What's the fix L02 proposes for the perceptron's "any arbitrary hyperplane" problem?
?
Replace the hard $\pm 1$ output with a smooth sigmoid that produces a probability, then minimize a loss (cross-entropy / logistic loss) so a unique "best" line emerges.

What's the formula for the sigmoid $\sigma(z)$, and what does it map to?
?
$\sigma(z) = \dfrac{1}{1+e^{-z}}$. Maps $\mathbb{R} \to (0, 1)$. $\sigma(0) = 0.5$.

What does the dot product $w^T x + b$ produce geometrically?
?
A "score" that is $0$ on the decision boundary, positive on the $+$ side (the side $w$ points toward), negative on the $-$ side. Magnitude is proportional to perpendicular distance from the boundary.

In 2-D, write the decision boundary $w_0 + w_1 x_1 + w_2 x_2 = 0$ in $y = mx + b$ form.
?
$x_2 = -\dfrac{w_1}{w_2}\,x_1 - \dfrac{w_0}{w_2}$. Slope $-w_1/w_2$, intercept $-w_0/w_2$.

Write the logistic-regression predictor as a single equation.
?
$\hat{y} = P(y=1 \mid x) = \sigma(w^T x + b)$, with decision rule "predict $+1$ if $\hat{y} > 0.5$".

What single-point loss does L02 derive, and what are the three equivalent forms?
?
Negative log-likelihood / cross-entropy. (A) $\ell_k = -[y \log p + (1-y)\log(1-p)]$ for $y \in \{0,1\}$. (B) $\ell_k = -\log P(y \mid x)$ in terms of the correct class. (C) $\ell_k = \log(1 + e^{-y(w^T x + b)})$ for $y \in \{-1, +1\}$ — the **logistic loss**.

Why does the L02 loss equal the negative log-likelihood?
?
Because under a Bernoulli model, $P(y \mid x) = p^y (1-p)^{1-y}$. Taking $-\log$ of the dataset likelihood and summing over examples gives exactly cross-entropy. So minimizing the loss = maximum likelihood estimation.

What is the *total* training loss in L02?
?
$\mathcal{L}(w, b) = \sum_{i=1}^{N} \ell_i$. Lower is better. Different $w, b$ give different scores, hence different per-point losses, hence different $\mathcal{L}$.

What does the logistic loss $\log(1 + e^{-m})$ tend to as the margin $m = y(w^T x + b) \to +\infty$? As $m \to -\infty$?
?
$m \to +\infty$: $\ell \to 0$ (confidently correct). $m \to -\infty$: $\ell$ grows roughly linearly in $|m|$ (confidently wrong).

What is the value of the logistic loss when $m = 0$ (point on the boundary)?
?
$\log 2 \approx 0.693$ — the prediction is $\hat{p} = 0.5$, maximum uncertainty.

Show the algebraic equivalence between binary cross-entropy ($y \in \{0,1\}$) and logistic loss ($y \in \{-1,+1\}$).
?
Substitute $y_{\pm} = 2y_{01} - 1$. Both forms collapse to $\log(1 + e^{-y_{\pm}\,z})$. Same loss, two label conventions.

Why doesn't the perceptron stop on non-linearly-separable data?
?
The perceptron updates only on misclassified points. If no separating hyperplane exists, mistakes never run out, so updates never stop. Logistic regression replaces the mistake-count with a smooth loss that always has a finite gradient → always has a "best line" to converge to.

What is the *templates* view of a linear classifier?
?
Rows of $W$ are per-class templates; the score for class $c$ is $W_c \cdot x + b_c$ — the dot product of the input with that class's template. The class with the highest score wins.

Can a linear classifier (perceptron, logistic regression, linear SVM) achieve zero training error on XOR?
?
**No.** XOR is not linearly separable. This is the standard mock-§2c trap — only non-linear models (kernel SVM, MLP, decision tree, $k$-NN with small $k$) can.

What is the relationship between the perceptron and logistic regression?
?
Same affine score $z = w^T x + b$. Perceptron applies $\mathrm{sign}(z)$ (hard); logistic regression applies $\sigma(z)$ (smooth probability). Trained differently — perceptron by mistake-driven updates, logistic by minimizing cross-entropy.

What is the *decision rule* for logistic regression once $\sigma(z)$ is computed?
?
Predict the positive class if $\sigma(z) > 0.5$, equivalently if $z = w^T x + b > 0$. The decision boundary itself is identical to the perceptron's; only the confidence representation changes.

What is the geometric interpretation of the weight vector $w$ relative to the decision boundary?
?
$w$ is **perpendicular** to the decision boundary $\{x : w^T x + b = 0\}$. It points from the boundary toward the $+$ side.

Why is "negative log-likelihood" called "cross-entropy" in this context?
?
Because $-\sum_i y_i \log \hat{y}_i$ is the cross-entropy $H(p, q)$ of the predicted distribution $q$ from the actual one $p$ (a degenerate one-hot at the true label). Bayes' likelihood derivation and Shannon's information-theoretic derivation give the same expression.

What does L02 *not* tell us how to do, and which lectures fix that?
?
L02 only *defines* the loss $\mathcal{L}(w, b)$. It does not tell us how to *minimize* it (no closed-form solution like OLS). Gradient descent / SGD comes in L03–L05 (training neural nets, backprop).

In the cross-entropy form $\ell_k = -[y \log p + (1-y)\log(1-p)]$, why does only one term contribute for any given example?
?
$y$ is binary $\{0, 1\}$, so one of $y$ or $(1-y)$ is zero — only the *correct* class's log-probability appears. Compactly: $\ell_k = -\log P(y^{(k)} \mid x^{(k)})$.

What does the SLP L02 closing slide hint about lecture 03?
?
Multi-class extension: rows of a matrix $W$ become per-class templates; multi-class scores are dot products of $x$ with each row, then softmax gives a probability distribution over classes.

Logistic loss vs. hinge loss — what behaviour distinguishes them for correctly-classified points with large margin?
?
Logistic: nonzero (but exponentially small) loss → gradient never zero, every example always contributes a tiny nudge to $w$. Hinge: exactly zero loss for $m \geq 1$ → only "support vectors" influence $w$ at all.
