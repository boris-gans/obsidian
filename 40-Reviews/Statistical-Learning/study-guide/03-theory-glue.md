---
type: study-guide-cluster
course: Statistical-Learning
cluster: "03-theory-glue"
theme: "Regularization and the bias-variance lens"
prerequisites:
  - 01-neural-foundation
  - 02-classical-supervised
covers-concepts:
  - regularization
  - l1-regularization
  - l2-regularization
  - elastic-net
  - exponential-loss
  - huber-loss
  - bias-variance-decomposition
  - learning-curve
  - generalization-error
  - expected-predictor
covers-lectures:
  - lecture-10-loss-functions-regularization
  - lecture-11-bias-variance
exam-weight: high
---

# Cluster 3: Theory glue — regularization and the bias-variance lens

> **The story of this cluster in one sentence.** *How do we control model complexity?* — first by giving the regularizer family its own treatment (L10), then by giving the **lens** (bias–variance) that explains why regularization works at all (L11), turning every prior cluster's ad-hoc dial into instances of one mechanism.

## Why this cluster exists

Cluster 2 left two ad-hoc complexity knobs on the table — the depth/pruning of a decision tree and the $C$ of soft-margin SVM — and didn't quite admit they were the same dial. Cluster 3 generalizes both. L10 lays out the **loss + regularizer template** $\min_w \frac{1}{N}\sum \ell(h_w(x_i), y_i) + \lambda\,\Omega(w)$, then enumerates the menu — losses (hinge, logistic, exponential, squared, Huber) and regularizers ($L_1$, $L_2$, Elastic Net) — that every supervised learner can be assembled from. L11 then asks the meta-question: *why does any of this work?* The bias–variance decomposition is the answer — three additive, non-negative terms (variance, bias², noise) that together name **everything we've informally called underfitting and overfitting**, and tell us which fix attacks which term. Without L11, regularization is just a heuristic. With L11, every prior cluster's complexity dial is an instance of one mechanism.

**Prerequisites you should feel solid on:**

- [[support-vector-machine]] — its $C$ is the *first concrete regularizer* the course saw; L10 generalizes it.
- [[overfitting-underfitting]] — the qualitative phenomenon L11 will name formally as high-variance / high-bias.
- [[hinge-loss]], [[logistic-loss]], [[mean-squared-error]] — losses re-enumerated and compared in L10.
- [[cross-validation]] — how $\lambda$ is actually chosen in practice.
- The U-shape of validation error vs. $k$ ([[k-nearest-neighbors|kNN]]) and vs. $C$ (SVM) — L11 explains the U-shape with the bias–variance lens.

## The arc

Two lectures, one big move: name the family, then explain the family.

### 1. [[regularization]] — the unifying framing

Every supervised learner can be written as **data-fit + complexity-control**:

$$\min_w\ \tfrac{1}{N}\sum_{i=1}^N \ell(h_w(x_i), y_i)\ +\ \lambda\,\Omega(w).$$

The data-fit term ($\ell$) measures how wrong each prediction is. The regularizer ($\Omega$) measures how *big* or *complex* the parameter vector is. The hyperparameter $\lambda$ is the **trade-off knob** — large $\lambda$ → simple model (underfit risk), small $\lambda$ → complex model (overfit risk). The framing is what unifies the SVM's $C$ (note $C \sim 1/\lambda$ — large $C$ = small $\lambda$ = trust-the-data), trees' depth limit, kNN's $k$, early stopping's iteration cap, and dropout's keep-probability into a single conceptual mechanism. Once you see the template, you stop memorizing different stories for each method.

### 2. [[exponential-loss]] + [[huber-loss]] — completing the loss menu

L10's loss menu rounds out what Clusters 1 & 2 already named. For **classification** (margin $z = y \cdot h(x)$): hinge (SVM, $\max(0, 1-z)$), logistic ($\log(1+e^{-z})$), and the new arrival **[[exponential-loss]]** $e^{-z}$ — extremely aggressive (loss grows exponentially in mis-prediction), nice convergence properties, **brittle on noisy labels** because a single mislabeled point can blow up the loss. Cluster 4 will reveal that AdaBoost is forward-stagewise additive modelling under exactly this loss. For **regression** (residual $z = h(x) - y$): squared loss estimates the mean, absolute loss estimates the median, and **[[huber-loss]]** is the smooth interpolation — quadratic near zero, linear in the tails — *outlier-resistant* without losing differentiability. The mean-vs-median framing is the key intuition: which central tendency does each loss target?

### 3. [[l2-regularization]] — Ridge, the smooth shrinker

$\Omega(w) = \tfrac{1}{2}\|w\|_2^2 = \tfrac{1}{2}\sum_j w_j^2$. **Strongly convex** — every direction has positive curvature, the optimum is unique, optimization is well-behaved. Geometric form: the constraint set $\|w\|_2 \le B$ is a **disk**; loss contours touch it at a generic boundary point with no axis-alignment. The result: $L_2$ **shrinks** all coefficients toward zero proportionally but **doesn't zero them out**. This is the §1e exam trap. The trap statement reads "L2 regularization adds the squared norm of the weights to the loss; it shrinks coefficients toward zero, leading to **sparse models**" — every clause is true *except the last*. L2 doesn't induce sparsity; **L1 does**. Read every clause to the end before answering.

### 4. [[l1-regularization]] — Lasso, the sparsifier

$\Omega(w) = \|w\|_1 = \sum_j |w_j|$. **Convex but not strictly convex** (Hessian zero almost everywhere). Geometric form: the constraint set $\|w\|_1 \le B$ is a **diamond** with corners on the axes; loss contours typically touch the diamond at a **corner**, where some coordinates are exactly zero. That's the geometric origin of sparsity. Useful for feature selection — fit Lasso, read off which $w_j$ ended up nonzero, those are your selected features. Downside: not differentiable at $w_j = 0$; with correlated features, the optimum is a flat ridge (Lasso arbitrarily picks one of the correlated pair) — unstable across resamples.

### 5. [[elastic-net]] — strict convexity + sparsity, the §1f answer

$\Omega(w) = \alpha\|w\|_1 + (1-\alpha)\|w\|_2^2$ with $\alpha \in [0, 1)$. The "have your cake and eat it" choice: the $L_1$ piece induces sparsity (corner solutions), the $L_2$ piece restores **strict convexity** (positive-definite Hessian everywhere). This is the §1f answer the prof tests directly: *why does Elastic Net give strict convexity but L1 alone doesn't?* Because $L_1$'s Hessian is zero a.e. — adding any positive-definite contribution from $L_2$ makes the combined Hessian positive definite throughout. Practically: with correlated features, Elastic Net spreads weight smoothly across them (the $L_2$ part breaks ties) while still zeroing out features unrelated to the target (the $L_1$ part).

### 6. [[generalization-error]] + [[expected-predictor]] — the right names for the right random variables

L11 sets up the bias–variance derivation by giving names to the two sources of randomness. **Generalization error** is the expected test loss on a fresh draw from $P$ — *not* training loss, which the model has explicitly optimized for. **Expected label** $\bar{y}(x) = \mathbb{E}[y \mid x]$ is the conditional mean of $y$ given $x$ — same input, different labels, the data's own irreducible scatter. **Expected predictor** $\bar{h}(x) = \mathbb{E}_D[h_D(x)]$ is the average prediction the *algorithm* would make at $x$ if you re-trained it on many independently drawn datasets — a thought experiment that becomes the [[bagging|bagging]] empirical estimator in L12. With these two averages named, the decomposition is forced.

### 7. [[bias-variance-decomposition]] — three terms, three fixes

Under squared loss:

$$\mathbb{E}_{x,y,D}[(h_D(x) - y)^2]\ =\ \underbrace{\mathbb{E}_{x,D}[(h_D - \bar{h})^2]}_{\text{Variance}}\ +\ \underbrace{\mathbb{E}_x[(\bar{h} - \bar{y})^2]}_{\text{Bias}^2}\ +\ \underbrace{\mathbb{E}_{x,y}[(\bar{y} - y)^2]}_{\text{Noise}}.$$

Each term has its own meaning *and* its own fix:
- **Variance** — how much $h_D$ wobbles when $D$ changes. Fix: more data, simpler model, **ensembles** (bagging — Cluster 4).
- **Bias²** — how off-target $\bar h$ is even with infinite data. Fix: more complex model, richer features, less regularization, **boosting** (Cluster 4).
- **Noise** — irreducible variation in $y$ given $x$. Fix: change the features, clean the labels — *not* the model.

The decomposition is exact under squared loss; the prof states it generalizes (with extra subtlety) to other losses and to classification. It's the lens that explains *why* L10's $\lambda$ knob has a U-shaped validation curve: increasing $\lambda$ trades variance for bias, and the sweet spot is wherever variance + bias² is minimum.

### 8. [[learning-curve]] — diagnosing which term is hurting you

A learning curve plots training error and validation error against $N$ (training-set size). Three regularities — **validation error ≥ training error**, **validation error decreases monotonically in $N$**, **training error never decreases with $N$** — give you a diagnostic. If training error is well below your acceptable threshold but validation is far above (large gap): high variance — get more data, simplify, regularize, ensemble. If training error itself is above the threshold: high bias — more capacity, more features, less regularization. If both stubbornly fail: the data has too much intrinsic noise, change the features or clean the labels. This is the **§2a sketch** on the past mock — the train-rises-with-$N$ / validation-falls-with-$N$ pair, with the gap between them telling you which bucket you're in.

## Connections worth seeing

- **The $C$ of L09, the $\lambda$ of L10, the $k$ of L01, and the $M$ of early-stopping are the same dial.** Each controls a complexity-vs-fit trade-off; each has the same U-shaped validation curve; each is searched on a **log scale** because its effect is multiplicative (L07's hyperparameter-tuning aside). Once you see this you only have to learn the shape of the trade-off once, not five times.
- **The L1 vs L2 geometric story is the *constraint form* of regularization.** $\min \mathcal{L} + \lambda\Omega(w)$ and $\min \mathcal{L}\ \text{s.t.}\ \Omega(w) \le B$ are equivalent (large $\lambda$ ↔ small $B$). The diamond-vs-disk picture isn't decorative — it's what proves $L_1$ is sparsifying and $L_2$ isn't, and it's the visual the §1e trap is designed to catch students who skipped.
- **Squared loss for *regression* and $L_2$ penalty on $w$ are the same operation in different dimensions.** Squared loss on residuals estimates the conditional **mean** of $y$; $L_2$ penalty on weights shrinks the weight vector toward the **mean-zero** point. Same convex curvature, same closed-form ridge solution $w = (XX^\top + \lambda I)^{-1} X y^\top$.
- **Bagging targets variance, boosting targets bias** — and the *reason* is in this cluster, not Cluster 4. Bagging averages high-variance / low-bias learners (deep trees) so the variance term shrinks and the bias term is unchanged. Boosting combines high-bias / low-variance learners (stumps) so the bias term shrinks. The formal slogans are direct readings off the L11 decomposition; Cluster 4 just builds the algorithms.
- **Cross-validation operationalizes the L11 lens.** When you sweep $\lambda$ over a log grid and pick the value with lowest CV error, you are *empirically* finding the bottom of the U — the variance / bias² trade-off point that L11 named.

## Common confusions

- **L1 vs L2 — what each does to weights** ($\!$§1e bait). L2 *shrinks* (smaller magnitudes, all nonzero); L1 *sparsifies* (some exactly zero). If you remember diamond → corners, L1 is sparsity. If you remember disk → no corners, L2 is shrinkage.
- **Convex vs strictly convex** ($\!$§1f bait). $L_1$ is convex (no negative curvature) but **not strictly** convex (zero curvature in some directions). $L_2$ is strictly convex. Elastic Net inherits strict convexity from the $L_2$ piece. Strict convexity = unique minimizer.
- **Bias vs variance** — both are *expectations over training sets*, not single-dataset quantities. Variance asks how much $h_D$ wobbles when $D$ changes; bias asks whether the *average* $\bar h$ matches the truth $\bar y$. A single trained model has *neither* on its own — the decomposition only makes sense as a thought experiment over many $D$.
- **Bias² vs noise** — bias² is the *model's* fault (even with infinite data, the model class can't reach $\bar y$); noise is the *data's* fault (even with the right model class, $y$ varies given $x$). Confusing these means you'll try the wrong fix.
- **Generalization error vs test error vs validation error** — generalization error is the *expectation*; test error is one *unbiased estimate* of it (on data the model has never seen); validation error is the estimate you used to tune hyperparameters (slightly optimistic by construction).
- **Exponential loss vs hinge loss vs logistic loss** — all three are convex surrogates for 0/1 loss, with different growth rates. Exponential grows fastest (so it's most sensitive to outliers, brittle on noisy labels — this is exactly why AdaBoost has trouble with noise). Hinge is zero past the margin (sparse gradients). Logistic decays smoothly (every point contributes).

## Self-check (synthesis, not recall)

1. **(blueprint, §1e)** "L2 regularization adds the squared norm of the weights to the loss; it shrinks coefficients toward zero, leading to sparse models." Identify the false clause and explain *why* L2 doesn't induce sparsity using the disk-vs-diamond geometry.
2. **(blueprint, §1f)** Why is Elastic Net strictly convex when $\alpha < 1$, while pure $L_1$ is not? Answer in one sentence by referencing the Hessian of each component.
3. **(blueprint, §2a)** Sketch the qualitative shape of training error and validation error as functions of $N$. State the three regularities (validation ≥ training, validation ↓ in $N$, training never ↓ in $N$). Now sketch the same two errors as functions of $\lambda$ — what's the U-shape, and which side is overfitting?
4. **(synthesis, back to Cluster 2)** The SVM's $C$ and L10's $\lambda$ control the same trade-off but point in opposite directions. Show that the soft-margin SVM $\min \tfrac{1}{2}\|w\|^2 + C \sum \xi_i$ is an instance of the L10 template with $\lambda = 1/C$. What is $\Omega(w)$, and what is $\ell$?
5. **(synthesis)** [[dropout]] (Cluster 1, L06) and [[bagging]] (preview, Cluster 4) both reduce *variance*. Explain in bias–variance language why a single deep MLP with no dropout has high variance, and what dropout's "exponential ensemble of weight-shared subnetworks" view does to that variance term.
6. **(synthesis, forward to Cluster 4)** L11 ends with the slogan *"bagging reduces variance, boosting reduces bias."* Without yet knowing the algorithms, predict what *kind* of base learner each technique should use — high-variance + low-bias, or high-bias + low-variance — and why averaging vs. additive combination targets different terms.

## If you have 10 minutes

The minimum viable review for this cluster:

1. The **table of pairings** in [[lecture-10-loss-functions-regularization]] (OLS / Ridge / Lasso / Elastic Net) — memorize the convexity column and the §1e/§1f answers
2. The **bias–variance formula** in [[bias-variance-decomposition]] — three terms, what each means, what fix attacks each one. The slogan "bagging reduces variance, boosting reduces bias" is the L12–L14 preview
3. The **§2a learning-curve sketch** in [[learning-curve]] — the train-rises / validation-falls picture and how the gap diagnoses which bucket you're in

## Next cluster

→ [[04-ensembles]] — Bias–variance gave us the **lens**. Cluster 4 builds the **practical machinery** for moving along the curve: bagging (averages many high-variance / low-bias learners → variance ↓), boosting (combines many high-bias / low-variance learners → bias ↓), and AdaBoost (the closed-form, exponential-loss instance of boosting that does the §5 mock by hand). Trees from Cluster 2 reappear as the canonical base learner in both directions — deep trees for bagging, depth-1 stumps for boosting.
