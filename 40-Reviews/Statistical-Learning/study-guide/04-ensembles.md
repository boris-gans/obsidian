---
type: study-guide-cluster
course: Statistical-Learning
cluster: "04-ensembles"
theme: "Ensembles — bagging, boosting, AdaBoost"
prerequisites:
  - 02-classical-supervised
  - 03-theory-glue
covers-concepts:
  - bagging
  - bootstrap-sampling
  - random-forest
  - boosting
  - gradient-boosting
  - weak-learner
  - adaboost
covers-lectures:
  - lecture-12-bagging
  - lecture-13-boosting
  - lecture-14-adaboost
exam-weight: high
---

# Cluster 4: Ensembles — bagging, boosting, AdaBoost

> **The story of this cluster in one sentence.** *Bias–variance gave us the lens; ensembles are the practical machinery for moving along it* — bagging averages many overfitting trees to kill **variance**, boosting additively combines many underfitting stumps to kill **bias**, and AdaBoost is the closed-form, exponential-loss instance you'll actually run by hand on §5.

## Why this cluster exists

Cluster 3 closed with the slogan *"bagging reduces variance, boosting reduces bias"* — but it didn't yet build either algorithm. Cluster 4 fills the gap. Bias–variance gave us the **lens** (the L11 decomposition); ensembles are the **practical machinery** for moving along that curve. The lens dictates the algorithm: if you want to attack the variance term, average many high-variance / low-bias predictors together (parallel ensemble — bagging); if you want to attack the bias term, additively combine many high-bias / low-variance predictors (sequential ensemble — boosting). Trees from Cluster 2 reappear as the canonical base learner in *both* directions: deep trees for bagging (because they overfit beautifully), depth-1 stumps for boosting (because they underfit beautifully). The cluster culminates in AdaBoost — the §5 mock exam's full-3-round-by-hand workout.

**Prerequisites you should feel solid on:**

- [[bias-variance-decomposition]] — the formal lens. Bagging shrinks the **variance** term; boosting shrinks the **bias²** term. Without this, the cluster is a list of algorithms with no spine.
- [[decision-tree]] + [[entropy]] / [[gini-impurity]] / [[information-gain]] — the base learner for both directions, including what a depth-1 **stump** is.
- [[expected-predictor]] — $\bar h(x) = \mathbb{E}_D[h_D(x)]$, the abstract object bagging *empirically* approximates.
- [[exponential-loss]] — the loss whose forward-stagewise additive instance turns out to be AdaBoost.
- [[regularization]] — boosting's iteration count $T$ is itself a regularizer (mock §1g).

## The arc

Three lectures, three points on the same conceptual axis. Bagging targets variance; boosting targets bias generically; AdaBoost is one specific boosting algorithm with a uniquely tractable closed form.

### 1. [[bootstrap-sampling]] — the resampling trick

Given a single dataset $D$ of $N$ points, **bootstrap** sampling draws $N$ new points from $D$ **with replacement**. Each bootstrap sample $D_j$ has $N$ entries, but with repeats — about $63\%$ of the original points appear at least once, the rest are duplicates of those that did. This is standard statistical resampling: it simulates having multiple draws from $P$ without actually collecting more data. Bootstrap is the engine of bagging in this cluster, but the same trick reappears in every "average over many resampled estimates" pattern in statistics.

### 2. [[bagging]] — bootstrap + aggregate

**B**ootstrap **Ag**gregating. Given dataset $D$, repeat: draw a bootstrap sample $D_j$, train a predictor $h_j = A(D_j)$, do this $m$ times, then aggregate $\bar h(x) = \tfrac{1}{m}\sum_j h_j(x)$ (mean for regression; majority vote / probability average for classification). Each $h_j$ is trained independently — embarrassingly parallel. The variance term in the L11 decomposition shrinks as $m$ grows because $\bar h$ approximates the $\bar h(x) = \mathbb{E}_D[h_D(x)]$ that L11 named only abstractly. Bagging *empirically computes* the expected predictor. Bias is unchanged — averaging biased predictors gives a biased average — but variance can drop dramatically when the base learner is high-variance.

### 3. [[random-forest]] — bagging with a twist

Apply bagging to fully-grown decision trees with one extra trick: at each split, **randomly subsample the candidate features** (typically $\sqrt{d}$ of them). The feature randomization decorrelates the individual trees — without it, all trees would tend to make the same first few splits on the dominant features, and their errors would be correlated, blunting the variance reduction. Random Forests are the workhorse "default classifier" for tabular data — strong, robust, almost no tuning required. The L12 formulation says it cleanly: *"CART trees through bagging plus random feature subsets."*

### 4. [[weak-learner]] — the > 50% accuracy primitive

Boosting's input. A **weak learner** is a hypothesis class whose best member achieves error rate strictly less than 50% on a binary task — *any* edge over random guessing is enough. The framework is famously forgiving: the L13 geometric proof shows that any non-orthogonal weak learner, with a small enough step, **must** decrease distance to $\vec y$ in label space. Decision **stumps** (depth-1 trees) are the canonical choice — easy to enumerate (sweep one threshold per feature), almost always achieve $\epsilon < 0.5$, maximally biased / minimally variant, exactly what boosting wants to combine.

### 5. [[boosting]] — gradient descent in function space

Where bagging is parallel, boosting is **sequential**. The L13 framing — *"boosting is gradient descent in function space; our function is $h(x)$"* — is the unifying view. Switch coordinate systems from feature space $\mathbb{R}^d$ to **label space** $\mathbb{R}^N$ (one axis per training point). The target is $\vec y$. Start at $\vec H = \vec 0$. Repeatedly fit a weak learner whose direction has acute angle with the residual $\vec y - \vec H$, take a small step, update. The final ensemble $H(x) = \sum_t \alpha_t h_t(x)$ is the weighted sum of all the directions traversed. Each weak learner is fit by *minimizing the linearized loss along its direction* — a Taylor expansion of the loss around the current $H$ shows that the next $h_t$ should align with the **negative gradient** of the loss in label space. The framework is loss-agnostic; the choice of $\ell$ specializes the algorithm.

### 6. [[gradient-boosting]] — boosting under squared loss

L13's worked example. Plug **squared loss** $\ell(H) = \tfrac{1}{2}\sum_i (H(x_i) - y_i)^2$ into the framework: the per-point partial derivative is the **residual** $H(x_i) - y_i$, so the negative gradient is $y_i - H(x_i)$. The algebra collapses to *"each new weak learner is fit by squared-loss regression on the current residuals."* Practically: train tree $h_t$ to predict $y_i - H_{t-1}(x_i)$, take a step $\alpha$ in that direction, repeat. Step size $\alpha$ is a **shrinkage hyperparameter** — smaller $\alpha$ generalizes better but takes more rounds. This is the basis of XGBoost and LightGBM in modern practice.

### 7. [[adaboost]] — boosting under exponential loss, with closed-form line search

L14's main event. Plug **[[exponential-loss]]** $\ell(H) = \sum_i e^{-y_i H(x_i)}$ into the framework. Three things become unique to this loss. First, the per-point gradient $-y_i e^{-y_i H(x_i)}$ defines a natural **per-point weight** $w_i \propto e^{-y_i H(x_i)}$ — exponentially large for points the current $H$ gets wrong, small for those it gets right. So *each new weak learner is chosen by minimizing the **weighted** error rate* $\epsilon_t = \sum_{i: h_t(x_i) \ne y_i} w_i^t$, not by fitting residuals. Second, the line-search step size has a **closed form** — differentiate $\ell(H + \alpha h)$ w.r.t. $\alpha$ and the answer is

$$\alpha_t = \tfrac{1}{2} \ln \frac{1 - \epsilon_t}{\epsilon_t}.$$

Sanity check: $\epsilon_t = 0.5 \Rightarrow \alpha_t = 0$ (worthless learner gets no vote); $\epsilon_t \to 0 \Rightarrow \alpha_t \to \infty$ (perfect learner dominates). Third, the weight-update rule in normalized closed form is unforgettable: correct points get $w_i / [2(1-\epsilon_t)]$; wrong points get $w_i / [2\epsilon_t]$. The final classifier is $H(x) = \mathrm{sign}(\sum_t \alpha_t h_t(x))$, and the initial weights are uniform $w_i^1 = 1/N$.

### 8. The 1/2 invariant — your sanity check on §5

After every weight update, the **sum of weights on correctly-classified points = sum of weights on misclassified points = 1/2**. Each round's reweighting *bisects* total mass between "what I got right" and "what I got wrong." This is the mock §5 sanity check: at the end of every round of your by-hand AdaBoost run, sum the correct weights and the incorrect weights — both should be $1/2$. If they aren't, you've made an arithmetic error and need to redo the round. The course-specific tie-break: when two stumps tie on weighted error, **prefer the lower-numbered stump** (a course convention from the SLP mock; carry it onto the final).

## Connections worth seeing

- **The "1/2 invariant" is what makes AdaBoost runnable by hand.** The closed-form weight update divides each weight by either $2(1-\epsilon)$ or $2\epsilon$ — the factors of 2 are not cosmetic; they're the constants that *force* the renormalization. After multiplying, you can immediately verify both groups sum to $1/2$ without even computing $Z$. On the mock §5 you should do this check after every round; if the invariant fails, your $\epsilon$ or your stump assignment is wrong.
- **Hinge loss (L09) and exponential loss (L14) are both 0/1-loss surrogates with very different gradient profiles.** Hinge is zero past the margin → sparse gradients → only support vectors matter → SVM concentration on the boundary. Exponential is unbounded → every misclassified point's gradient explodes → AdaBoost concentrates on the *worst-classified* points round-by-round. The concentration mechanism is reversed: SVM concentrates by *not penalizing* the well-classified; AdaBoost concentrates by *over-penalizing* the poorly-classified.
- **Boosting iterations $T$ are a regularizer (mock §1g).** Stop at small $T$ → underfit; very large $T$ → overfit; sweet spot in between with the same U-shaped validation curve as Cluster 3's $\lambda$ and Cluster 1's $k$. The early-stopping intuition transfers directly. This is also the §1l answer for "boosted *trees* grown fully?" — for stumps the answer is "of course, they're already as small as possible"; for deeper boosted trees the question becomes about per-round depth, not iteration count, and can flip.
- **AdaBoost and gradient boosting are siblings, not parent and child.** Both descend from L13's general "gradient descent in function space" framework. Gradient boosting is the squared-loss instance (fit residuals); AdaBoost is the exponential-loss instance (reweight points). Neither contains the other. LogitBoost is the logistic-loss instance, which fixes AdaBoost's noise-sensitivity by bounding per-point influence.
- **Dropout (Cluster 1, L06) is bagging done implicitly inside one network.** Cluster 1 told you dropout trains an "exponential ensemble of weight-shared subnetworks" and averages over them at test time. That's bagging — many models, average their predictions — but the models share weights instead of being trained independently on bootstrap samples. The variance-reduction story is the same; the implementation is wildly different.
- **Mock §1h ("gradient-boosted = linear combo of stumps") is true for both AdaBoost *and* gradient boosting with stumps.** Both produce ensembles of the form $\mathrm{sign}(\sum_t \alpha_t h_t(x))$ — a linear combination whose decision boundary can be far more complex than any individual stump because the *sum* of axis-aligned splits can approximate arbitrary boundaries.

## Common confusions

- **Bagging vs. boosting** — bagging is **parallel** (each $h_j$ trained independently on its own bootstrap sample, then averaged); boosting is **sequential** (each $h_t$ trained to fix what the running ensemble is currently getting wrong). Bagging targets *variance*; boosting targets *bias*. Different base learner: deep trees for bagging, stumps for boosting.
- **AdaBoost vs. gradient boosting** — siblings, not parent-child. Both descend from the L13 framework. AdaBoost uses **exponential loss** + **per-point reweighting** + **closed-form** $\alpha_t$. Gradient boosting (L13's worked example) uses **squared loss** + **fit-the-residuals** + a fixed shrinkage $\alpha$.
- **Stumps vs. fully-grown trees** — stumps for boosting (you want maximally biased weak learners); fully-grown for bagging (you want maximally varying high-variance learners). Mock §1l plays on this distinction.
- **Random Forest vs. Bagged Trees** — Random Forests add **per-split random feature subsampling** on top of bagged trees. Without that, all trees tend to use the same dominant features at the top and become correlated, hurting the variance-reduction.
- **Weighted error vs. unweighted error in AdaBoost** — every $\epsilon_t$ is computed using the **current** weights $w_i^t$, not the original uniform $1/N$. After round 1 the weights are no longer uniform; if you forget that on round 2, you'll pick the wrong stump.
- **The tie-break rule** — when two stumps yield equal weighted error, prefer the **lowest-numbered** stump. Course convention; carry it to the final.
- **Initial weights are $1/N$** (uniform), not $1$. This matters: $\sum_i w_i^1 = 1$, and the 1/2 invariant relies on the weights summing to 1 throughout.

## Self-check (synthesis, not recall)

1. **(blueprint, §5)** From a fixed list of three numbered stumps, run AdaBoost for one round on a 4-point dataset of your choosing. Compute $\epsilon_1$, $\alpha_1$, the updated weights using the closed-form rule, then verify the 1/2 invariant. If two stumps tie, which do you pick?
2. **(blueprint, §1g)** "Boosting iterations $T$ act as a regularizer like $\lambda$." True or false, and what's the U-shape this implies for validation error vs. $T$? Connect to early stopping.
3. **(blueprint, §1l)** "Boosted trees are typically grown fully without pruning." Under what condition is this true, and under what condition can it flip? (Hint: stumps vs. deep boosted trees.)
4. **(synthesis, back to Cluster 3)** Bagging shrinks the *variance* term of the L11 decomposition because $\bar h(x) = \tfrac{1}{m}\sum_j h_j(x)$ converges to $\mathbb{E}_D[h_D(x)]$ as $m \to \infty$. Why doesn't bagging shrink the *bias* term? Phrase your answer as a one-sentence proof from the definitions.
5. **(synthesis)** Hinge loss (L09) and exponential loss (L14) are both convex surrogates for 0/1 loss but produce *opposite* concentration patterns — SVMs end up depending on a few support vectors near the boundary, AdaBoost ends up reweighting toward the worst-classified points. Explain why, in one or two sentences each, by reasoning about the *shape* of each loss.
6. **(synthesis, forward to Cluster 5)** AdaBoost combines many simple linear-in-features (axis-aligned) decisions into a complex non-linear boundary. Kernel SVM (Cluster 5) takes a *single* SVM and lets it become non-linear by working in a higher-dimensional implicit feature space. Are these two strategies orthogonal, or do they overlap? Could you AdaBoost a kernel SVM?

## If you have 10 minutes

The minimum viable review for this cluster:

1. The **"AdaBoost algorithm" verbatim block** in [[adaboost]] (or [[lecture-14-adaboost]]'s "memorize verbatim" section) — initialize $w_i^1 = 1/N$, find min-$\epsilon$ stump (tie-break low number), $\alpha_t = \tfrac{1}{2}\ln\tfrac{1-\epsilon}{\epsilon}$, weight update closed form, $H(x) = \mathrm{sign}(\sum_t \alpha_t h_t(x))$
2. The **1/2 invariant** in [[adaboost]] — your live sanity check during §5
3. The **bias–variance slogan** at the end of [[bias-variance-decomposition]] (revisit it with this cluster's algorithms in mind): bagging targets variance via averaging, boosting targets bias via additive combination

## Next cluster

→ [[05-kernels]] — Boosting linearly combines many *weak* learners into one complex boundary. Cluster 5 does something orthogonal: it enriches a *single* learner's feature space. The kernel trick lets you do max-margin classification in a higher-dimensional (sometimes infinite-dimensional) feature space **without ever computing the features explicitly** — you just need to know the inner product $K(x, x')$, and the SVM dual handles the rest. Mock §6 (quadratic-kernel SVM with slack) is the cross-cluster culmination of L09 + L15.
