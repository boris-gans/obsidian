---
type: study-guide-cluster
course: Statistical-Learning
cluster: "01-neural-foundation"
theme: "Neural nets foundation"
prerequisites: []
covers-concepts:
  - k-nearest-neighbors
  - minkowski-distance
  - voronoi-diagram
  - curse-of-dimensionality
  - overfitting-underfitting
  - hyperparameter
  - train-validation-test-split
  - cross-validation
  - feature-normalization
  - perceptron
  - linear-classifier
  - sigmoid
  - logistic-regression
  - cross-entropy
  - logistic-loss
  - activation-function
  - softmax
  - gradient-descent
  - stochastic-gradient-descent
  - linear-regression
  - mean-squared-error
  - multilayer-perceptron
  - relu
  - hidden-layer
  - universal-approximation-theorem
  - backpropagation
  - computational-graph
  - chain-rule
  - softmax-cross-entropy-gradient
  - vanishing-exploding-gradients
  - weight-initialization
  - dropout
  - learning-rate-schedule
  - early-stopping
covers-lectures:
  - lecture-01-knn
  - lecture-02-linear-classifiers-perceptron
  - lecture-03-intro-neural-nets
  - lecture-04-mlps
  - lecture-05-backprop
  - lecture-06-improving-mlps
  - lecture-07-training-deep-nets
exam-weight: high
---

# Cluster 1: Neural nets foundation

> **The story of this cluster in one sentence.** *From a single neuron to a trainable deep network* — seven lectures that take the perceptron's three failure modes and, one fix at a time, end up at a deep MLP you can actually train end-to-end.

## Why this cluster exists

The course has no prior cluster — this is the **starting point**. Phase A is unusual in that it doesn't bridge from anything; it begins by parking k-NN as a memory baseline (no training, no parameters) so that everything that follows can be framed by contrast: *parametric* models with a *trainable* weight vector. The phase is a single arc — perceptron → logistic regression → multi-class softmax → MLP → backprop → "all the things that go wrong when you actually try to run gradient descent on a deep net." By the end you have a model class (deep MLP) and the full toolkit needed to train it, which Phase B will then deliberately set aside in favour of classical machinery.

**Prerequisites you should feel solid on:**

- *None.* Assumes basic linear algebra (dot products, matrix-vector products), basic calculus (gradients, chain rule), and probability (Bernoulli, log-likelihood). Everything else is built up from scratch.

## The arc

A walkthrough of how the concepts in this cluster build on each other. The shape is the perceptron's **three issues** (binary only, doesn't terminate unless linearly separable, returns *any* separating hyperplane) being patched, one fix at a time, until depth becomes possible.

### 1. [[k-nearest-neighbors]] — the no-training baseline

L01 starts not with a trainable model but with a non-parametric one: predict by majority vote over the $k$ closest training points. There is **no learning phase** — the "model" is just the training set itself. This sets the bar everything else has to clear: you can already classify *something* without any optimization, so why bother with parameters? The answer comes immediately in the form of [[curse-of-dimensionality|the curse of dimensionality]] (in high $d$, all distances look the same and "closest" stops being informative), [[hyperparameter|hyperparameter sensitivity]] ($k$ controls overfit vs underfit but is *set*, not learned), and [[feature-normalization|scale sensitivity]] (k-NN's distance is meaningless if features live on different scales). Mock §1d ("1-NN training error is 0") and §2c ("which classifiers can fit XOR") both depend on knowing exactly what k-NN does and doesn't do.

### 2. [[minkowski-distance]], [[voronoi-diagram]], [[overfitting-underfitting]] — the supporting cast

The L01 vocabulary that you'll keep using for the rest of the course. [[minkowski-distance|Minkowski distance]] $(\sum_r |x_r - z_r|^p)^{1/p}$ subsumes Manhattan ($p=1$) and Euclidean ($p=2$) — same family, different geometry. [[voronoi-diagram|Voronoi diagrams]] visualize 1-NN's decision regions: each cell is the locus of points closest to one training example. And [[overfitting-underfitting]] gets named here for the first time — small $k$ overfits (every training point is its own region), large $k$ underfits (one giant blurry region) — the **complexity dial** that every later cluster will name with a different parameter.

### 3. [[train-validation-test-split]] + [[cross-validation]] — the protocol for *measuring* generalization

If $k$ is a knob you have to set, you need a principled way to set it. L01 introduces the **three-way split**: train fits the model, validation chooses the hyperparameter, test gives one unbiased generalization estimate. The §1 traps in the mock are full of "1-NN training error is 0" (true — every point is its own nearest neighbour on the training set) and similar gotchas that hinge on knowing which of the three splits a given quantity lives on. [[cross-validation|Cross-validation]] is the variance-reducing refinement: rotate the validation fold, average across folds.

### 4. [[perceptron]] + [[linear-classifier]] — the parametric model and its three problems

L02 reintroduces the [[perceptron]] from prior coursework — $y = \mathrm{sign}(w^\top x + b)$, the canonical [[linear-classifier]] — and immediately diagnoses **the perceptron's three issues** (the framing the lecture explicitly names): (i) **binary only** — one output neuron, $\pm 1$, no multi-class extension; (ii) **doesn't terminate** unless the data is linearly separable; (iii) **returns *any* arbitrary separating hyperplane** — even a clearly-bad one running an inch from the closest points. The rest of Phase A is the answer to those three issues, fix by fix. L03 will solve (i) with softmax. L02 will solve (ii) and (iii) by replacing the hard sign with a smooth probability and defining a *loss* whose minimizer is uniquely the "good line." (Vapnik's full geometric answer to issue (iii) — the widest street, the maximum-margin hyperplane — has to wait until [[support-vector-machine|L09]] in Cluster 2.)

### 5. [[sigmoid]] + [[logistic-regression]] + [[cross-entropy]] + [[logistic-loss]] — fix issues (ii) and (iii)

L02's fix: pass the dot product $z = w^\top x + b$ through a smooth squashing function, the [[sigmoid]] $\sigma(z) = 1/(1+e^{-z})$. Now the output is a *probability* in $(0,1)$, not a discrete sign — points far from the boundary get confident outputs, points near the boundary hover around $0.5$. The model — [[logistic-regression]] — is the same linear score followed by sigmoid. The matching loss is [[cross-entropy|cross-entropy]] (the negative log-likelihood for a Bernoulli, written with $y \in \{0,1\}$) or equivalently [[logistic-loss]] (the same thing written with $y \in \{-1,+1\}$ as $\log(1 + e^{-y(w^\top x)})$). Now the perceptron's "any line will do" indeterminacy is gone: we'll pick the line that *minimizes the loss*. L02 only sketches the loss curve — it doesn't yet say how to minimize it.

### 6. [[softmax]] + [[linear-regression]] + [[mean-squared-error]] + [[gradient-descent]] + [[stochastic-gradient-descent]] — fix issue (i), and learn how to *minimize*

L03 is two moves at once. First, fix issue (i) — multi-class — by replacing sigmoid with [[softmax]] (one score per class, normalized to a probability distribution); the matching loss is cross-entropy on one-hot labels. Second, introduce the **how**: minimize $\mathcal{L}(\theta)$ by stepping in the direction of $-\nabla\mathcal{L}$ — [[gradient-descent]] — and its noisy-but-cheap version [[stochastic-gradient-descent]] that updates after each example (or each mini-batch) instead of after the full epoch. Mock §1k is exactly this trap: SGD updates per-example in random order, *not* once per epoch on the full gradient. L03 also rounds out the loss family with [[linear-regression]] and [[mean-squared-error]] (same neuron, no activation, real-valued targets), so by lecture's end the student has three matched (model, loss) pairs: linear-regression+MSE, logistic-regression+cross-entropy, softmax-classifier+cross-entropy.

### 7. [[multilayer-perceptron]] + [[hidden-layer]] + [[activation-function]] + [[universal-approximation-theorem]] + [[relu]] — go deep

L04 stacks neurons: a [[multilayer-perceptron]] is a sequence of (linear → non-linear) layers. The non-linearity in [[hidden-layer|hidden layers]] is essential — without it, depth collapses (a stack of linear maps is still a linear map). The [[universal-approximation-theorem]] says a single hidden layer with enough units can approximate any continuous function arbitrarily well, but in practice depth wins (more efficient, more compositional). [[activation-function]] becomes a design choice: sigmoid (the L02/L05 baseline), $\tanh$, and the modern default [[relu]] $= \max(0, z)$ — cheap, non-saturating on the positive side, sparse activations. The output layer is still a linear classifier on the *learned* features.

### 8. [[backpropagation]] + [[chain-rule]] + [[computational-graph]] — train the deep network

L05 fills in the *how* for deep MLPs. [[backpropagation]] is "chain rule on a computational graph" — sweep forward to compute the loss, then sweep backward propagating $\partial \mathcal{L} / \partial \theta$ through every node by local-derivative multiplication. The data structure backprop operates on is the [[computational-graph]] (nodes = operations, edges = data dependencies); the calculus identity it rests on is the [[chain-rule]]. Once backprop is in hand, you can train *any* feed-forward graph by composing differentiable pieces and running SGD on the gradients backprop produces.

### 9. [[softmax-cross-entropy-gradient]] + [[vanishing-exploding-gradients]] + [[weight-initialization]] + [[dropout]] — the things that go wrong, and their patches

L06 is "everything that breaks when you naively train a deep MLP." [[softmax-cross-entropy-gradient]] is why softmax + cross-entropy is the right pairing at the output: $\partial \mathcal{L}/\partial z_k = a_k - y_k$, a clean per-logit residual that doesn't suffer from the saturation problem MSE+sigmoid does. [[vanishing-exploding-gradients]] is the central pathology — sigmoid's derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$ peaks at $0.25$, so chained gradients shrink geometrically with depth (and ReLU's "dead unit" failure mode is the other side of the same coin). The patches: switch hidden activations to ReLU (no positive-side saturation); pick [[weight-initialization]] so initial pre-activations are well-scaled (Xavier for $\tanh$/sigmoid; He for ReLU); add [[dropout]] as a regularizer that empirically reduces variance by training an exponentially large ensemble of weight-shared subnetworks.

### 10. [[learning-rate-schedule]] + [[early-stopping]] + [[hyperparameter]] tuning — the training-loop knobs

L07 tightens the loop. The learning rate $\eta$ is the single most important hyperparameter; you tune it on a **log scale** because its effect is multiplicative (random search beats grid search precisely because hyperparameter importance is uneven). [[learning-rate-schedule]] decays $\eta$ over time — step / cosine / exponential / linear / $1/\sqrt{t}$ / linear-warmup. [[early-stopping]] watches the validation loss and stops training when it plateaus or rises, which is itself an implicit regularizer (it caps the effective complexity of the trained network without modifying the loss). By the end of L07 you have a fully-trainable deep network and a recipe for tuning it.

## Connections worth seeing

- **The perceptron's three issues are the cluster's syllabus.** L02 names them; L02–L05 fix them in order; L06–L07 patch the things that break when those fixes meet a real deep network. If you can recite the three issues and which lecture fixes which one, you can recite the cluster.
- **Sigmoid + cross-entropy and softmax + cross-entropy are the same idea in different dimensions.** The Bernoulli case (sigmoid) is the binary specialization of the categorical case (softmax). Both pair with cross-entropy because cross-entropy *is* the negative log-likelihood — the loss is forced by the model's probabilistic interpretation, not chosen separately.
- **Dropout is bagging in disguise.** Cluster 4 will introduce [[bagging]] as "train many models on bootstrap samples, average their predictions." Dropout does the same thing implicitly inside one network: each forward pass uses a random subnetwork, and test-time averaging over all-units-on is approximately the ensemble mean. The L06 framing as "exponential ensemble of weight-shared subnetworks" is the explicit version of this connection.
- **The $C$ of L09 and the $\lambda$ of L10 are the same dial as the $k$ of L01 and the $M$ of early-stopping.** Each is a complexity knob with the same U-shaped validation curve. Naming them as one dial across the course means you only have to learn the *shape* of the trade-off once.

## Common confusions

- **Perceptron vs. logistic regression** — same linear score $w^\top x + b$, but perceptron emits $\mathrm{sign}$ (hard, no confidence, no smooth loss), LR emits $\sigma$ (soft, calibrated probability, cross-entropy loss). LR is the perceptron with sigmoid + log-likelihood.
- **Sigmoid vs. softmax** — sigmoid is one number to one probability (binary); softmax is a vector of $K$ scores to a $K$-way probability distribution (multi-class). Sigmoid is the $K=2$ case of softmax.
- **Gradient descent vs. SGD** — GD updates after computing the gradient over the *full* training set (one update per epoch); SGD updates after **each example** (or mini-batch), in random order. The mock §1k trap targets exactly this distinction.
- **MSE + sigmoid vs. cross-entropy + sigmoid** — MSE on a sigmoid output saturates: when the prediction is confidently wrong, $\sigma'$ is tiny and the gradient vanishes. Cross-entropy cancels the $\sigma'$ in the chain rule and gives the clean $a - y$ residual ([[softmax-cross-entropy-gradient]]). This is *why* CE is the right loss for classification, not just convention.
- **ReLU dying vs. sigmoid saturating** — both kill gradients, but ReLU dies on the *negative* side only ($z \le 0 \Rightarrow$ output $= 0$, derivative $= 0$, unit can't escape) and is binary; sigmoid saturates on *both* extremes and is a smooth squashing.
- **Universal approximation vs. depth helps** — UAT says one hidden layer is *theoretically* enough; depth wins in *practice* because deep networks compose features hierarchically (mock §1b answer: "the model learns **hierarchical** representations") and use parameters more efficiently.

## Self-check (synthesis, not recall)

1. **(blueprint, §1k)** SGD updates parameters in random order — but per *example*, not per *epoch*. Why is the per-example version better at escaping shallow local minima than full-batch GD? (Hint: it's the same mechanism that makes dropout work.)
2. **(blueprint, §1d)** Why is 1-NN training error always exactly $0$, and what does that tell you about how to evaluate k-NN's generalization? (Connect to why we need a held-out test set, not just training error.)
3. **(synthesis)** Map each of the perceptron's three issues to the lecture (L02–L05) where it's resolved, and name the *single concept* that resolves it. If a friend can only memorize three concepts from Phase A, which three?
4. **(synthesis)** [[dropout]] (L06) and [[bagging]] (which you haven't seen yet — preview from L12) both reduce variance by averaging over many models. What's the architectural difference, and why does dropout *also* reduce co-adaptation between units in a way bagging can't?
5. **(blueprint, §1j)** "The loss surface depends on the training data" — true or false? Reason from the definition of the loss as a sum over training examples; what would have to be true for this to be false?
6. **(synthesis, forward to Cluster 2)** Phase A built deep MLPs that can fit basically anything, but the mock §3 tests **decision trees**, not networks. What does a tree give you that an MLP doesn't, and why is it worth a whole new modality?

## If you have 10 minutes

The minimum viable review for this cluster:

1. The "three issues with the perceptron" diagram in [[lecture-02-linear-classifiers-perceptron]] — memorize the three issues and which subsequent lecture fixes each
2. [[softmax-cross-entropy-gradient]] — read the derivation; it's the single equation that justifies why classification networks pair softmax with cross-entropy
3. [[vanishing-exploding-gradients]] + [[relu]] together — understand the failure mode and why ReLU + good init is the standard fix; mock §1j and the L06 dead-unit story both ride on this

## Next cluster

→ [[02-classical-supervised]] — Phase A built one model class (the deep MLP) and the toolkit to train it. Phase B *deliberately* sets that toolkit aside and turns to **classical supervised learning** — decision trees and linear SVMs — two very different ways of solving the same classification problem with very different machinery. The reason isn't that MLPs are bad; it's that the next phase needs the geometric and algorithmic vocabulary of classical methods (stumps, support vectors, margins) to set up Clusters 3–5.
