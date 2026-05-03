---
tags: [flashcards, Statistical-Learning]
---

# Lecture 16 — Kernels II (Kernel machines)

What is the **2-step kernel-machines recipe** for kernelizing an algorithm?
?
**Step 0:** prove $w = \sum_i \alpha_i x_i$ (solution lies in span of training points).
**Step 1:** rewrite training and prediction so $x_i$ only appears inside inner products $\langle x_i, x_j \rangle$.
**Step 2:** define a kernel $k$ and substitute $k(x_i, x_j)$ for $\langle x_i, x_j \rangle$.

Name the three popular kernels and write their formulas.
?
- **Linear**: $k(x, z) = x^\top z$.
- **Polynomial degree $d$**: $k(x, z) = (1 + x^\top z)^d$.
- **RBF / Gaussian**: $k(x, z) = \exp(-\|x - z\|^2 / \sigma^2)$.

What's distinctive about the Gaussian / RBF kernel's feature space?
?
**Infinite-dimensional** — cannot be materialized. Provably a "universal approximator": with appropriate $\sigma$, can make almost any finite labeled dataset linearly separable in feature space (provided no two identical points have different labels).

How does the RBF bandwidth $\sigma$ trade off bias and variance?
?
- **Small $\sigma$**: narrow Gaussians, only nearby points matter → flexible, jagged boundary, **overfit risk**.
- **Large $\sigma$**: wide Gaussians, distant points still influence each other → smoother boundary, **underfit risk**.
Cross-validate $\sigma$ alongside the SVM $C$.

What is **Mercer's condition** for a function $k$ to be a valid kernel?
?
The **Gram matrix** $K_{ij} = k(x_i, x_j)$ must be **symmetric and positive semi-definite (PSD)** on every finite set of inputs. Equivalent characterizations: (1) all eigenvalues of $K \ge 0$, (2) $K = P^\top P$ for some real $P$, (3) $\mathbf{x}^\top K \mathbf{x} \ge 0$ for all $\mathbf{x}$.

Why does PSD validity matter?
?
PSD $\Leftrightarrow$ $\exists\, \phi$ such that $k(x, z) = \langle \phi(x), \phi(z) \rangle$. Without it, the implicit feature space doesn't exist, the optimization problem may not be convex, and kernel methods become unstable.

List the 8 well-defined-kernel construction rules.
?
With $k_1, k_2$ already well-defined, $c \ge 0$, $g$ a polynomial with non-negative coefficients, $f$ any function, $A \succeq 0$:
1. $k(x, z) = x^\top z$
2. $k(x, z) = c \cdot k_1(x, z)$
3. $k(x, z) = k_1(x, z) + k_2(x, z)$
4. $k(x, z) = g(k_1(x, z))$
5. $k(x, z) = k_1(x, z) \cdot k_2(x, z)$
6. $k(x, z) = f(x)\, k_1(x, z)\, f(z)$
7. $k(x, z) = e^{k_1(x, z)}$
8. $k(x, z) = x^\top A x$

Prove the polynomial kernel $k(x, z) = (1 + x^\top z)^d$ is well-defined.
?
$x^\top z$ is well-defined by rule 1. $1 + x^\top z$ by rule 3 (constant 1 + the linear kernel). $(1 + x^\top z)^d$ by rule 4 (polynomial composition with non-negative coefficients). ✓

Why is the set kernel $K(S_1, S_2) = e^{|S_1 \cap S_2|}$ well-defined?
?
Represent each set as a binary indicator vector $\vec{x}_i$. Then $|S_1 \cap S_2| = \vec{x}_1^\top \vec{x}_2$ is well-defined by rule 1. Exponentiating via rule 7 gives a valid kernel. ✓

What's the **kernel-induced distance** formula?
?
$\|\phi(x) - \phi(z)\|^2 = k(x, x) + k(z, z) - 2\, k(x, z)$. Used to kernelize distance-based algorithms (k-NN, k-means).

Give an example of a kernel for non-vector data.
?
**String / DNA k-mer kernel**: $k(s_1, s_2) = \sum_{u \in \Sigma^k} \mathrm{count}_u(s_1) \cdot \mathrm{count}_u(s_2)$. Implicit feature space has dimension $|\Sigma|^k$ (e.g., $4^{10} \approx 10^6$ for DNA 10-mers), but computable in $O(\text{length})$ via hash map.

Give a graph kernel example.
?
**Random walk kernel**: count matching random walks (of various lengths) between two graphs. Implicit feature space is infinite (all possible walks). Computed by sampling many random walks from each graph.

What's the speedup the kernel matrix gives at training time?
?
Precompute $K_{ij} = k(x_i, x_j)$ once into an $n \times n$ matrix. Every gradient step then just touches matrix entries — no recomputation, no high-dim feature evaluation. Cost: $O(n^2)$ memory; payoff: training is independent of feature dimension $D$ (which can be infinite).

Which algorithms are kernelizable, which aren't?
?
**Kernelizable** (depend on data only via inner products / distances): SVM, ridge regression, perceptron, k-NN, k-means, PCA. **Not kernelizable**: decision trees, random forests, naive Bayes — they look at individual feature values, not inner products.

Why is the linear kernel sometimes useful even though it's "trivial"?
?
Equivalent to using a linear classifier — but if $d$ (feature dimension) is huge, **storing and updating the kernel matrix can be faster** than working with raw features. Especially useful for high-dim sparse data (BoW text).

What's the typical default kernel for tabular non-text data?
?
**RBF / Gaussian.** Most flexible, only one hyperparameter ($\sigma$), translation-invariant, near-universal. Cross-validate $\sigma$ with the median pairwise distance as a starting heuristic.

Is the **sigmoid kernel** $\tanh(a\, x^\top z + b)$ always a valid kernel?
?
**No.** It's only Mercer-PSD for some hyperparameter values $(a, b)$. Despite being widely used (it imitates a single-hidden-layer NN), it's not always a valid kernel — modern kernel libraries warn about this.

Where do kernel methods stop scaling well?
?
Memory cost is $O(n^2)$ for the Gram matrix, training $O(n^2)$ to $O(n^3)$. **Don't scale past ~10–100k samples.** Modern alternatives for large $n$: linear models with feature crossing, gradient-boosted trees, or neural networks.

What's the kernelized SVM prediction formula?
?
$h(x_{\text{new}}) = \mathrm{sign}\!\big(\sum_i \alpha_i y_i\, k(x_i, x_{\text{new}}) + b\big)$. Only training points with $\alpha_i > 0$ (the support vectors) contribute.

Why does kernelizing structured objects (strings, graphs, sets) win over hand-crafted feature vectors?
?
- Avoids extreme-dimensional feature spaces.
- Avoids loss of structural information from forced flattening.
- Avoids enumerating features that may be impossible to enumerate explicitly.
Kernels compute similarity directly on the structure.
