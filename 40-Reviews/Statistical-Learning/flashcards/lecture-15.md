---
tags: [flashcards, Statistical-Learning]
---

# Lecture 15 — Kernels I (Inner products in higher dimensions)

What problem motivates kernel methods?
?
Linear classifiers underfit on non-linearly-separable data (e.g. concentric circles — high bias). Adding non-linear features makes the data separable, but enumerating all polynomial combinations gives a $2^d$-dim embedding — infeasible to materialize directly.

What's the central insight that lets us avoid storing $\phi(x)$ explicitly?
?
For a linear classifier trained by gradient descent (and most convex methods), the optimal weights are a **linear combination of training points**: $w^* = \sum_i \alpha_i x_i$. So the algorithm only ever needs **inner products** $\langle x_i, x_j \rangle$ between training points — never $w$ explicitly.

Prove $w^* = \sum_i \alpha_i x_i$ for GD on squared loss.
?
Induction on iterations. **Base:** $w_0 = 0 = \sum_i 0 \cdot x_i$. **Step:** if $w_t = \sum_i \alpha_i x_i$, then $w_{t+1} = w_t - \eta g$ where the squared-loss gradient $g = \sum_i \gamma_i x_i$ (with $\gamma_i = 2(w^\top x_i - y_i)$) is itself a linear combo of $x_i$'s. So $w_{t+1} = \sum_i (\alpha_i - \eta\gamma_i) x_i$ — same form, new $\alpha$'s. ☐

What is the "dual algorithm" for GD on squared loss?
?
Track $\alpha_i \in \mathbb{R}^n$ instead of $w \in \mathbb{R}^d$. Precompute the kernel matrix $k_{ij} = \langle x_i, x_j \rangle$ (size $n \times n$). Update: $\gamma_i = \sum_j \alpha_j k_{ij} - y_i$, then $\alpha_i \leftarrow \alpha_i - \eta \gamma_i$. Predict via $h(x_{\text{new}}) = \sum_i \alpha_i \langle x_i, x_{\text{new}} \rangle$.

Define a **kernel function** $K(x, z)$.
?
$K(x, z) = \langle \phi(x), \phi(z) \rangle$ for some feature map $\phi: \mathbb{R}^d \to \mathbb{R}^D$. The trick: $K$ is computable cheaply (often $O(d)$), even when $D \gg d$ or infinite.

What is the **kernel trick**?
?
Whenever a linear algorithm depends on its data **only through inner products** $\langle x_i, x_j \rangle$, you can substitute $K(x_i, x_j)$ everywhere. The algorithm now solves the same problem in the high-dim feature space $\phi$, **without ever materializing $\phi(x)$**.

Compute the kernel for the all-subsets-products embedding $\phi(x) \in \mathbb{R}^{2^d}$.
?
$K(x, z) = \prod_{k=1}^d (1 + x_k z_k)$. Computable in $O(d)$ time, despite the embedding being $2^d$-dimensional.

Verify the all-subsets kernel for $d = 2$.
?
$\phi(x) = (1, x_1, x_2, x_1 x_2)$. $\phi(x) \cdot \phi(z) = 1 + x_1 z_1 + x_2 z_2 + x_1 x_2 z_1 z_2 = (1 + x_1 z_1)(1 + x_2 z_2)$. ✓

What's the polynomial kernel of degree $p$?
?
$K(x, z) = (x^\top z + c)^p$. Implicit feature space = all monomials of degree $\le p$ — dimension $\binom{d+p}{p}$.

What's the **quadratic kernel** used on mock §6?
?
$K(x, z) = (x \cdot z + 1)^2$ — polynomial kernel with $p = 2$, $c = 1$. Embeds into degree-≤2 monomials.

How is the SVM dual derived from the soft-margin primal?
?
Form Lagrangian with multipliers $\alpha_i \ge 0$ for margin constraints, $\mu_i \ge 0$ for slacks. KKT stationarity gives $w^* = \sum_i \alpha_i y_i x_i$ and $\sum_i \alpha_i y_i = 0$ and $0 \le \alpha_i \le C$. Substitute back to get the dual.

Write the **SVM dual** problem.
?
$\max_\alpha \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle$ s.t. $0 \le \alpha_i \le C$ and $\sum_i \alpha_i y_i = 0$. A convex QP in $n$ variables.

How do you kernelize the SVM dual?
?
Replace $\langle x_i, x_j \rangle$ with $K(x_i, x_j)$ everywhere. Same QP, now solving in $\phi$-space. Same form for prediction: $h(x) = \mathrm{sign}(\sum_i \alpha_i y_i K(x_i, x) + b)$ — only support vectors ($\alpha_i > 0$) contribute.

Why are **support vectors** the only ones that matter for prediction?
?
KKT complementary slackness: $\alpha_i > 0$ iff the margin constraint $y_i(w^\top x_i + b) = 1 - \xi_i$ is active. All other points have $\alpha_i = 0$, hence contribute zero to $w^* = \sum_i \alpha_i y_i x_i$ and zero to predictions. (This is mock §1c.)

Which algorithms are kernelizable, and which aren't?
?
**Kernelizable** (depend on data only via inner products): SVM, ridge regression, perceptron, k-NN (after distance trick), PCA (after dualization). **Not kernelizable**: decision trees, random forests, naive Bayes — they look at individual feature values, not inner products.

What's the cost trade-off of kernel methods?
?
Kernel methods cost $O(n^2)$ memory and $O(n^2)$–$O(n^3)$ training time for the $n \times n$ Gram matrix. They pay off when feature dimension $d$ is huge (or infinite, e.g., RBF) and $n$ is moderate. For $n \gg d$ on linear-friendly data, primal methods are cheaper.

What's mock §6 testing about the quadratic-kernel SVM?
?
The slack primal is restated inline (formula sheet provides nothing else). You need to: sketch boundaries for very large $C$ (narrow margin, boundary nails every training point in $\phi$-space, possibly curvy) vs very small $C$ (wide margin, smoother boundary in original space, may misclassify some training points). And **justify in writing** — partial credit lives there.

What does the Gaussian / RBF kernel look like, and what's distinctive about its feature space?
?
$K(x, z) = \exp(-\|x - z\|^2 / 2\sigma^2)$. Implicit feature space is **infinite-dimensional**. Bandwidth $\sigma$ controls smoothness — small $\sigma$ → narrow influence, very flexible boundary (overfit risk); large $\sigma$ → smooth boundary. (Detailed in L16.)

What's the relationship between L09 (linear SVM) and L15 (kernels)?
?
L09 derives the **primal** form geometrically (max-margin, "widest street", soft-margin slack). L15 derives the **dual** via Lagrangian/KKT and shows that the dual depends only on inner products $\langle x_i, x_j \rangle$. The kernel trick then substitutes $K(x_i, x_j)$ to lift the SVM into a non-linear feature space at the cost of an $n \times n$ matrix.

Is XOR linearly separable? What does the kernel trick buy you?
?
No — XOR is the canonical non-linearly-separable 2-class dataset (mock §2c). Apply a quadratic or RBF kernel and it becomes separable in $\phi$-space, even though no straight line works in the original 2D plane.

What does the original $w$ have to do with kernel-SVM testing?
?
Nothing — you never store $w$. Test-time prediction is $h(x_{\text{new}}) = \mathrm{sign}(\sum_i \alpha_i y_i K(x_i, x_{\text{new}}) + b)$ — sum over training points (or just support vectors), evaluate the kernel, sign it. The whole algorithm runs in dual ($\alpha$-space).
