---
type: study-guide-cluster
course: Statistical-Learning
cluster: "06-unsupervised"
theme: "Unsupervised learning — clustering and dimensionality reduction"
prerequisites:
  - 05-kernels
covers-concepts:
  - k-means
  - hierarchical-clustering
  - dendrogram
  - principal-component-analysis
  - covariance-matrix
  - eigendecomposition
  - singular-value-decomposition
  - intrinsic-dimension
  - multidimensional-scaling
  - kernel-pca
  - isomap
  - manifold-learning
  - geodesic-distance
  - position-distance-similarity
covers-lectures:
  - lecture-17-clustering-kmeans
  - lecture-18-pca
  - lecture-19-dim-reduction-ii
exam-weight: medium-high
---

# Cluster 6: Unsupervised — clustering and dimensionality reduction

> **The story of this cluster in one sentence.** *Everything so far has been supervised — labels were given. We now turn to unsupervised learning, where the only signal is the data itself* — first by **grouping** points (k-means, hierarchical) and then by **summarizing** them (PCA, MDS, kernel-PCA, ISOMAP), all unified by the **Golden Trio**: Position ↔ Similarity ↔ Distance.

## Why this cluster exists

Every previous lecture in this course (L01–L16) was supervised. The training data came as $\{(x_i, y_i)\}_{i=1}^N$ — features paired with labels — and the objective was always "predict $y_i$ from $x_i$ accurately." Cluster 6 turns that off. The training data is now just $\{x_i\}_{i=1}^N$, with no $y$. The only signal comes from the data's own structure: which points are close to each other, how the data spreads, whether clusters emerge naturally from pairwise distances. The training objective changes from "predict $y$" to **"describe the data's structure usefully."** There's no ground-truth answer to compare against; success is judged by how interpretable, useful, or stable the structure you discover is. Model evaluation, hyperparameter selection, and even the notion of "correctness" all change. Clustering (L17, mock §7) and PCA (L18, mock §4) are the two by-hand-compute lectures; L19 generalizes the kernel trick (Cluster 5) into dim-reduction via the **Golden Trio** unification.

**Prerequisites you should feel solid on:**

- [[kernel-trick]] — L19 reuses the L15 machinery for kernel-PCA. The "data-only-through-inner-products" pattern is the key.
- [[support-vector-machine]] (the dual / KKT view) — same reason; L19's Golden Trio is the unsupervised generalization.
- Linear algebra: eigendecomposition, dot products, squared distances. PCA is *built* out of these.
- [[overfitting-underfitting]] / [[bias-variance-decomposition]] — cluster 6 has no labels, so these don't transfer directly, but the *intuition* about underfit/overfit translates to "too few clusters / too few principal components" vs. "too many."

## The arc

Two themes — **grouping** (L17) then **summarizing** (L18–L19) — and one unifying picture (the Golden Trio) tying L19 back to Cluster 5.

### 1. [[k-means]] — hard-assignment, centroid-based clustering

L17 starts simple. Specify the number of clusters $K$ upfront. Initialize $K$ centroids $\mu_1, \ldots, \mu_K$ (e.g., random training points). Iterate: **E-step** — assign each point to the nearest centroid (hard assignment, $r_{nk} \in \{0,1\}$); **M-step** — recompute each centroid as the mean of its assigned points. Repeat until no assignment changes. The objective being minimized is the **distortion** $J = \sum_n \sum_k r_{nk}\|x_n - \mu_k\|^2$ — sum of squared distances from each point to its assigned centroid. Both steps are coordinate descent on $J$, which is why convergence is guaranteed (to a *local* minimum — not necessarily global, since the objective is non-convex). The mock §7 prompt: from a given centroid initialization, draw the clusters after the **1st iteration** and at **convergence**. Most exam problems converge in 2–3 iterations.

### 2. [[hierarchical-clustering]] + [[dendrogram]] — bottom-up, no $K$ required

K-means makes you commit to a $K$ upfront. Hierarchical clustering doesn't. **Agglomerative** (bottom-up): start with each point as its own cluster ($N$ singleton clusters), then repeatedly **merge the two closest clusters** until only one remains. The full merge sequence is a **[[dendrogram]]** — a tree of nested clusterings — and you can cut it at any level to get any number of clusters. The "closeness" of two clusters is defined by a **linkage criterion**: **single-linkage** $d(A, B) = \min_{a \in A, b \in B}\|a-b\|$ (closest pair across the two clusters); complete-linkage uses max; average-linkage uses the average pairwise distance; Ward minimizes within-cluster variance. The mock §7 explicitly tests **single-linkage** — *"treat squares vs. dots as just-symbols (no label distinction for hierarchical)"*. Build the initial pairwise distance matrix; merge the closest pair; recompute distances; repeat.

### 3. [[principal-component-analysis]] + [[covariance-matrix]] + [[eigendecomposition]] — directions of maximum variance

L18 keeps the points and asks the dual question: *which directions in the data carry the most information so we can throw the rest away?* The pipeline you must reproduce from memory for **mock §4** is **center → covariance → eigendecompose → project**:

1. **Center** the data: $\tilde x_i = x_i - \bar x$ where $\bar x = \tfrac{1}{N}\sum_i x_i$.
2. Compute the **[[covariance-matrix]]** $\Sigma = \tfrac{1}{N}\sum_i \tilde x_i \tilde x_i^\top \in \mathbb{R}^{d \times d}$.
3. **Eigendecompose** $\Sigma = V\Lambda V^\top$ — eigenvalues sorted descending, eigenvectors orthonormal.
4. The **first principal component** is the unit eigenvector with the largest eigenvalue (the direction of maximum projected variance); the second PC is the largest-eigenvalue direction orthogonal to the first; and so on.
5. **Project** onto the top $p$ PCs to get a $p$-dim representation.

Two equivalent framings give the same answer (Pythagoras): minimize squared reconstruction error or maximize projected variance. The mock §4 case is a **3-point 2-D toy set** — sample mean (3 numbers averaged), covariance shape ($d \times d = 2 \times 2$), first PC vector. Practice it cold.

### 4. [[singular-value-decomposition]] + [[intrinsic-dimension]] — the practical machinery and the failure mode

For large $d$, eigendecomposing $\Sigma$ directly is expensive. **SVD** of the centered data matrix $X = U\Sigma_{\text{SVD}}V^\top$ gives the same principal directions in $V$ without forming $\Sigma$ explicitly. SVD is also more numerically stable. Choosing the number of components $p$: plot the cumulative explained variance and pick where the curve plateaus (the "elbow") — typically 90–99% of variance.

PCA's failure mode is named **[[intrinsic-dimension]]**: the data may live on a low-dimensional **manifold** that's curved within the ambient space. The canonical example is the **Swiss roll** — 2D points lying on a curved surface in 3D. PCA, being linear, projects onto the best 2D plane and *destroys* the relative positions of points "close along the surface." The intrinsic dimension is 2; the ambient is 3; PCA can't see the difference. This is L19's setup.

### 5. [[position-distance-similarity]] — the Golden Trio

L19 opens with a unifying observation. In Euclidean space, **three matrices encode the same dataset**, and any one can be computed from the others:

| Matrix | Symbol | Definition |
| --- | --- | --- |
| **Position** | $X$ | $n \times d$, rows = data points |
| **Similarity** (Gram) | $K$ | $n \times n$, $K_{ij} = \langle x_i, x_j\rangle$ |
| **Squared distance** | $D$ | $n \times n$, $D_{ij} = \|x_i - x_j\|^2$ |

Conversions: $K = XX^\top$ (position → similarity); $D_{ij} = K_{ii} + K_{jj} - 2K_{ij}$ (similarity → distance); $K = -\tfrac{1}{2} C D C$ where $C = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$ is the centering matrix (distance → similarity, "double-centering" trick); $K = U\Lambda U^\top$, $X = U\Lambda^{1/2}$ (similarity → position, recovered up to a rigid transformation). The Golden Trio is the conceptual scaffolding for everything in L19: choose the matrix most natural for your input, convert to whichever the algorithm needs.

### 6. [[multidimensional-scaling]] — distance → embedding

**Classic MDS** is the literal $D \to X$ algorithm: given a squared-distance matrix $D$, compute $K = -\tfrac{1}{2}CDC$, eigendecompose $K = U\Lambda U^\top$, embed $X = U\Lambda^{1/2}$ truncated to the top $p$ eigenvalues. Useful when your raw input is *distances* — perceptual ratings of how similar two images are, road distances between cities, or a custom domain dissimilarity — and you want a Euclidean embedding for downstream visualization or modelling. The reconstructed $X$ is correct only up to a rigid transformation (rotation, reflection, translation) — the most you can recover from distances alone.

### 7. [[kernel-pca]] — non-linear PCA via kernels

**Kernel PCA** is PCA with the inner-product matrix replaced by a kernel matrix. Replace $K = XX^\top$ with $K_{ij} = k(x_i, x_j)$ for an arbitrary PSD kernel; eigendecompose the (centered) kernel matrix; the leading eigenvectors give the projection of the data onto the principal components **in the implicit feature space** $\phi$. With a Gaussian kernel, kernel PCA can find non-linear structure (e.g., circular structure, manifold curvature) that linear PCA misses. This is the same kernel trick from L15, applied to PCA instead of SVM — the cluster's literal call-back to Cluster 5.

### 8. [[geodesic-distance]] + [[isomap]] + [[manifold-learning]] — unrolling the Swiss roll

**ISOMAP** unrolls the Swiss roll. Step 1: build a $k$-nearest-neighbour graph on the data points (each point connects to its $k$ nearest by Euclidean distance). Step 2: compute **[[geodesic-distance|geodesic distances]]** by shortest paths on this graph (Dijkstra) — distance "along the surface" rather than "through the ambient space." Step 3: feed those geodesic distances into MDS. Result: the unrolled 2D embedding of the Swiss roll. ISOMAP belongs to the broader class of **[[manifold-learning]]** algorithms (LLE, Laplacian Eigenmaps, t-SNE, UMAP) that all share the move "respect local neighbourhoods, ignore global Euclidean structure." All of them inherit the Golden Trio framing — they differ only in *what kind* of distance they measure and *which conversion* of the trio they exploit.

## Connections worth seeing

- **The Golden Trio generalizes the kernel trick (Cluster 5).** The L15 move was *"the SVM dual depends on data only through inner products $\langle x_i, x_j\rangle$, so substitute a kernel"*. The L19 move is the same shape: *MDS depends on data only through squared distances $\|x_i - x_j\|^2$, kernel-PCA only through similarities $K_{ij}$, ISOMAP only through geodesic distances*. All four algorithms are saying "I don't need positions; give me one of {position, similarity, distance} and I'll recover what I need." The Golden Trio is the unsupervised, dim-reduction version of the kernel trick's supervised, classifier version.
- **K-means' E/M alternation is the same shape as gradient descent's iterative refinement.** Both pick a coordinate-block (assignments / centroids; weights), optimize that block while holding the rest fixed, then alternate. The objective monotonically decreases until convergence to a local optimum. K-means is just non-differentiable coordinate descent on the distortion; SGD is stochastic differentiable descent on the loss.
- **PCA and OLS-Ridge (L10) both diagonalize $X^\top X$.** Ridge solves $w = (X^\top X + \lambda I)^{-1} X^\top y$; PCA eigendecomposes $\Sigma = X^\top X / N$ (after centering) and reads off the leading directions. Same matrix, different uses — supervised regression coefficient vs. unsupervised compression direction. The eigenvalue spectrum of $X^\top X$ tells you both how *informative* directions are (PCA) and how *well-conditioned* OLS will be (Ridge regularization is exactly the fix when the spectrum has small eigenvalues).
- **Hierarchical clustering's dendrogram is the unsupervised analogue of a decision tree (L08).** Both produce hierarchical structures by recursive splitting/merging. A decision tree splits *examples* by *attribute values* to predict labels; hierarchical clustering merges *examples* by *pairwise distance* to discover structure. One is supervised top-down; the other is unsupervised bottom-up.
- **The Swiss roll is the Cluster 5 "concentric circles" of unsupervised learning.** Both are visual proofs that linear methods fail on curved data. The fix is the same shape — work in a non-linear feature space — but the tools are reversed: kernel SVM lifts up via $\phi$ and classifies; ISOMAP measures distance along the manifold and embeds back down.

## Common confusions

- **Supervised vs. unsupervised** — the modality flip is real. *Evaluation* changes (no test labels to compare against), *hyperparameter selection* changes (CV on what?), and even *correctness* changes (a "wrong" clustering may still be useful structure). Don't import supervised intuitions blindly.
- **K-means vs. hierarchical clustering** — k-means: hard assignments, requires $K$ upfront, fast ($O(tNKd)$), local optimum. Hierarchical: full merge sequence, no $K$ needed (cut the dendrogram), slow ($O(N^3)$ naive), deterministic. Different beasts; both tested on §7.
- **Single-linkage vs. complete-linkage vs. Ward** — single = closest pair across clusters (creates "chaining"); complete = farthest pair (compact, spherical clusters); Ward = minimizes within-cluster variance (most balanced). Mock §7 specifies single-linkage explicitly.
- **PCA vs. ICA vs. autoencoders** — PCA finds orthogonal directions of maximum variance; ICA finds statistically independent components; autoencoders learn non-linear projections via NN. Out of scope here, but easy to mix up.
- **Eigendecomposition of $\Sigma$ ($d\times d$) vs. of $K$ ($n\times n$)** — PCA can use either; SVD on $X$ unifies both. For $d \ll n$ use $\Sigma$; for $n \ll d$ use $K$ (kernel PCA). Same eigenvalues (up to scaling), corresponding eigenvectors related by $X^\top$.
- **MDS vs. PCA** — both produce a $p$-dim embedding. PCA needs *positions* and computes principal directions in $\mathbb{R}^d$. MDS needs *distances* and recovers positions consistent with them. If you start from positions and use Euclidean distances, MDS reduces exactly to PCA — the two are equivalent in that case.
- **Geodesic vs. Euclidean distance** — Euclidean is "straight line through ambient space," geodesic is "shortest path along the manifold." For a Swiss roll, two points that look close in Euclidean (across the roll) may be far in geodesic (around the roll). ISOMAP uses geodesic, PCA uses Euclidean — that's why ISOMAP unrolls and PCA squashes.
- **The double-centering matrix $C = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$** — pre- and post-multiplying $D$ by $C$ subtracts the row mean and column mean. It's how you go from squared distances back to a centered Gram matrix.

## Self-check (synthesis, not recall)

1. **(blueprint, §7)** Given 5 points in 2D and an initial centroid pair, run k-means by hand for one iteration: assign each point to the nearer centroid, then recompute the centroids as means. Then redo the problem with single-linkage hierarchical clustering: compute the initial pairwise distance matrix, merge the two closest, repeat. Sketch both 1st-iteration states.
2. **(blueprint, §4)** For three 2D points, compute the sample mean, the shape of the covariance matrix, and the first principal component vector by hand. Don't skip the "what shape is $\Sigma$" step — it's $d \times d$, *not* $N \times N$, regardless of how many data points you have.
3. **(synthesis, back to Cluster 5)** Trace through how kernel PCA reuses the L15 machinery. Start from PCA's "diagonalize the Gram matrix $K = XX^\top$" formulation; substitute $K_{ij} \to k(x_i, x_j)$ for an arbitrary PSD kernel; what changes about the eigenvectors and what does it mean to "project a new point onto the kernel PCs"?
4. **(synthesis)** ISOMAP and kernel-PCA both produce non-linear embeddings, but the *route* is different. Describe in one or two sentences each what each algorithm "respects" — local neighbourhoods (ISOMAP) or pairwise similarities in implicit feature space (kernel-PCA) — and why they give different answers on a Swiss roll.
5. **(synthesis)** The Golden Trio says position, similarity, and distance are interconvertible. Why is the conversion *distance → position* unique only up to a rigid transformation? (Hint: what happens to inner products and Euclidean distances when you rotate, reflect, or translate the data?)
6. **(synthesis, course capstone)** Pick one supervised algorithm from Clusters 1–5 (e.g., a kernel SVM with RBF kernel) and one unsupervised algorithm from this cluster (e.g., kernel PCA with the same kernel). Compare what they share (the kernel matrix; the implicit feature space) and what they don't (label use, optimization objective). What does this tell you about the kernel matrix as a "data interface" that can serve both modalities?

## If you have 10 minutes

The minimum viable review for this cluster:

1. The **§4 PCA pipeline** in [[principal-component-analysis]] (or [[lecture-18-pca]]'s "The PCA algorithm" section) — center → covariance → eigendecompose → project, with the 3-point 2-D worked example
2. The **§7 by-hand workflow** in [[lecture-17-clustering-kmeans]] — k-means iteration mechanic and single-linkage hierarchical merge mechanic
3. The **Golden Trio** in [[position-distance-similarity]] — the three matrices, the four conversions, and why kernel-PCA / MDS / ISOMAP all live inside this picture

## Next cluster

→ *None — this is the last cluster.* The course ends here; the unsupervised methods are the syllabus's coda. From here, look back through the spine: the perceptron's three issues (Cluster 1) → trees and SVMs as classical alternatives (Cluster 2) → regularization and bias-variance theory (Cluster 3) → ensembles for moving along the bias-variance curve (Cluster 4) → kernels for enriching a single learner (Cluster 5) → unsupervised structure when the labels are gone (Cluster 6). If you can re-tell that chain in your own words without notes, you can write the exam.
