---
tags: [flashcards, Statistical-Learning]
---

# Lecture 17 — Clustering: K-means and hierarchical

What's the modality flip from L01–L16 to L17?
?
**Supervised → unsupervised.** L01–L16 had labeled data $\{(x_i, y_i)\}$ and trained models to predict $y$ from $x$. L17 onward has only $\{x_i\}$ — no labels — and the goal becomes **describing the data's structure usefully** (e.g., grouping points by similarity).

Define the **k-means objective function** $J$.
?
$J = \sum_{n=1}^N \sum_{k=1}^K r_{nk} \|x_n - \mu_k\|^2$ — the sum of squared distances from each point to its assigned cluster's center. Also called the **distortion** or **within-cluster sum of squares**.

Write the k-means **E-step**.
?
$r_{nk} = 1$ iff $k = \arg\min_j \|x_n - \mu_j\|^2$, else 0. Each point goes to the nearest centroid. **Hard assignment**.

Write the k-means **M-step**.
?
$\mu_k = \frac{\sum_n r_{nk} x_n}{\sum_n r_{nk}}$ — the new cluster center is the **mean of the points currently assigned to that cluster** (the centroid).

Why does k-means converge?
?
Each E-step and M-step **monotonically decreases (or holds) $J$**. Since $J \ge 0$ is bounded below, the algorithm must terminate in finitely many iterations. Convergence is to a **local minimum** — different initializations may give different final clusterings.

What's the computational cost of one k-means iteration? Total cost?
?
Per iteration: $O(NKd)$ for the E-step + $O(Nd)$ for the M-step ≈ $O(NKd)$. Total: $O(t N K d)$ where $t$ = number of iterations until convergence. **Scales linearly** in $N$ and $d$.

List the four main limitations of k-means.
?
1. **Sensitive to initialization** — random restarts mitigate.
2. Converges to a **local minimum**, not global.
3. **$K$ must be specified upfront** (use elbow method or domain knowledge).
4. **Assumes spherical clusters of similar size** — squared-Euclidean distance breaks on elongated or anisotropic clusters.

What's the **elbow method** for choosing $K$?
?
Plot $J$ (or "inertia") vs $K$. Look for the "elbow" — the $K$ at which adding another cluster stops dramatically reducing $J$. Pick the elbow value. Heuristic, not exact.

What does the agglomerative hierarchical clustering algorithm do?
?
1. Start with each point as its own cluster ($N$ singleton clusters).
2. Compute pairwise inter-cluster distances using a linkage criterion.
3. Merge the closest pair.
4. Update the distance matrix.
5. Repeat until all points are merged into one cluster.

What is a **dendrogram**?
?
A tree visualization of agglomerative clustering's merge sequence. **Leaves** = data points (along x-axis); **node height** = distance at which two child clusters were merged. Cut horizontally at any height to extract a flat clustering at that granularity.

Define **single linkage**.
?
$d(A, B) = \min_{a \in A, b \in B} \|a - b\|$ — the distance between two clusters is the **minimum pairwise distance** between their members. This is the linkage tested on mock §7.

Define **complete linkage** and **average linkage**.
?
- **Complete:** $d(A, B) = \max_{a, b} \|a - b\|$.
- **Average:** $d(A, B) = \frac{1}{|A|\cdot|B|} \sum_{a, b} \|a - b\|$.

What is **Ward's linkage**?
?
Merge the two clusters that **minimize the increase in total within-cluster variance**. Tends to produce balanced, similarly-sized, spherical clusters. Often the default in practice.

Compare cluster-shape preferences of single, complete, and Ward linkages.
?
- **Single:** elongated chains, non-convex shapes; sensitive to noise (chaining).
- **Complete:** compact, spherical clusters; sensitive to outliers (the max is dragged by the worst point).
- **Ward:** balanced, similarly-sized, spherical clusters.

What's the computational cost of naive hierarchical clustering?
?
$O(N^3)$ time, $O(N^2)$ memory in the naive implementation. Optimized variants exist (e.g., SLINK for single linkage achieves $O(N^2)$). **Doesn't scale past ~10k points** without careful implementation.

What are the main pros of hierarchical vs k-means?
?
- **No $K$ needed upfront** (cut the dendrogram at any height).
- **Multi-resolution**: see structure at every granularity.
- **Deterministic** — no random initialization, no local-optimum issue.

What are the main cons of hierarchical vs k-means?
?
- **$O(N^3)$ cost** vs $O(tNKd)$ for k-means.
- **Greedy and irreversible**: bad early merges propagate.
- **Sensitive to noise**, especially single linkage.

How would you walk through the **mock §7** k-means problem from a given centroid initialization?
?
Loop until assignments stabilize: (1) compute distance from each data point to each centroid; (2) assign each point to the nearest centroid; (3) recompute each centroid as the mean of its assigned points. Sketch clusters after iteration 1 and at convergence. Most exam problems converge in 2–3 iterations.

How would you walk through the mock §7 single-linkage hierarchical clustering problem?
?
1. Build initial $N \times N$ pairwise distance matrix between all data points.
2. Find the smallest entry; merge those two points/clusters.
3. Update the distance matrix: new cluster's distance to each remaining cluster = **minimum** over its constituent points.
4. Repeat until you reach the requested number of clusters (or all merged).
5. Sketch state after iteration 1 and at convergence.

Why is the "E-step / M-step" naming used for k-means?
?
K-means is the **hard-assignment limiting case** of EM (Expectation-Maximization) for a Gaussian mixture model with isotropic covariance. As cluster variance $\sigma^2 \to 0$, the soft posterior $p(k \mid x_n)$ collapses to a one-hot vector — exactly the k-means assignment. Hence the EM-style naming.

What does it mean that k-means assumes "spherical clusters of similar size"?
?
The squared-Euclidean distance metric implicitly assumes **isotropy** (no preferred direction) and **uniform scale** across clusters. Elongated, anisotropic, or wildly different-sized clusters violate this — k-means tends to break them up incorrectly. Other algorithms (DBSCAN, GMM with full covariance) handle these cases.
