---
tags: [flashcards, Statistical-Learning]
lecture: 19
created: 2026-05-03
---

# Lecture 19 — Dim Reduction II: MDS, Kernel PCA, ISOMAP

What three matrices form the "Golden Trio" describing a dataset, and what shape is each?
?
**Position** $X$ ($n \times d$, rows = points), **Similarity** (Gram) $K$ ($n \times n$, $K_{ij} = \langle x_i, x_j\rangle$), **Squared distance** $D_s$ ($n \times n$, $D_{s,ij} = \|x_i - x_j\|^2$). All three are interconvertible in Euclidean space.

How do you compute the Gram matrix $K$ from the position matrix $X$?
?
$K = X X^\top$. Direct.

How do you compute squared Euclidean distances from the Gram matrix $K$?
?
$D_{s,ij} = K_{ii} + K_{jj} - 2 K_{ij}$. (Expand $\|x_i - x_j\|^2$ as $\langle x_i, x_i\rangle + \langle x_j, x_j\rangle - 2\langle x_i, x_j\rangle$.)

How do you recover $K$ from a squared-distance matrix $D_s$?
?
Double-centring: $K = -\tfrac{1}{2} C D_s C$ where $C = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$ is the centring matrix. Naive $K = -\tfrac{1}{2} D_s$ is not centred.

What are the three steps of classic MDS (multidimensional scaling)?
?
1) Compute kernel $K = -\tfrac{1}{2} C D_s C$. 2) Eigendecompose $K = U \Lambda U^\top$. 3) Embed $X = U_p \Lambda_p^{1/2}$ taking the top $p$ eigenpairs. Coordinates are unique up to a rigid transformation.

When is MDS useful in practice?
?
When the input is naturally a distance matrix rather than coordinates: psychological similarity ratings, geographic distances, edit distances between strings, geodesic distances on a manifold (then it becomes ISOMAP).

What problem does kernel PCA solve, and what does it diagonalize?
?
Kernel PCA performs PCA in an implicit feature space $\phi(x)$ defined by a kernel $k(x_i, x_j) = \langle \phi(x_i), \phi(x_j)\rangle$. It diagonalizes the **kernel matrix $K$** ($n \times n$) instead of $X^\top X$ ($d \times d$).

How are the eigenvalues of kernel PCA related to the eigenvalues of $K$?
?
$\lambda_{\text{PCA}} = \tfrac{1}{n} \lambda_K^2$. The eigenvectors of $K^2$ are the same as those of $K$, so kernel-PCA eigenvectors are just $K$'s eigenvectors.

When is kernel PCA cheaper than standard PCA?
?
When $n \ll d$ (kernel matrix $n \times n$ is smaller than $d \times d$ covariance), or when the implicit feature space is infinite-dimensional (RBF kernel). Standard PCA is cheaper when $d \ll n$.

What is a geodesic distance, and how does it differ from Euclidean distance?
?
Geodesic = shortest path **along the surface of a manifold**. Euclidean = straight-line distance through the **ambient space**. Two points on opposite layers of a Swiss roll are close in 3-D Euclidean but far apart along the surface — the geodesic captures the manifold's true geometry.

How does ISOMAP approximate geodesic distances?
?
1) Build a k-NN graph: each point connects to its $k$ Euclidean-nearest neighbours (edge weights = Euclidean distance). 2) For any pair of points, geodesic distance ≈ shortest-path distance in the k-NN graph (Dijkstra / Floyd-Warshall). Locally Euclidean ≈ geodesic; globally graph paths approximate the manifold path.

What are the three steps of the ISOMAP algorithm?
?
1) Build the k-NN graph on the data. 2) Compute pairwise geodesic distances as shortest paths in the graph. 3) Run classic MDS on the geodesic distance matrix to get a Euclidean embedding.

Why does ISOMAP succeed where PCA fails on the Swiss roll?
?
Geodesic distance is invariant under unrolling — flatten the roll into a strip and along-the-surface distances are preserved. MDS on geodesic distances finds the Euclidean embedding consistent with these distances, which *is* the unrolled strip. PCA uses Euclidean distances and squashes layers onto each other.

Two ways the ISOMAP k-NN graph can fail — what are they?
?
**Too small $k$:** the graph disconnects, geodesic distances become undefined between components. **Too large $k$:** the graph "short-circuits" across folds — neighbours are picked across nearby manifold folds, and shortest paths cut through the manifold instead of going around it.

What property must a kernel matrix have for it to come from a valid kernel function?
?
**Symmetric and positive semi-definite** (Mercer's condition). This guarantees the implicit feature space exists and is Euclidean.

In one sentence, what is the relationship between standard PCA and classic MDS on Euclidean distances?
?
They produce the **same** embedding. MDS diagonalizes $K = -\tfrac{1}{2} C D_s C$ ($n \times n$); PCA diagonalizes $X^\top X$ ($d \times d$). Both have the right singular vectors of $X$ as outputs.

Why can't kernel PCA be derived by directly substituting kernels into the standard PCA objective?
?
Standard PCA depends on outer products $\sum_i x_i x_i^\top$, not inner products. The substitution $u = X^\top \alpha$ (representer-style "the optimal direction lies in the span of training examples") rewrites the objective as $\alpha^\top K^2 \alpha$, which only depends on $K = XX^\top$. That dualization is the kernelization.

Is L19 likely to appear on the past mock exam?
?
Not directly — the past mock has no L19 problem, and most exam mass goes to L08/L09/L13/L14/L17/L18. But L19 is fair game under MCQs (per the prof's release note about more MCQs in Spring 2026), so know the Golden Trio, kernel-PCA / MDS / ISOMAP at the conceptual level.
