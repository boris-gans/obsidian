---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: kmeans_slides.pdf
created: 2026-05-03
---

# K-means clustering

A hard-assignment, centroid-based, partitional clustering algorithm. **Specify $K$ upfront**, partition $N$ data points into $K$ clusters by alternating between two steps until the assignments stabilize. Minimizes within-cluster variance (the "distortion") locally.

K-means is the canonical **unsupervised** algorithm — no labels are involved at any stage. Often referred to as "Lloyd's algorithm" after its inventor (1957, published 1982).

## Setup

- Data: $\{x_1, \ldots, x_N\}$ with $x_n \in \mathbb{R}^D$.
- Number of clusters $K$ is given (you choose it).
- Indicator variables $r_{nk} \in \{0, 1\}$: $r_{nk} = 1$ iff $x_n$ assigned to cluster $k$. **Hard assignment** — each point in exactly one cluster.
- Cluster centers (centroids) $\mu_1, \ldots, \mu_K \in \mathbb{R}^D$.

## Objective: distortion

$$
J = \sum_{n=1}^N \sum_{k=1}^K r_{nk}\, \|x_n - \mu_k\|^2.
$$

The sum of squared distances from each point to its assigned cluster's center. Minimize jointly over $\{r_{nk}\}$ and $\{\mu_k\}$.

## The algorithm

1. **Initialize** centers $\mu_1, \ldots, \mu_K$ (e.g., pick $K$ random training points).

**Repeat until convergence:**

2. **E-step** (assign): each point goes to the nearest centroid.
$$
r_{nk} = \begin{cases} 1 & k = \arg\min_j \|x_n - \mu_j\|^2 \\ 0 & \text{otherwise}\end{cases}
$$

3. **M-step** (update centers): each new center is the mean of the points currently assigned to it.
$$
\mu_k = \frac{\sum_{n=1}^N r_{nk}\, x_n}{\sum_{n=1}^N r_{nk}}.
$$

4. **Stop** when no assignment changes (convergence).

## Why it converges

Each step **monotonically decreases $J$** (or leaves it unchanged):
- E-step: each point picks the assignment minimizing $\|x_n - \mu_k\|^2$ over $k$ — locally optimal.
- M-step: the mean is the unique minimizer of $\sum_n r_{nk} \|x_n - \mu_k\|^2$ over $\mu_k$.

Since $J \ge 0$ is bounded below, the sequence of $J$ values must converge in finitely many iterations. But convergence is to a **local minimum**, not the global one — different initializations give different final clusterings.

## Limitations

| Limitation | Why it matters |
| --- | --- |
| Sensitive to initialization | Bad seeds → bad local minimum. Mitigation: random restarts; pick lowest-$J$ run. K-means++ is the standard smart initialization. |
| Local minimum | Not globally optimal. Restarts help. |
| $K$ is a hyperparameter | Must be specified. Use elbow method, silhouette, or domain knowledge. |
| Spherical-cluster assumption | Squared-Euclidean distance favors spherical, similar-sized clusters. Elongated, anisotropic, or wildly different-sized clusters break k-means. |
| Sensitive to feature scale | Use [[feature-normalization]]. |

## Computational cost

- E-step: $O(NKd)$ per iteration.
- M-step: $O(Nd)$ per iteration.
- **Total: $O(t N K d)$** where $t$ = number of iterations until convergence.
- $t$ is typically small (5–20). K-means **scales linearly** in $N$ and $d$ — large-data-friendly.

## Choosing K — the elbow method

Plot $J$ (sometimes called "inertia") vs $K$. Look for the "elbow" — the $K$ at which adding another cluster stops dramatically reducing $J$. Pick the elbow value. Heuristic only.

Alternatives: silhouette coefficient, gap statistic, BIC (for the EM/Gaussian-mixture connection).

## Connection to EM and Gaussian mixtures

K-means is the **hard-assignment limiting case** of EM for a Gaussian mixture model with isotropic covariance: as the cluster variance $\sigma^2 \to 0$, the soft posterior $p(k \mid x_n)$ collapses to a one-hot vector — exactly the k-means assignment. The "E-step / M-step" terminology is borrowed from this EM connection.

## Mock §7 mechanics

Given an initial set of centroids:
1. For each data point, compute the distance to each centroid; assign to the nearest.
2. Recompute each centroid as the mean of the points now in its cluster.
3. Iterate until no assignment changes.
4. Sketch the cluster boundaries after iteration 1 and at convergence.

Most exam problems converge in 2–3 iterations. The blueprint expects **draw clusters after 1st iteration and at convergence**.

## Exam-relevant facts

- Objective $J = \sum_n \sum_k r_{nk}\|x_n - \mu_k\|^2$ — within-cluster sum of squared distances.
- E-step: assign each point to nearest centroid (Euclidean).
- M-step: each new centroid is the mean of points currently in that cluster.
- Iterates until no assignment changes — converges to a local minimum.
- $K$ must be specified upfront.
- Sensitive to initialization → run multiple random restarts.
- Cost $O(tNKd)$ — scales linearly in $N$ and $d$.

## Related

- [[hierarchical-clustering]] — alternative bottom-up approach that doesn't need $K$ upfront.
- [[feature-normalization]] — k-means is scale-sensitive; normalize first.
- [[bias-variance-decomposition]] — clustering also faces a bias-variance trade-off (small $K$ underfits structure; large $K$ overfits noise).
- [[lecture-17-clustering-kmeans|SLP L17]] — source.
