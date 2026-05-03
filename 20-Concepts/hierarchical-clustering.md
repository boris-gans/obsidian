---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: kmeans_slides.pdf
created: 2026-05-03
---

# Hierarchical clustering

A clustering paradigm that builds a **full tree of nested clusterings** instead of a single flat partition. The tree (the **dendrogram**) shows how clusters merge (or split) at increasing distance scales. Cut the tree at any height to extract a flat clustering of any granularity. Unlike [[k-means]], **no $K$ needs to be specified in advance**.

## Two flavors

- **Agglomerative (bottom-up)**: start with each point as its own cluster; iteratively merge the closest pair until everything's one cluster. The textbook standard. The deck focuses here.
- **Divisive (top-down)**: start with all points in one cluster; recursively split. Less common.

## Agglomerative algorithm

> 1. Start with each point as its own cluster ($N$ singleton clusters).
> 2. Compute pairwise distances between clusters using a chosen [[#Linkage criteria|linkage criterion]].
> 3. Merge the two closest clusters.
> 4. Update the distance matrix.
> 5. Repeat steps 3–4 until all points are merged into one cluster.

Each merge corresponds to a node in the [[dendrogram]]; the merge's distance becomes the height of that node.

## Linkage criteria

The "distance between two clusters" $A$ and $B$ collapses pairwise point distances into a single number. Different linkages give dramatically different dendrograms.

| Linkage | $d(A, B)$ | Cluster shape preference | Failure mode |
| --- | --- | --- | --- |
| **Single linkage** | $\min_{a \in A, b \in B} \|a - b\|$ | elongated chains, non-convex shapes | "chaining": noise points bridge real clusters |
| **Complete linkage** | $\max_{a \in A, b \in B} \|a - b\|$ | compact, spherical clusters | sensitive to outliers (the max is dragged by the worst point) |
| **Average linkage** | $\frac{1}{|A|\cdot |B|}\sum_{a, b} \|a - b\|$ | compromise between single & complete | balanced but not specifically optimized for any shape |
| **Ward's method** | merge the pair that minimizes the **increase in total within-cluster variance** | balanced, similarly-sized, spherical | similar to complete; can over-prefer balance |

The **mock-exam single-linkage** convention is "min over all pairs." On the exam:
1. Build initial $N \times N$ distance matrix.
2. Find smallest entry, merge those two points/clusters.
3. Update: new cluster's distance to others is the **minimum** over its constituent points.
4. Repeat until you have the requested number of clusters.

## Pros

- **No $K$ upfront**: cut the dendrogram wherever you want.
- **Multi-resolution view**: see structure at every granularity.
- **Deterministic** given linkage and metric — no random initialization, no local-optimum issue.
- **Interpretable** — dendrogram visualizes the merge order.

## Cons

- **Computationally expensive**: naive $O(N^3)$ time, $O(N^2)$ memory. Optimized variants for some linkages: $O(N^2 \log N)$ (e.g., SLINK for single linkage achieves $O(N^2)$).
- **Greedy and irreversible**: once two clusters merge, the algorithm can't undo it. Bad early merges propagate.
- **Sensitive to noise**, especially single linkage. An outlier can chain otherwise-distinct clusters.
- **Doesn't scale past ~10k points** without optimized implementations.

## Compared to k-means

| Property | K-means | Hierarchical |
| --- | --- | --- |
| Hard assignment | Yes | Yes |
| Specify $K$ upfront | Required | Not needed (cut tree at any height) |
| Output | Flat partition | Full hierarchy |
| Random init | Required | Not needed (deterministic) |
| Time complexity | $O(tNKd)$, scales linearly | $O(N^3)$ naive, doesn't scale well |
| Cluster shapes | Spherical, similar-sized | Depends on linkage (single→chains, complete/Ward→spherical) |
| Output stability | Varies with init | Deterministic |

K-means wins on speed; hierarchical wins on flexibility and interpretability.

## Exam-relevant facts

- Bottom-up: start with $N$ singleton clusters, iteratively merge the closest pair.
- **Single linkage**: $d(A, B) = \min_{a \in A, b \in B} \|a - b\|$. Mock §7's specified linkage.
- Complete: max; average: mean over pairs; Ward: minimize increase in variance.
- Produces a [[dendrogram]] — cut at any height for a flat clustering.
- $O(N^3)$ naive cost; doesn't scale past ~10k points.
- Single linkage is sensitive to noise (chaining); complete is sensitive to outliers.

## Related

- [[k-means]] — the partitional alternative.
- [[dendrogram]] — the tree visualization.
- [[lecture-17-clustering-kmeans|SLP L17]] — source.
