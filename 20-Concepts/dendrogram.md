---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: kmeans_slides.pdf
created: 2026-05-03
---

# Dendrogram

A tree-shaped diagram visualizing the merge order produced by [[hierarchical-clustering|agglomerative clustering]].

## Structure

- **Leaves** (bottom): the $N$ individual data points, listed along the x-axis.
- **Internal nodes**: each represents a merge of two child clusters.
- **Node height** (y-axis): the distance at which the two children were merged. Higher = farther apart at merge time.
- **Root** (top): the final cluster containing all $N$ points.

## Reading a dendrogram

- Two points joined at a **low height** were merged early — they're highly similar.
- Two points joined at a **high height** were merged late — they're dissimilar.
- The **horizontal cut** at any height $h$ partitions the tree into clusters: all leaves below the cut that share an unbroken path are one cluster.

## Choosing the number of clusters by cutting

- **Cut low** → many small, fine-grained clusters.
- **Cut high** → few coarse clusters.
- **Cut at the largest vertical gap** between consecutive merges — heuristic for "natural" $K$. Where the dendrogram has a long vertical edge with no merges, that's a natural breakpoint.

## What dendrograms reveal

- **Cluster cohesion**: short vertical edges mean tight clusters.
- **Outliers**: points that join only at very high heights.
- **Hierarchy**: which clusters are sub-parts of larger ones.
- **Chain effects** (with single linkage): a long thin chain of merges suggests outliers being snake-bridged into a real cluster.

## Limitations

- The visual gets unreadable for $N > $ a few hundred.
- The horizontal **ordering of leaves is not unique** — many valid orderings of the leaves give the same tree topology; software picks one. Don't read meaning into left-to-right adjacency unless the implementation guarantees it.
- The merge **distances depend on the linkage** — comparing dendrograms across linkages directly is misleading.

## Mock §7

The exam asks you to **draw the cluster state after the 1st iteration and at convergence** for hierarchical single-linkage clustering. The full dendrogram isn't requested explicitly, but the merge sequence IS — which is equivalent to the dendrogram up to leaf ordering.

## Exam-relevant facts

- Tree visualization of agglomerative clustering's merge sequence.
- Y-axis = distance at merge; x-axis = leaf points.
- Cut horizontally to extract a flat clustering at any granularity.
- Largest vertical gap → natural choice of $K$.

## Related

- [[hierarchical-clustering]] — what produces dendrograms.
- [[k-means]] — the alternative without a tree.
- [[lecture-17-clustering-kmeans|SLP L17]] — source.
