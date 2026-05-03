---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-DimRed-II(1).pdf
created: 2026-05-03
---

# Manifold learning

A family of **non-linear dimensionality-reduction** methods that recover low-dimensional structure from data lying on (or near) a curved manifold embedded in a higher-dimensional ambient space.

## The premise: the manifold hypothesis

Real-world high-dimensional data (images, text embeddings, gene expression) often lies near a low-dimensional manifold within its ambient space. The intrinsic dimension is small even when the ambient dimension is huge:

- 64×64 grayscale images of a single object rotating: ambient dim = 4096, intrinsic dim ≈ 1 (the rotation angle).
- Face images at fixed identity, varying lighting and pose: ambient dim ≈ pixel count, intrinsic dim ≈ 5–10.

Linear methods like [[principal-component-analysis|PCA]] capture the best linear subspace but miss the curvature. Manifold learning recovers the manifold itself.

## When linear PCA fails (the canonical examples)

| Example | Manifold | PCA verdict |
| --- | --- | --- |
| 1-D spiral in 3-D | curved 1-D curve | fails — squashes the spiral onto a 2-D plane |
| Swiss roll in 3-D | rolled 2-D surface | fails — projects layers onto each other |
| Sphere of points in 3-D | 2-D surface (curved, non-flat) | fails — no faithful 2-D linear projection exists |

## Method families

| Method | Idea |
| --- | --- |
| **[[isomap|ISOMAP]]** | Replace Euclidean with [[geodesic-distance|geodesic]] distances (shortest paths in k-NN graph), run [[multidimensional-scaling|MDS]] |
| **[[kernel-pca\|Kernel PCA]]** | PCA in implicit feature space via a non-linear kernel (RBF, polynomial) |
| **LLE** (Locally Linear Embedding) | Each point ≈ a linear combination of its neighbours; find embedding preserving these weights |
| **Laplacian Eigenmaps** | Eigendecompose the graph Laplacian of the k-NN graph |
| **t-SNE** | Preserve local probabilities of being neighbours (KL-divergence objective). Visualization-focused. |
| **UMAP** | Topological refinement of t-SNE; preserves more global structure. |
| **Autoencoders** | Train a neural net to reconstruct the input through a low-dim bottleneck |

L19 covers ISOMAP and kernel PCA explicitly. The others are not in the deck but are the modern practical methods.

## Trade-offs

- **Global vs local geometry.** ISOMAP preserves global geodesic structure; t-SNE preserves local neighbourhoods at the cost of global distances.
- **Embedding quality vs cost.** Manifold methods cost $O(n^2)$ or worse in $n$ — much more expensive than PCA.
- **Sensitivity to neighbourhood graph.** Many methods need a k-NN graph; the choice of $k$ matters a lot. Too small → disconnected; too large → "short-circuiting" across folds.
- **Out-of-sample extension.** Manifold methods generally don't give a function $f: \mathbb{R}^d \to \mathbb{R}^p$ — embedding new points requires retraining or special tricks. PCA gives an explicit projection matrix.

## Related

- [[isomap]] — the canonical geodesic-MDS algorithm.
- [[kernel-pca]] — the kernelized PCA approach.
- [[multidimensional-scaling]] — used as a subroutine.
- [[geodesic-distance]] — what most manifold methods preserve.
- [[intrinsic-dimension]] — what manifold learning recovers.
- [[principal-component-analysis]] — the linear baseline.
- [[lecture-19-dim-reduction-ii]].
