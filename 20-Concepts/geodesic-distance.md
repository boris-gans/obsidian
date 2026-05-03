---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: SLP-DimRed-II(1).pdf
created: 2026-05-03
---

# Geodesic distance

A **geodesic** is the generalization of a "straight line" to curved surfaces (or higher-dimensional manifolds). The **geodesic distance** between two points is the length of the shortest path *along the manifold's surface*, in contrast to **Euclidean distance**, which is measured *through the ambient space*.

## Concrete example

The shortest path between two cities on Earth is a great-circle arc along the surface (the path a plane flies). The Euclidean distance through the planet's interior is meaningless for travel.

| Path | Type | Used by |
| --- | --- | --- |
| Tunnel through the Earth | Euclidean | none — the Earth is in the way |
| Surface arc (great circle) | Geodesic | airplanes, ships, kNN on a manifold |

## On the Swiss roll

Two points on opposite layers of a rolled-up sheet might be 1 cm apart through space (Euclidean) but 30 cm apart along the surface (geodesic). The geodesic captures the dataset's true geometry; the Euclidean distance gives misleading "shortcuts" through the ambient space.

## Computing geodesic distances on data

For a sampled point cloud lying on a manifold, the true geodesic is unknown but can be **approximated by graph distances**:

1. Build a k-NN graph: each point connects to its $k$ Euclidean-nearest neighbours.
2. Compute shortest paths in the graph (Dijkstra / Floyd-Warshall).
3. For nearby points, edge weight ≈ true geodesic (manifold is locally flat).
4. For distant points, the shortest path through the graph approximates the true geodesic.

This approximation is the heart of [[isomap|ISOMAP]].

## Why geodesics matter for dimensionality reduction

Geodesic distances are **invariant under unrolling**. If a manifold can be flattened (Swiss roll, paper sheet), the geodesic distances between points are preserved by the flattening. Plug them into [[multidimensional-scaling|MDS]] and you recover the unrolled embedding.

[[principal-component-analysis|PCA]] uses Euclidean distances and **cannot** unroll a curved manifold. ISOMAP uses geodesics and **can**.

## Related

- [[isomap]] — the canonical algorithm.
- [[manifold-learning]] — the broader class of methods.
- [[multidimensional-scaling]] — the embedder that turns geodesic distances into coordinates.
- [[intrinsic-dimension]].
- [[lecture-19-dim-reduction-ii]].
