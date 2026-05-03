---
tags: [flashcards, Statistical-Learning]
lecture: 18
created: 2026-05-03
---

# Lecture 18 — PCA

What is PCA in one sentence?
?
A linear, unsupervised dimensionality-reduction method that finds an orthonormal basis where the projection onto the first axis has the largest variance, the second the largest among directions orthogonal to the first, and so on.

PCA solves which two equivalent optimization problems?
?
(1) Minimize sum of squared perpendicular distances from points to a $p$-dimensional subspace. (2) Maximize the variance of the projected data. They give the same solution because $\|x_i\|^2 = \|\text{projection}\|^2 + \|\text{residual}\|^2$ (Pythagoras).

What are the four steps of the PCA algorithm?
?
1) Center the data ($\tilde{x}_i = x_i - \bar{x}$). 2) Compute the covariance matrix $\Sigma = \frac{1}{n}\tilde{X}^\top \tilde{X}$. 3) Eigendecompose $\Sigma$ to get sorted eigenpairs $(\lambda_k, v_k)$. 4) The $k$-th principal component is $v_k$; project a centered point as $V_p^\top \tilde{x}$.

What shape is the covariance matrix $\Sigma = \frac{1}{n} X^\top X$ for $n$ data points in $\mathbb{R}^d$?
?
$d \times d$, regardless of $n$. (Common exam mistake: writing $n \times n$.)

What is the formula for the $(i, j)$ entry of $\Sigma$ when $X$ is centered?
?
$\Sigma_{ij} = \frac{1}{n}\sum_k x_i^{(k)} x_j^{(k)}$ — the covariance of features $i$ and $j$. Diagonal entries $\Sigma_{ii}$ are the variances of each feature.

Why is $\Sigma$ guaranteed to be eigendecomposable with orthonormal eigenvectors?
?
$\Sigma = \frac{1}{n} X^\top X$ is real and symmetric. Real symmetric matrices have real eigenvalues and an orthonormal eigenbasis (spectral theorem).

What is the first principal component of the data?
?
The unit eigenvector $v_1$ of the covariance matrix $\Sigma$ corresponding to the largest eigenvalue $\lambda_1$. $v_1$ points along the direction of maximum projected variance.

What does the eigenvalue $\lambda_k$ represent?
?
The variance of the data projected onto the $k$-th principal component. (With the $1/n$ convention, $\lambda_k$ already equals that variance.)

Why does setting $w = v_1$ maximize $\|Xw\|^2$ over unit vectors $w$?
?
Writing $w = \sum_k c_k v_k$ in the eigenbasis with $\sum_k c_k^2 = 1$, $\|Xw\|^2 = \sum_k \lambda_k c_k^2$, a convex combination of the eigenvalues. The max is at $c_1 = 1$ (all weight on the largest eigenvalue), which means $w = v_1$.

What is the better algorithm for PCA when $d$ is huge?
?
Run SVD on the centered data: $\tilde{X} = U \Sigma V^\top$. The right singular vectors (columns of $V$) are the principal components; the variances are $\sigma_k^2 / n$. This avoids forming the prohibitive $d \times d$ matrix $X^\top X$.

How does SVD relate the singular values of $X$ to the eigenvalues of $X^\top X$?
?
$X^\top X = V \Sigma^2 V^\top$, so the eigenvalues of $X^\top X$ equal the squared singular values $\sigma_k^2$, and the eigenvectors of $X^\top X$ equal the right singular vectors of $X$.

Worked example — given $x_1 = (0,0)$, $x_2 = (2,2)$, $x_3 = (4,4)$, what is the sample mean?
?
$\bar{x} = (2, 2)$.

Worked example — for the same three points, what is the covariance matrix and its first principal component?
?
$\Sigma = \begin{pmatrix} 8/3 & 8/3 \\ 8/3 & 8/3 \end{pmatrix}$. First PC: $v_1 = \tfrac{1}{\sqrt{2}}(1, 1)^\top$. Eigenvalues: $\lambda_1 = 16/3$, $\lambda_2 = 0$.

How do you choose the number of components $p$ to keep?
?
Pick $p$ to retain a target fraction of total variance: $\frac{\sum_{k=1}^p \lambda_k}{\sum_{k=1}^d \lambda_k} \ge 0.90$ (or 0.95, 0.99). Or look for an "elbow" in the scree plot of $\lambda_k$ vs $k$.

Why does PCA fail on the Swiss roll dataset?
?
PCA is linear — it picks the best $p$-dim subspace through the data. The Swiss roll is a 2-D surface non-linearly embedded in 3-D; projecting onto the best 2-D plane squashes the rolled layers onto each other and destroys the relative positions of points that were close along the surface. Maximum-variance directions ≠ most interesting directions.

What is the difference between intrinsic dimension and ambient dimension?
?
Intrinsic = the minimum number of coordinates needed to describe each point given the geometry of the data. Ambient = the number of features in the input space. A 1-D spiral in 3-D has intrinsic dim 1 but ambient dim 3; PCA only recovers intrinsic dim if the data lies on a linear subspace.

What are eigenfaces?
?
The principal components of a face-image dataset, reshaped from $\mathbb{R}^{hw}$ back into $h \times w$ images. They look like ghostly face-patterns capturing dominant modes of variation (lighting, pose, identity). Any face $x$ can be approximately reconstructed as $\bar{x} + \sum_k c_k v_k$ with $c_k = v_k^\top \tilde{x}$.

Is PCA supervised or unsupervised, and what part of Phase F does it occupy?
?
Unsupervised — uses only $\{x_i\}$, no labels. Phase F covers k-means (L17), PCA (L18), and dim-reduction-II (L19); PCA is the canonical linear unsupervised compression method.

If a problem says "the data is centered and we divide by $n$", what does $\frac{1}{n}X^\top X$ represent?
?
The sample covariance matrix of the features. Diagonal entries are per-feature variances; off-diagonal entries are pairwise covariances.

In the rank-1 sum form, how does the eigendecomposition write $A$ for a real symmetric matrix?
?
$A = \sum_k \lambda_k v_k v_k^\top$ — a weighted sum of rank-1 outer products of the orthonormal eigenvectors.

Why is the projection of a centered point onto the $k$-th PC just $v_k^\top \tilde{x}$ (a scalar)?
?
$v_k$ is a unit vector. The signed length of the projection of $\tilde{x}$ onto the line spanned by $v_k$ is the dot product $v_k^\top \tilde{x}$. To get the projection vector itself, multiply by $v_k$: $(v_k^\top \tilde{x}) v_k$.
