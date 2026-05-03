---
tags: [flashcards, Statistical-Learning]
course: Statistical-Learning
lecture: 04
source: SLP-04(1).pdf
created: 2026-05-03
---

# Lecture 04 — Multi-layer perceptrons

What problem does an MLP solve that a single neuron / single linear layer cannot?
?
**Non-linearly-separable** classification (e.g. XOR, donut-vs-disk, concentric rings). A single linear neuron can only fit a hyperplane; an MLP with hidden layers can fit arbitrarily complex decision boundaries.

Write the forward pass of a one-hidden-layer MLP.
?
$h = f(W_1 x + b_1)$, $\hat{y} = g(W_2 h + b_2)$. $f$ is the hidden activation (sigmoid / tanh / ReLU, applied componentwise); $g$ is the output activation (sigmoid / softmax / identity).

Why must the hidden activation function be non-linear?
?
Because $W_2 (W_1 x + b_1) + b_2 = (W_2 W_1) x + \text{const}$ — a single linear layer. Without non-linearity, depth adds no expressive power. Linear maps can only rotate, reflect, scale, shear; composing them stays linear.

Can softmax be used as a hidden-layer activation? Why or why not?
?
**No.** Softmax couples its outputs through the normalizing constant — the outputs sum to 1, which destroys the rich multi-feature representation hidden layers should learn. Softmax is appropriate only for the *output* layer of a multi-class classifier.

What are the "two views" of a hidden layer in SLP L04?
?
(1) **Decision-boundary view:** each hidden neuron is a half-plane; the output combines them via AND/OR-like operations. (2) **Feature-transformation view:** the hidden layer maps inputs to a new space where the data becomes linearly separable; the output is then a linear classifier in that space.

What's the polar-coordinate analogy in L04?
?
Concentric rings $(x_1, x_2)$ are not linearly separable in Cartesian coordinates, but they are in polar $(r, \theta)$. The hidden layer learns a transformation analogous to that — pushes the data into a representation where a linear output classifier can do the rest.

How many hidden neurons does it take to carve out a triangular positive region in 2-D using AND combination?
?
**Three.** Each defines a half-plane (one side of one edge of the triangle). The AND output is positive only inside the intersection of all three positive half-planes.

What is the universal approximation theorem for neural nets?
?
A feedforward network with **one hidden layer** of sufficient width and a non-polynomial (e.g. sigmoid, ReLU) activation can approximate any continuous function on a compact domain to arbitrary precision (Cybenko 1989, Hornik–Stinchcombe–White 1989).

Three things the universal approximation theorem does NOT guarantee.
?
(1) Doesn't bound the required width — could be exponential in $d$. (2) Doesn't say SGD can find those parameters in practice. (3) Doesn't bound generalization — the approximator can still fail on test data.

If one hidden layer is sufficient in theory, why use multiple hidden layers in practice?
?
(1) **Easier to optimize with SGD** — the loss landscape is empirically more navigable. (2) **Parameter-efficient for compositional problems** — depth $L$ may need $\Theta(d)$ neurons where width-1 needs $\Theta(2^d)$. (3) **Hierarchical representations** — each layer builds features on previous layers' features (mock §1b answer).

What kind of decision regions can a 1-hidden-layer MLP express? 2 hidden layers? 3+?
?
1 hidden layer: convex polygons (intersections of half-planes via AND-like combination). 2 hidden layers: unions of convex polygons → concave shapes, holes. 3+: arbitrary regions (compositions of compositions).

What does the term "deep" in "deep learning" formally mean?
?
The network has **multiple hidden layers** stacked between input and output. Depth = number of hidden layers. The expressive benefit comes from *compositional* feature learning — each layer learns features built on the previous layer's features.

What is ReLU and what's its formula?
?
Rectified Linear Unit: $\mathrm{ReLU}(z) = \max(0, z)$. Derivative is $1$ for $z > 0$ and $0$ for $z < 0$.

Why is ReLU usually preferred over sigmoid for hidden activations?
?
(1) Cheap to compute (max + compare, no exp). (2) Doesn't saturate on the positive side — gradient is exactly 1, no vanishing-gradient problem there. (3) Produces sparse activations (many exact zeros), which empirically helps generalization.

What is a "dead" ReLU unit?
?
A unit whose pre-activation $z = w^T x + b \le 0$ for **every** training example. Its gradient is always 0, so it never updates — effectively dead. Caused by bad initialization or too-large gradient steps.

In an MLP, what does "width" of a hidden layer mean? "Depth"?
?
**Width** = number of neurons (units) in the layer. **Depth** = number of hidden layers in the network. More width = more expressive at one level; more depth = more layers of feature composition.

What is the "decision-boundary view" useful for understanding?
?
For seeing how a small MLP solves XOR-like problems by hand: each hidden neuron carves out a half-plane, and the output Boolean-combines them (e.g. 2 neurons + OR-output for XOR-like). Makes the construction intuitive on small examples.

What is the "feature-transformation view" useful for understanding?
?
For seeing why deep nets work on real data: the hidden layers progressively *un-tangle* the data into a representation where the output classifier's job is easy. Matches how natural data has hierarchical structure (edges → textures → objects).

True or False: a linear classifier can solve the XOR problem if it has enough features.
?
**False** — XOR is not linearly separable in its raw 2D feature space. A linear classifier, regardless of feature count, can only fit a hyperplane; XOR needs a non-linear boundary. Solutions: kernels (L09/15), MLPs (L04), decision trees, $k$-NN.

What three things can linear transforms ($W$ alone, no activation) do to space?
?
**Rotate, reflect, scale, shear.** They preserve grid-line parallelism and even spacing. They cannot bend or curve space — that's what non-linear activations add.

In the L04 worked example, what does the network with 2 hidden sigmoid neurons + 1 OR-like output neuron compute?
?
A two-half-plane intersection / union (depending on weights and biases). For the XOR-shaped toy data, it transforms each input $(x_1, x_2)$ into $(a_1, a_2)$ where the four classes become linearly separable by the output OR neuron.

Why is "more hidden units = more complex decision boundaries" only a *practical* heuristic?
?
Because the universal-approximation theorem already says even modest width can approximate anything. The practical claim is about how *easily* SGD finds good parameters — more units gives more "knobs" to tune, which empirically helps. But more units also risks overfitting and increases compute cost.

What's the role of the output activation in an MLP?
?
Match the output to the task: **sigmoid** for binary classification (probability), **softmax** for multi-class (distribution), **identity / no activation** for regression (real number). The hidden activations are independent of the task; the output activation isn't.
