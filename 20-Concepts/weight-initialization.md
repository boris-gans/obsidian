---
tags: [concept]
courses: [Statistical-Learning]
sources:
  - course: Statistical-Learning
    file: Lec-06-improving-MLPs(1).pdf
created: 2026-05-03
---

# Weight initialization (Xavier / He)

The choice of initial weights $W^{(\ell)}$ in a deep network controls whether **forward activations and backward gradients stay balanced across layers** or collapse / explode. Bad init can make a network untrainable even with the right activation function and the right loss.

## Why init matters: the variance-balance problem

Pre-activation at layer $\ell + 1$ is

$$
z^{(\ell+1)} = W^{(\ell+1)} a^{(\ell)}.
$$

If we initialize $W^{(\ell+1)}_{ij} \sim \mathcal{N}(0, \sigma^2)$ with the inputs $a^{(\ell)}$ having variance $v$ across $D_\text{in}$ dimensions:

$$
\text{Var}(z^{(\ell+1)}) \;=\; D_\text{in} \cdot \sigma^2 \cdot v.
$$

Two failure modes:
- **$\sigma$ too small** → $\text{Var}(z) \to 0$ layer by layer → activations vanish, sigmoid units land near $0.5$ everywhere, network can't represent anything. ([[30-Sources/Statistical-Learning/pdf/Lec-06-improving-MLPs(1).pdf#page=78|slide ~78]] — `W = 0.01 * np.random.randn(...)` "works ~okay for small networks, but problems with deeper networks.")
- **$\sigma$ too large** → $\text{Var}(z)$ grows exponentially → activations saturate (sigmoid/tanh) or explode (ReLU).

**The fix:** pick $\sigma^2$ so that $D_\text{in} \cdot \sigma^2 = 1$ — i.e., $\sigma^2 = 1 / D_\text{in}$. Then variance is preserved layer-by-layer in the forward pass (assuming $v \approx 1$).

## Xavier / Glorot initialization (designed for tanh)

$$
W_{ij} \sim \mathcal{N}\!\left(0,\; \frac{1}{D_\text{in}}\right) \quad \text{or equivalently} \quad W = \frac{1}{\sqrt{D_\text{in}}} \mathcal{N}(0, 1).
$$

Sometimes written with the **average** of $D_\text{in}$ and $D_\text{out}$:

$$
\sigma^2 = \frac{2}{D_\text{in} + D_\text{out}},
$$

which balances forward-pass *and* backward-pass variance simultaneously. The difference is small in practice.

L06 motivates Xavier by showing the variance-balance derivation for $\tanh$ ([[30-Sources/Statistical-Learning/pdf/Lec-06-improving-MLPs(1).pdf#page=85|slides ~83–88]]).

## He / Kaiming initialization (designed for ReLU)

$$
W_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{D_\text{in}}\right).
$$

**The factor of 2 compensates for ReLU.** When $W$ is zero-mean Gaussian and inputs are zero-mean, half the pre-activations are negative in expectation — ReLU outputs zero on those, so only **half** the input variance is "passed through" each layer. To preserve activation variance, double $\sigma^2$.

L06's pseudocode ([[30-Sources/Statistical-Learning/pdf/Lec-06-improving-MLPs(1).pdf#page=90|slide ~90]]):

```python
W = np.random.randn(Din, Dout) * np.sqrt(2 / Din)
```

The "Optional derivation (won't be covered in the exam)" slide section ([[30-Sources/Statistical-Learning/pdf/Lec-06-improving-MLPs(1).pdf#page=92|slides ~91–95]]) walks through the variance algebra. The prof flagged it as off the exam — **memorize the formulas $1/D_\text{in}$ (Xavier/tanh) and $2/D_\text{in}$ (He/ReLU)**, not the derivation.

## Why this matters in practice — the empirical gap

Without good init, a 7-layer MLP with `W = 0.01 * randn(...)` produces activations that vanish to ~0 in the first 2–3 layers. With `W = randn(...) / sqrt(Din)`, activations stay healthy across all 7 layers. The change is just **a scale factor** in `W`, but it's the difference between a network that trains and one that doesn't.

L06 demos this with a histogram-per-layer plot ([[30-Sources/Statistical-Learning/pdf/Lec-06-improving-MLPs(1).pdf#page=82|slides ~80–88]]): bad init → all activations crammed near zero in deep layers; Xavier/He → roughly the same distribution in every layer.

## Init prevents dead ReLUs at the start

Connection to the [[relu#The dead-ReLU problem L06|dead-ReLU problem]]: a unit dies when $w^\top x + b < 0$ on every training example. He init keeps initial pre-activations centered at zero with controlled variance, so a typical unit fires for roughly half the training set — well clear of the all-negative trap. **L06 explicitly notes "good weight initialization also helps prevent this!"** ([[30-Sources/Statistical-Learning/pdf/Lec-06-improving-MLPs(1).pdf#page=75|slide ~75]]).

## Quick reference table

| Activation | Init | Formula |
| --- | --- | --- |
| **tanh** | Xavier / Glorot | $\sigma^2 = 1 / D_\text{in}$ |
| **ReLU / Leaky ReLU** | He / Kaiming | $\sigma^2 = 2 / D_\text{in}$ |
| **sigmoid** | Xavier (rarely used in hidden layers anyway) | $\sigma^2 = 1 / D_\text{in}$ |
| **GELU / SELU** | Variants of He, sometimes paired with batch norm | — |

## Exam-relevant facts

- Initial weights are drawn from a zero-mean Gaussian; only the **variance** changes by activation choice.
- **Xavier ↔ tanh** ($\sigma^2 = 1/D_\text{in}$); **He ↔ ReLU** ($\sigma^2 = 2/D_\text{in}$).
- The factor of 2 in He init **compensates for ReLU's half-zeroing of negative pre-activations**.
- Naive small-random init (`0.01 * randn`) is *not* a valid choice for deep networks — it causes activations to vanish layer-by-layer.
- The variance-balance derivation is **not on the exam** per the prof.

## Related

- [[relu]] — the activation that motivates He init.
- [[vanishing-exploding-gradients]] — the failure mode init prevents.
- [[multilayer-perceptron]] — where initialization happens.
- [[lecture-06-improving-mlps|SLP L06]] — source.
