---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 18 - Beyonf RNNs.pdf
created: 2026-05-02
---

# Gated Rectified Unit (GRU)

A simplified gated [[recurrent-neural-network|RNN]] cell that **merges the LSTM's forget and input gates into a single update gate** and removes the separate cell state — the hidden state itself carries the information ([[30-Sources/NLP/pdf/Session 18 - Beyonf RNNs.pdf#page=8|slide 8]]). Faster and less parameter-heavy than [[lstm|LSTM]] but with usually slightly weaker long-range modelling.

The blueprint flags this as **medium weight**: Quiz IV Q4, Q4.B target the gate roles (reset / update).

## The two gates ([[30-Sources/NLP/pdf/Session 18 - Beyonf RNNs.pdf#page=8|slide 8]])

$$z_t = \sigma(W_z x_t + U_z h_{t-1}) \qquad \text{update gate}$$
$$r_t = \sigma(W_r x_t + U_r h_{t-1}) \qquad \text{reset gate}$$
$$\tilde{h}_t = \tanh(W x_t + U(r_t \odot h_{t-1})) \qquad \text{candidate hidden state}$$
$$\boxed{h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t} \qquad \text{hidden state — convex blend}$$

Note the absence of a separate cell state: **the hidden state itself carries the information** ([[30-Sources/NLP/pdf/Session 18 - Beyonf RNNs.pdf#page=8|slide 8]]).

## What each gate does

| Gate | Symbol | Role |
|---|---|---|
| **Reset** | $r_t$ | Controls **how much past info enters the candidate**. $r_t \approx 0$ → ignore $h_{t-1}$ when computing $\tilde{h}_t$; $r_t \approx 1$ → use $h_{t-1}$ fully. |
| **Update** | $z_t$ | Controls the **mix between old and new hidden state**. $z_t \approx 0$ → keep $h_{t-1}$ unchanged; $z_t \approx 1$ → fully replace with $\tilde{h}_t$. |

The update gate effectively performs the role of LSTM's forget + input gates **combined**: it both gates the previous state ($1 - z_t$) and gates the new candidate ($z_t$) in a single coupled operation.

## GRU vs LSTM ([[30-Sources/NLP/pdf/Session 18 - Beyonf RNNs.pdf#page=8|slide 8]], [[30-Sources/NLP/pdf/Session 18 - Beyonf RNNs.pdf#page=9|slide 9]])

| Property | LSTM | GRU |
|---|---|---|
| Number of gates | 3 (forget, input, output) | 2 (reset, update) |
| Separate cell state? | Yes ($c_t$) | No — hidden state carries info |
| Parameters per cell | More | Fewer (~75% of LSTM) |
| Speed | Slower | Faster |
| Long-range capacity | Stronger | Slightly weaker |
| When preferred | Long sequences, complex dependencies | Smaller datasets, faster training |

> "GRU simplifies the LSTM structure by merging the forget and input gates into a single update gate, and does not have a memory cell. It is faster and less complex than LSTM, but its ability to capture long-range dependencies is usually weaker." ([[30-Sources/NLP/pdf/Session 18 - Beyonf RNNs.pdf#page=8|slide 8]])

## Why GRU works

The hidden-state update $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ is a **convex combination** of the previous hidden state and the new candidate. When $z_t \approx 0$, $h_t \approx h_{t-1}$ — information is preserved across many steps without multiplicative shrinkage. This is the same additive-path trick LSTMs use, just with one gate instead of two.

## Exam framing

| Question | Answer |
|---|---|
| What's the role of the **update** gate? | Mixes old and new hidden state: $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$. $z_t \to 0$ keeps the past, $z_t \to 1$ refreshes (Quiz IV Q4) |
| What's the role of the **reset** gate? | Controls how much past info enters the candidate $\tilde{h}_t$ — $r_t \to 0$ ignores $h_{t-1}$ when forming the candidate |
| Does GRU have a separate cell state? | **No** — the hidden state itself carries the information |
| GRU vs LSTM — when to prefer GRU? | Smaller datasets, faster training, fewer parameters. When to prefer LSTM: long sequences, complex dependencies. |

## Related

- [[lstm|LSTM]] — the more elaborate 3-gate sibling
- [[recurrent-neural-network|Vanilla RNN]] — what GRU extends
- [[vanishing-exploding-gradients]] — what gating addresses
