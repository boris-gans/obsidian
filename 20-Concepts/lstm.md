---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 18 - Beyonf RNNs.pdf
created: 2026-05-02
---

# Long Short-Term Memory (LSTM)

A gated [[recurrent-neural-network|RNN]] architecture that introduces a **memory cell** $c_t$ and **three gates** (forget, input, output) to control the flow of information across time, allowing the network to capture **long-range dependencies** that vanilla RNNs cannot ([[30-Sources/NLP/pdf/Session 18 - Beyonf RNNs.pdf#page=7|slide 7]]).

The blueprint flags this as **medium weight**: Quiz IV Q3, Q3.B target the gate roles.

## The three gates and the cell state ([[30-Sources/NLP/pdf/Session 18 - Beyonf RNNs.pdf#page=7|slide 7]])

$$f_t = \sigma(W_f x_t + U_f h_{t-1}) \qquad \text{forget gate}$$
$$i_t = \sigma(W_i x_t + U_i h_{t-1}) \qquad \text{input gate}$$
$$\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1}) \qquad \text{candidate cell update}$$
$$\boxed{c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t} \qquad \text{cell state — long-term memory}$$
$$o_t = \sigma(W_o x_t + U_o h_{t-1}) \qquad \text{output gate}$$
$$h_t = o_t \odot \tanh(c_t) \qquad \text{hidden state — what to expose}$$

All gate vectors lie in $[0, 1]^d$ (sigmoid output) and act as **element-wise switches**.

## What each gate does

| Gate | Symbol | Operates on | Role |
|---|---|---|---|
| **Forget** | $f_t$ | $c_{t-1}$ | "What to **erase**" from long-term memory. $f_t \approx 0$ → forget; $f_t \approx 1$ → keep. |
| **Input** | $i_t$ | $\tilde{c}_t$ | "What to **write**" into long-term memory. Modulates the candidate update. |
| **Output** | $o_t$ | $\tanh(c_t)$ | "What to **expose**" — selects which parts of the cell state appear in the hidden state $h_t$. |

The **cell state** $c_t$ is the long-term memory channel; the **hidden state** $h_t$ is what gets exposed to the next step and to any downstream layers (output, attention, etc).

## Why gating fixes vanishing gradients

The vanilla RNN update $h_t = \tanh(W h_{t-1} + U x_t)$ applies the **same multiplicative transformation** at every step — the BPTT gradient becomes $\prod_{k} W^\top \mathrm{diag}(1 - \tanh^2)$, which decays exponentially when the spectral radius < 1.

The LSTM cell state update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ has an **additive path** through time. When $f_t \approx 1$, $c_t \approx c_{t-1} + \text{small}$ — gradients pass through the additive path without exponential decay. **The model can learn long-range dependencies that vanilla RNNs cannot reach.**

## Pros and cons ([[30-Sources/NLP/pdf/Session 18 - Beyonf RNNs.pdf#page=9|slide 9]])

**Pros:**
- Learns **what to retain, update, or discard** via gating
- Preserves signals across longer sequences than vanilla RNNs
- **Structured memory** — different components of the cell state correspond to stored, filtered, or exposed information

**Cons:**
- Still **strictly sequential** — $h_t$ depends on $h_{t-1}$; no parallelism across time
- All past info still compressed into a single hidden state at each step → bottleneck for very long contexts
- More parameters than vanilla RNN; slower per step

The compressed-bottleneck critique motivates [[attention|attention mechanisms]] (Session 19), which let the model directly access specific parts of the past instead of routing everything through a single fixed-size vector.

## Exam framing

| Question | Answer |
|---|---|
| What's the role of the forget gate? | Element-wise multiplies $c_{t-1}$ — controls **what to erase** from long-term memory (Quiz IV Q3) |
| What's the role of the input gate? | Element-wise multiplies the candidate $\tilde{c}_t$ — controls **what new info to write** into the cell state |
| What's the role of the output gate? | Element-wise multiplies $\tanh(c_t)$ — controls **what to expose** as the hidden state |
| How is the cell state updated each step? | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ — combines a gated copy of the previous cell state with a gated candidate update |
| Why is LSTM better than a vanilla RNN at long-range dependencies? | The cell state has an **additive path** through time — gradients flow back without exponential decay |
| Why does LSTM still have limits? | Sequential computation (no parallelism); single fixed-size hidden state must encode all past info |

## Related

- [[gru|GRU]] — simpler 2-gate variant, no separate cell state
- [[recurrent-neural-network|Vanilla RNN]] — what LSTM extends
- [[vanishing-exploding-gradients]] — what gating addresses
- [[attention]] — what eventually replaces the bottleneck
