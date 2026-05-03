---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 19 - Transformers-1.pdf
created: 2026-05-02
---

# Cross-attention

In a [[transformer]] **encoder-decoder** model, **cross-attention** is the layer where the decoder attends to the **encoder's output**: queries come from the decoder; keys and values come from the encoder ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=15|slides 15]], 323, 324). It's how the decoder "looks at" the source while generating each target token — the transformer-era successor to the Bahdanau attention in [[encoder-decoder|RNN encoder-decoder]] systems.

The blueprint flags this as **medium weight**: Quiz IV Q12, Q20 (and B variants) test cross-attention placement and its role in encoder-decoder architectures.

## The setup ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=15|slide 15]])

Inside a transformer **decoder block**, three sublayers run in sequence:
1. **Masked self-attention** — Q/K/V all from the decoder; causal mask
2. **Cross-attention** — **Q from decoder, K/V from encoder output**
3. **Feed-forward** layer

Cross-attention is what gives the decoder access to the source — without it, the decoder would only see its own past tokens and have no way to condition generation on the input.

## Where Q, K, V come from

| Layer | Q | K | V |
|---|---|---|---|
| Encoder self-attention | Encoder | Encoder | Encoder |
| Decoder masked self-attention | Decoder | Decoder | Decoder |
| **Cross-attention** | **Decoder** | **Encoder** | **Encoder** |

The asymmetry — Q from one source, K/V from another — is exactly what makes the layer "cross". Each decoder position formulates a query, looks up encoder positions, and retrieves a weighted blend of encoder values.

## How it replaces the fixed-context bottleneck

In the basic [[encoder-decoder]] (Session 18), the entire source was compressed into a **single fixed-length context vector**. Cross-attention removes that bottleneck:

> "The 'context' is a sequence of hidden states produced by attention layers, which implies that there is **no global fixed-length summary of the input**." ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=16|slide 16]])

The decoder accesses the **full sequence of encoder outputs** at every step, with attention weights selecting which positions matter for the current target token.

## Computation ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=15|slide 15]])

Same scaled dot-product attention as everywhere else, but with mismatched Q vs K/V sources:
$$\mathrm{score}_{ij} = q_i^{\mathrm{dec}} \cdot k_j^{\mathrm{enc}}$$
$$\beta_{ij} = \mathrm{softmax}_j(\mathrm{score}_{ij})$$
$$c_i^{\mathrm{cross}} = \sum_j \beta_{ij}\, v_j^{\mathrm{enc}}$$

The deck calls these scores $\beta_{ij}$ to distinguish from the masked self-attention scores $\alpha_{ij}$ in the same decoder layer ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=15|slide 15]]).

## Where it appears in transformer family

- **Encoder-decoder transformers** (BART, T5, original transformer for translation): cross-attention in every decoder block
- **Decoder-only transformers** (GPT family): **no cross-attention** — there's no encoder, so generation is conditioned only on the prompt + previous tokens via masked self-attention
- **Encoder-only transformers** (BERT): **no cross-attention** — there's no decoder

## Exam framing

| Question | Answer |
|---|---|
| What is cross-attention? | A decoder layer where queries come from the decoder and keys/values come from the encoder output ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=15|slide 15]], Quiz IV Q12) |
| Why is cross-attention important? | Lets the decoder access the **full sequence of encoder hidden states** at every generation step — no fixed-context bottleneck (Quiz IV Q20) |
| Where does cross-attention sit in a transformer decoder block? | Between the **masked self-attention** sublayer and the **feed-forward** sublayer |
| Which transformer family models use cross-attention? | Encoder-decoder ones — **BART, T5, the original transformer**. Not BERT (encoder-only), not GPT (decoder-only). |
| What's the source of $K, V$ in cross-attention? | The **encoder's output** (its top-layer hidden states) |

## Related

- [[self-attention]] — the other attention type in the decoder; same formula, different Q/K/V sources
- [[attention]] — the parent concept; cross-attention is the natural transformer evolution of Bahdanau attention
- [[encoder-decoder]] — the architecture cross-attention enables
- [[causal-masking]] — only on decoder self-attention; cross-attention has no causal mask
- [[transformer]] — the host architecture
