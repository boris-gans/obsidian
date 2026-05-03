---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: quizzes/quiz IV - Model A - Solutions.pdf
  - course: NLP
    file: quizzes/quiz IV - Model B - Solutions.pdf
  - course: NLP
    file: 20_Text_Generation.ipynb
created: 2026-05-02
---

# Text generation: sampling strategies

Once a language model produces a probability distribution $p_t$ over the vocabulary at step $t$, we need a **decoding strategy** — a rule for selecting the next token. The choice trades off **quality, diversity, and computational cost**.

The blueprint flags this as **medium weight**: Quiz IV Q15, Q16, Q17 (and B variants) test temperature, beam search, and top-k. The Session 19 deck does **not** cover these explicitly — they're sourced from quiz coverage, not the slides.

## The four standard strategies

| Strategy | Rule | Tradeoff |
|---|---|---|
| **Greedy / Argmax** | Pick the most probable token: $\hat{y}_t = \arg\max_w p_t(w)$ | Deterministic; fast; can produce repetitive / low-quality text |
| **Beam search** | Maintain top-$k$ partial sequences ranked by joint probability | Higher quality than greedy; **more compute**; beam size 1 = greedy |
| **Temperature** | Divide logits by $T$ before softmax: $p \propto \exp(z / T)$ | $T \uparrow$ → flatter distribution, more diverse / random; $T \downarrow$ → sharper, more deterministic |
| **Top-k** | Keep only the $k$ most probable tokens, renormalize, sample | Restricts to plausible candidates; common $k$ = 40 or 50 |

## Temperature in detail (Quiz IV Q15)

Standard softmax: $p_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$.

Temperature-scaled: $p_i = \dfrac{e^{z_i / T}}{\sum_j e^{z_j / T}}$.

- **$T = 1$**: standard softmax
- **$T < 1$** (e.g. 0.5): **sharpens** the distribution — probabilities concentrate on the most likely token; **more deterministic, peaked on argmax**
- **$T > 1$** (e.g. 1.5): **flattens** the distribution — probabilities spread out; **more diverse, more random**

> "Temperature ↑ flattens distribution (more diverse); ↓ sharpens (more deterministic, peaked on argmax)." (blueprint, "what the formula sheet does NOT provide")

In the limit $T \to 0$, sampling becomes deterministic argmax (greedy). In the limit $T \to \infty$, sampling becomes uniform.

## Beam search in detail (Quiz IV Q16)

Greedy commits to one token at every step — this can lead to suboptimal global sequences (the highest-probability *first* token may have a poor continuation). **Beam search** explores **$k$ partial sequences in parallel**:

1. At each step, expand each beam by every possible next token
2. Keep only the top-$k$ extended sequences ranked by joint probability $\prod_t p_t$ (or its log)
3. Stop when all beams produce EOS

Trade-offs:
- **Beam size = 1**: equivalent to **greedy decoding**
- **Larger beam**: better global quality, but **higher compute** (linear in $k$)
- Standard beam sizes: 4–8 for translation; up to 50 for summarization

## Top-k sampling in detail (Quiz IV Q17)

Pure stochastic sampling from $p_t$ over the entire vocabulary can pick very low-probability tokens, producing nonsense. **Top-k** restricts to the $k$ highest-probability tokens:

1. Find the top-$k$ tokens by probability
2. Set all other probabilities to 0
3. Renormalize the kept probabilities
4. Sample from this truncated distribution

Common $k = 40$ or $50$. Filters out the long tail of unlikely tokens while keeping diversity.

A close cousin is **top-p (nucleus) sampling** — keep the smallest set of tokens whose cumulative probability exceeds $p$ (e.g. 0.9). [not in source — supplementary]

## Choosing a strategy

| Use case | Recommended strategy |
|---|---|
| Translation | **Beam search** (size 4–8) — quality matters, output is short |
| Code completion | **Greedy** or low-temperature — determinism preferred |
| Creative writing / chat | **Sampling with temperature 0.7–1.0** + top-k or top-p — diversity matters |
| Summarization | **Beam search** with large beam (10–20) |

## Exam framing

| Question | Answer |
|---|---|
| What does increasing temperature do to a softmax? | **Flattens** the distribution — output becomes more diverse / random (Quiz IV Q15) |
| What does temperature → 0 reduce sampling to? | **Greedy / argmax** — always pick the most probable token |
| What's the relationship between beam search and greedy? | **Greedy = beam search with $k = 1$**. Larger beams trade compute for quality (Quiz IV Q16) |
| What does top-$k$ sampling do? | Restricts sampling to the $k$ most probable tokens, renormalizes, then samples — filters out the long tail (Quiz IV Q17) |
| Standard beam sizes for translation? | **4–8** beams typically |

## Related

- [[softmax]] — produces the distribution that sampling strategies operate on
- [[transformer]] / GPT / decoder-only models — the typical generators
- [[extractive-question-answering]] — uses argmax (deterministic), no sampling
- [[recurrent-neural-network]] — also uses these decoding strategies for next-token prediction

## Canonical HuggingFace generation skeleton

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
model = AutoModelForCausalLM.from_pretrained("gpt2-xl").to("cuda")

input_ids = tokenizer("Madrid is", return_tensors="pt")["input_ids"].to("cuda")

# Greedy
out_greedy = model.generate(input_ids, max_length=64, do_sample=False)

# Beam search (size 5)
out_beam = model.generate(input_ids, max_length=64, num_beams=5, do_sample=False)

# Sampling with temperature
out_temp = model.generate(input_ids, max_length=64, do_sample=True,
                          temperature=0.7, top_k=0)

# Top-k sampling
out_topk = model.generate(input_ids, max_length=64, do_sample=True,
                          top_k=10)

# Nucleus (top-p) sampling
out_topp = model.generate(input_ids, max_length=64, do_sample=True,
                          top_p=0.95)

print(tokenizer.decode(out_beam[0]))
```

Reference: `[Generation strategies (cells 6–37)](30-Sources/NLP/notebooks/20_Text_Generation.ipynb)`.

**Exam-critical kwargs of `model.generate()`:**

| Kwarg | Effect |
|---|---|
| `do_sample=False` | Greedy / beam (deterministic given input) |
| `do_sample=True` | Stochastic sampling |
| `num_beams=k` | Beam search with $k$ beams; $k=1$ ≡ greedy |
| `temperature=T` | Divides logits by $T$ before softmax. `T < 1` sharpens; `T > 1` flattens |
| `top_k=k` | Keep only the $k$ highest-probability tokens |
| `top_p=p` | Nucleus sampling — keep smallest set whose cumulative probability ≥ $p$ |
| `max_length` | Total output length (input + generated) |
| `max_new_tokens` | Number of new tokens to generate (preferred over max_length) |
