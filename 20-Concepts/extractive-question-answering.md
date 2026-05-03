---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 19 - Transformers-1.pdf
created: 2026-05-02
---

# Extractive Question Answering

**Extractive QA** is the task where the model selects an **answer span** from a context passage that answers a given question. Unlike generative QA (which writes a free-form answer), extractive QA outputs **start and end positions** within the input — the answer is always a contiguous chunk of the original text.

The blueprint flags this as **very high weight**: mock Q19 (extractive QA definition) and **mock Q30 = Code 2 fill-in-blanks (10 pts)** on the HuggingFace QA pipeline. The exam tests whether the student can **reproduce the canonical 5–15 line skeleton from memory**, including the specific HF library tokens.

## The task

Given:
- A **question**: e.g. "Where was the model trained?"
- A **context** (passage): e.g. "The BERT model was trained on Wikipedia in 2018."

Predict:
- **Answer span**: the start and end **token indices** in the context (or a tokenized concatenation thereof)
- Decoded back to text: e.g. "Wikipedia"

## Why BERT-style (encoder-only) is the natural fit

BERT ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=18|slide 18]], [[transformer]]) is designed for **representation learning** with bidirectional attention — every token incorporates context from the entire sequence. For QA:
1. The question and context are concatenated with separator tokens
2. Each token's contextual embedding is projected to **two scalars**: a `start_logit` and an `end_logit`
3. The predicted answer span is `argmax(start_logits)` to `argmax(end_logits)`

This task design lets BERT exploit its bidirectional understanding without needing autoregressive generation.

## Input format (formula sheet)

The transformer input for extractive QA is:
```
[CLS] question [SEP] context [SEP]
```
- `[CLS]` and `[SEP]` are special tokens
- The model returns:
  - `input_ids` — token ID sequence
  - `attention_mask` — 1 for real tokens, 0 for padding
  - `start_logits`, `end_logits` — per-token scores for span start/end

## Canonical HuggingFace pipeline (mock Q30 fill-in-blanks)

This is the exact skeleton the exam tests. **Memorize these tokens verbatim** — every HF literal is a potential blank:

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_ckpt = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)

inputs = tokenizer(question, context, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits) + 1

answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx])
```

Every red-flagged literal:
- `model_ckpt = "deepset/minilm-uncased-squad2"` — a MiniLM (~66M params) fine-tuned on SQuAD2; this is the exact checkpoint used in notebook 22, NOT vanilla `bert-base-uncased`
- `AutoTokenizer.from_pretrained(model_ckpt)` — the tokenizer load
- `AutoModelForQuestionAnswering.from_pretrained(model_ckpt)` — the model load
- `return_tensors="pt"` — return PyTorch tensors (not numpy / TF)
- `with torch.no_grad():` — disable gradient tracking for inference
- `outputs.start_logits` and `outputs.end_logits` — per-token span-position scores
- `torch.argmax(...)` — pick the highest-scoring position
- `+ 1` on the end index — Python slice exclusive at the end
- `tokenizer.decode(...)` — convert token IDs back to text

## Why each line matters

| Line | Why |
|---|---|
| `AutoTokenizer.from_pretrained(...)` | Loads the same tokenizer the pretrained model was trained with — token IDs must match |
| `AutoModelForQuestionAnswering` | Adds the QA head (start/end logit projections) on top of the base model |
| `tokenizer(question, context, ...)` | Builds `[CLS] question [SEP] context [SEP]` automatically |
| `return_tensors="pt"` | Makes outputs PyTorch tensors so they feed directly into `model(**inputs)` |
| `with torch.no_grad():` | Skips building the autograd graph — faster + less memory at inference |
| `outputs.start_logits` / `end_logits` | The per-token scores predicted by the QA head |
| `+ 1` on end index | Python's slice end is exclusive; the predicted end token must be **included** in the answer |
| `tokenizer.decode(...)` | Reverses tokenization to produce human-readable text |

## Common pitfalls

- Forgetting `return_tensors="pt"` → outputs are Python lists, not tensors
- Forgetting `+ 1` on end index → answer truncated by one token
- Forgetting `with torch.no_grad():` → still runs but wastes memory and slows down
- Mismatched tokenizer / model checkpoints → token IDs don't match what the model expects

## Exam framing

| Question | Answer |
|---|---|
| What is extractive QA? | Identifying a **span of tokens within the context** that answers the question (mock Q19) |
| What two outputs does the QA head produce? | **`start_logits`** and **`end_logits`** — per-token scores for span start and end (mock Q30) |
| How is the answer text recovered? | $\mathrm{argmax}$ of start and end logits give token indices; slice `input_ids` and decode (mock Q30) |
| Why `with torch.no_grad():` for inference? | Disables gradient tracking — saves memory and speeds up the forward pass |
| What does `return_tensors="pt"` do? | Returns inputs as PyTorch tensors |
| What's the input format for BERT QA? | `[CLS] question [SEP] context [SEP]` — concatenated and tokenized together |

## Related

- [[transformer]] / BERT — the model family; BERT-style encoder-only is ideal for extractive QA
- [[softmax]] — formally not applied to start/end logits in the canonical pipeline (argmax suffices)
- [[huggingface-text-classification]] — the sibling pipeline using `AutoModelForSequenceClassification`
- [[text-generation-sampling]] — generative QA (free-form output) uses sampling instead of argmax
