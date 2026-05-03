---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: 21_Text_Summarization.ipynb
created: 2026-05-02
---

# HuggingFace summarization

The canonical HuggingFace pipeline for **abstractive text summarization** using `AutoModelForSeq2SeqLM` (encoder-decoder transformers like BART, T5, Pegasus). Distinguished from extractive summarization — which selects existing sentences — by **generating new text** that compresses the input.

The blueprint flags this as **medium weight** (no specific exam question, but listed under sequence-to-sequence patterns the prof discusses). The notebook compares BART, T5, Pegasus, and GPT-2 outputs against a baseline.

## Canonical encoder-decoder skeleton (BART)

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

inputs = tokenizer(
    article_text,
    return_tensors="pt",
    truncation=True,
    max_length=1024,        # BART's context limit
).to(device)

summary_ids = model.generate(
    **inputs,
    max_new_tokens=120,
    num_beams=4,             # beam search for quality
    length_penalty=2.0,      # encourage longer outputs
    early_stopping=True,
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

## GPT-2 "TL;DR" trick (decoder-only)

GPT-2 is decoder-only — but you can elicit summaries by **prompt engineering**: append "TL;DR" at the end of the input and let the model continue.

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="gpt2-xl", device=device)
gpt2_query = article_text + "\nTL;DR:\n"
out = pipe(gpt2_query, max_length=512)
summary = out[0]["generated_text"][len(gpt2_query):]
```

This works because GPT-2's training data contains many "TL;DR" summaries, so the prompt elicits a similar pattern. **Statistical, not semantic** — exactly what [[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=18|slide 18]] of Session 19 warns about.

## ROUGE evaluation skeleton

```python
import evaluate
rouge = evaluate.load("rouge")

reference = "Mentally ill inmates are housed on the forgotten floor of Miami-Dade jail..."
predictions = [bart_summary, t5_summary, pegasus_summary]

for pred in predictions:
    rouge.add(prediction=pred, reference=reference)
    score = rouge.compute()
    print(score)  # {'rouge1': ..., 'rouge2': ..., 'rougeL': ..., 'rougeLsum': ...}
```

ROUGE metrics:
- **ROUGE-1**: unigram overlap → measures content coverage
- **ROUGE-2**: bigram overlap → measures local fluency / phrase matching
- **ROUGE-L**: longest common subsequence → structural similarity
- **ROUGE-Lsum**: summary-level LCS → coherence across multiple sentences

Higher = better. BART tends to lead; GPT-2 (no fine-tuning for summarization) typically lowest on ROUGE-2.

Reference: `[BART/T5/Pegasus summarization (cells 27–39)](30-Sources/NLP/notebooks/21_Text_Summarization.ipynb)`.

## Exam framing

| Question | Answer |
|---|---|
| Which model class is used for summarization? | **`AutoModelForSeq2SeqLM`** — encoder-decoder transformers (BART, T5, Pegasus) |
| Why is BART well-suited for summarization? | Encoder-decoder architecture: bidirectional encoder reads the full input; autoregressive decoder generates the summary ([[30-Sources/NLP/pdf/Session 19 - Transformers-1.pdf#page=19|slide 19]]) |
| What's the prompt trick that lets GPT-2 summarize without fine-tuning? | Append `"\nTL;DR:\n"` to the input and let the model continue — relies on training-data patterns |
| What does `length_penalty=2.0` encourage? | Longer outputs (penalty > 1 favors longer beams; < 1 favors shorter) |
| What does ROUGE-2 measure? | **Bigram overlap** between candidate and reference — captures phrase-level fluency |

## Related

- [[transformer]] — encoder-decoder underlies BART/T5
- [[encoder-decoder]] — the architectural pattern
- [[text-generation-sampling]] — beam search is the standard decoding strategy here
- [[extractive-question-answering]] — sibling task with different output structure (span vs new text)
