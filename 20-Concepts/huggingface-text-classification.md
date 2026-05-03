---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: 19_Text_Classification_and_Hugging_Face.ipynb
created: 2026-05-02
---

# HuggingFace text classification

The canonical HuggingFace pipeline for **multi-class text classification** using `AutoModelForSequenceClassification`. The blueprint flags this directly: **mock Q18 tests `num_labels`**, mock Q20 tests `padding=True`/`truncation=True`, and `[19_Text_Classification_and_Hugging_Face.ipynb](30-Sources/NLP/notebooks/19_Text_Classification_and_Hugging_Face.ipynb)` is the source.

## Canonical skeleton (memorize cold)

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from datasets import load_dataset

# 1. Load dataset and inspect labels
emotions = load_dataset("emotion")
num_labels = 6   # ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# 2. Tokenize
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

# 3. Model with classification head
device = "cuda" if torch.cuda.is_available() else "cpu"
model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
         .to(device))

# 4. Trainer
training_args = TrainingArguments(
    output_dir="emotion-clf",
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"],
)
trainer.train()
```

## Why every line matters

| Line | Why |
|---|---|
| `AutoTokenizer.from_pretrained(...)` | Loads the same tokenizer the model was pretrained with — token IDs must match the model's vocab |
| `padding=True` | Pads shorter sequences to the longest in the batch (mock Q20) |
| `truncation=True` | Truncates sequences longer than the model's max context (BERT: 512 tokens) |
| `AutoModelForSequenceClassification` | Adds a **classification head** on top of the base model |
| `num_labels=num_labels` | **Number of output classes** (mock Q18 tests this exact kwarg) |
| `.to(device)` | Move to GPU if available |
| `learning_rate=2e-5` | Standard fine-tuning LR for BERT-family models |

## Exam framing

| Question | Answer |
|---|---|
| What does `num_labels` specify? | The **number of output classes** for the classification head (mock Q18) |
| What do `padding=True` and `truncation=True` do? | Pad short sequences and truncate long ones to a uniform length within the batch (mock Q20) |
| What's the input pipeline? | Tokenizer → `input_ids` + `attention_mask` → model → `logits` over `num_labels` classes |
| Which class adds the classification head? | `AutoModelForSequenceClassification` (vs `AutoModel` which has no head; vs `AutoModelForQuestionAnswering` which has a QA head) |

## Related

- [[transformer]] / BERT — the underlying architecture
- [[extractive-question-answering]] — sibling pipeline using `AutoModelForQuestionAnswering`
- [[softmax]] — converts the model's logits to class probabilities
- [[cross-entropy]] — the training loss (Trainer applies it under the hood)
