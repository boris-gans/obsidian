---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 05 - Python Modules.pdf
created: 2026-05-02
---

# NLP libraries (NLTK, spaCy, gensim, HuggingFace)

The four Python libraries used throughout the course. Each encodes a different theoretical commitment about what NLP is doing — using a library means **adopting its view** of what matters in language processing.

## Library map

| Library | What it is | Strength | Mock fit |
|---|---|---|---|
| **NLTK** | Classical, **linguistically motivated** view; exposes tokenization, tagging, parsing, corpora explicitly | Learning, experimentation | Quiz I Q35: "library most commonly used for **educational NLP experiments**" |
| **spaCy** | **Industrial NLP pipeline** with strong defaults; fast tokenize/tag/parse/NER using pretrained models | Production apps | Quiz I Q36: "best described as **an industrial NLP pipeline**" |
| **gensim** | Focused on **vector space models and semantic representations** (BoW, TF-IDF, LSA, LDA, Word2Vec at scale) | Distributional semantics, topic models | Quiz I Q37: "primarily associated with **word embeddings**" |
| **HuggingFace Transformers** | High-level interface to **pretrained models**; embed syntactic and semantic regularities implicitly | Modern downstream tasks | Quiz I Q38: "mainly provide **pretrained language models**" |

## Boundaries

- spaCy does **not** train transformers (Quiz I Q39). It uses pretrained ones; for training, you go to HuggingFace.
- NLTK exposes pieces; spaCy hides them inside an end-to-end pipeline. spaCy "gains efficiency at the cost of transparency".
- gensim minimizes syntax — its world is distributional / co-occurrence statistics.

## Common idioms

```python
# NLTK
import nltk
tokens = nltk.word_tokenize("NLP is fun")  # → ['NLP', 'is', 'fun']

# spaCy
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("NLP is fun")                    # full pipeline: tokens, POS, NER

# gensim
from gensim import corpora, models

# HuggingFace
from transformers import pipeline           # auto-loads model on first use
```

## Exam-ready facts

- `doc = nlp(text)` (spaCy) **processes text through an NLP pipeline** (Quiz I Q40)
- `word_tokenize('NLP is fun')` returns a **list of tokens** (Quiz I Q41)
- The bug `tokens = nltk.word_tokenize` (no parentheses) **doesn't tokenize anything** — the function is not called (Quiz I Q42)
- **Tokenization** must occur before counting word frequencies (Quiz I Q43)
- Modern libraries reduce manual coding because they **provide reusable pipelines** (Quiz I Q44) — they don't eliminate ambiguity, avoid computation, or guarantee understanding

## Why use a library at all

Each library encodes assumptions about language, representation, and processing pipelines. Understanding them means understanding the **theoretical compromises they embody**. An NLP library is not just a collection of functions — it defines representations, preprocessing steps, and default assumptions about language.
