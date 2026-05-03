---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 11 - Sentiment Analysis.pdf
  - course: NLP
    file: 06_Sentiment_Analysis.ipynb
created: 2026-05-02
---

# Sentiment analysis

The task of assigning a polarity (positive / negative / neutral) or a numeric score to a piece of text — document, sentence, or aspect. One of the canonical applied tasks in NLP and a common entry point for [[text-classification]].

## Two scoring approaches

### Lexical / dictionary-based

Average per-word polarity scores from a **sentiment lexicon** (AFINN, VADER, custom polar lexicons):
$$S(d) = \frac{1}{n} \sum_{i=1}^n \text{score}(w_i)$$

The Quiz III header gives this exact formula. It's a **simple, interpretable baseline** that ignores context, negation, and word order.

### Learned classifier

Apply [[naive-bayes|Naïve Bayes]] or [[logistic-regression]] to a [[bag-of-words|BoW]] / [[tf-idf|TF-IDF]] representation, with binary or multi-class output (positive / negative; or 1–5 stars for finer-grained sentiment). The model learns **which words signal which polarity**:

- LR coefficient $w_i > 0$ for word $i$ → contributes to positive polarity
- LR coefficient $w_i < 0$ → contributes to negative polarity
- $|w_i|$ large → strong evidence either way

## Time-series / discourse interpretation

When applied to a **sequence of documents over time**, sentiment values are not treated as static labels — they become a **dynamical signal evolving across observations** (Quiz III Q2 answer).

Common framings:

| Domain | What sentiment-over-time reveals |
|---|---|
| **Financial news** | Discourse dynamics that may correlate with market signals (Quiz III Q16) |
| **Discourse analysis** | **Narrative shifts** in storytelling / reporting (Quiz III Q16 Model B) |
| **Brand / customer analytics** | Trend in customer perception |
| **Political analysis** | Mood / framing changes over a campaign or news cycle |

Quiz III Q2 frames this directly: when analyzing sentiment over time, the right interpretation is **dynamical signal**, not "static lexical property attached to tokens", and not a topic distribution.

## Limitations

- **Ignores word order and negation** in the lexical version — "not good" averages to the same as "good" without negation handling
- **Sarcasm and irony** routinely fool surface-level sentiment models — same problem as [[language-ambiguity|language ambiguity]] elsewhere in NLP
- **Domain shift** — sentiment lexicons don't transfer across domains (financial vs movie reviews use different polar vocabulary)
- **BoW limitations apply** — see [[bag-of-words]] for the broader picture

These motivate moving from BoW + LR to dense [[word-embeddings]] and contextual transformer-based sentiment classifiers (Session 19+).

## Canonical lexical-scoring skeleton (AFINN)

```python
from afinn import Afinn
import spacy

nlp = spacy.load("en_core_web_sm")
afinn = Afinn()

def score_sentence(sentence: str) -> float:
    return afinn.score(sentence)

def split_sentences(text: str):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

# Per-sentence scores → time series
sentences = split_sentences(corpus)
scores = [score_sentence(s) for s in sentences]
```

Reference: `[AFINN per-sentence scoring (cells 6–10)](30-Sources/NLP/notebooks/06_Sentiment_Analysis.ipynb)`.

Each $S(d_t)$ is plotted as a function of $t$ — the **time-series interpretation** the prof emphasizes (Quiz III Q2). The exam-relevant insight: lexical scoring is the **simplest baseline** and the time-series framing **changes the interpretation** of the result, not the formula.
