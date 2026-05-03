---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 01 - Introduction.pdf
created: 2026-05-02
---

# NLP understanding vs generation tasks

NLP tasks split into two families based on output type.

## Understanding tasks

Take language as input, produce a label / structure / extract:

- **Sentiment analysis** — polarity / score for a document
- **Named entity recognition (NER)** — identify entity spans + types ([[named-entity-recognition]])
- **Question answering** — locate an answer in a passage ([[extractive-question-answering]])
- **Information extraction** — pull structured facts from unstructured text

## Generation tasks

Produce language as output:

- **Machine translation** — text in one language → text in another
- **Text completion** — continue a prompt
- **Summarization** — condense a longer input
- **Dialogue systems** — multi-turn conversational response

## Critical distinction (slide 12)

> NLP does not imply true human-level understanding. Most systems operate by detecting patterns in data rather than by reasoning about meaning in a human sense.

This is the recurring theme of the course. A system can produce fluent generation output without "understanding" — see [[eliza]], the [[turing-test|Turing imitation test]], and the modern echo in **hallucinated responses** (Quiz I Q23). The exam-ready phrasing: "behaviour can mask lack of understanding" (Quiz I Q20).

## Common exam fodder

- "Which task is an example of natural language generation?" → Machine translation (Quiz I Q6). Sentiment / NER / IR are *understanding* tasks.
- "Which is a core NLP task?" → Named entity recognition (Quiz I Q5). Numerical optimization, image segmentation, graph traversal are not.
