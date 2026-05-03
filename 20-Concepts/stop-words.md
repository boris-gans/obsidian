---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 03 - Fundamental Concepts.pdf
created: 2026-05-02
---

# Stop-words

Words that occur **extremely frequently** across texts but contribute little to certain tasks. Common examples: *the, is, on, with, at, and, of*. Removing them is often motivated by reducing noise and focusing on **content-bearing terms**.

## Trade-off

The decision reflects an assumption about **relevance**:

- "The cat sat **on the** mat."
- "The cat sat **under the** table."

Removing *the* and *on* reduces their influence on frequency-based models. But:

- For [[bag-of-words]] / [[tf-idf]] in topic-classification, stop-word removal helps focus.
- For **syntax analysis** or **question answering**, these words **may be crucial** — they carry grammatical structure.

Stop-word removal is therefore **not universally beneficial**. It's a preprocessing decision that should match the downstream task.

## Related concept

In [[information-retrieval-ranking|classical IR]], very common words are **downweighted by IDF** rather than removed outright — same intuition (rare = informative), gentler instrument. See [[tf-idf]].
