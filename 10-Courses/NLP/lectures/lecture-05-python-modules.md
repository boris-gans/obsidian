---
tags: [lecture]
course: NLP
source: Session 05 - Python Modules.pdf
lecture: 05
slides: "[[30-Sources/NLP/pdf/Session 05 - Python Modules.pdf|Open slides ↗]]"
created: 2026-05-02
---

# Lecture 05 — Python Modules

## Overview

Bridges theory and practice. The first half is conceptual: Chomsky's symbolic perspective on language as a structured system governed by explicit rules, context-free grammars, the **semantic gap** between syntax and meaning, and Abstract Meaning Representation (AMR). The second half tours the four NLP libraries used throughout the course: **NLTK** (educational), **spaCy** (industrial pipeline), **gensim** (vector-space / topic models), **HuggingFace Transformers** (pretrained models).

Key conceptual contrast: **statistical vs symbolic** views of language — the same dichotomy from Session 02 (rules vs behaviour), now seen through the choice of library.

## Key concepts

- [[context-free-grammar]] — finite rules generate infinite sentences; parse trees; structural well-formedness
- [[semantic-gap]] — syntactic structure does not yield meaning; sentences with similar structure differ in meaning, and vice versa
- [[abstract-meaning-representation]] (AMR) — graph of concepts and relations; semantic equivalence even with different surface syntax
- [[nlp-libraries]] — NLTK / spaCy / gensim / HuggingFace; each encodes a different theoretical commitment

## Equations

None — this is a conceptual / tooling deck.

## Diagrams

```mermaid
flowchart TD
    L[Natural language] --> A{Approach}
    A -->|statistical| S["Observable data;<br/>infer from frequency<br/>and co-occurrence"]
    A -->|symbolic| Y["Structured system<br/>governed by explicit rules<br/>(Chomsky)"]
    S --> SE["NLTK / gensim<br/>(corpora, vectors)"]
    Y --> SY["Parsers / context-free<br/>grammars / AMR"]
    SE --> M[Modern NLP:<br/>both perspectives,<br/>statistical dominates]
    SY --> M
```
*Two ways of thinking about language map to different libraries (slide 86).*

```mermaid
flowchart TD
    NLTK[NLTK<br/>educational, classical view] -->|expose tokenization, tagging, parsing, corpora| Use1[Learning, experimentation]
    SP[spaCy<br/>industrial pipeline] -->|fast tokenize/tag/parse/NER + pretrained| Use2[Production apps]
    GS[gensim<br/>vector-space + topic models] -->|BoW, TF-IDF, LSA, LDA, Word2Vec at scale| Use3[Distributional semantics]
    HF[HuggingFace Transformers<br/>pretrained models] -->|tokenizer + AutoModel*| Use4[Modern downstream tasks]
```
*Library map (slides 93–96).*

## Open questions

- The semantic gap motivated AMR. Modern transformers learn distributed semantic representations end-to-end without explicit AMR — but does that solve the gap or just hide it? Returns in Session 19 / 24.
