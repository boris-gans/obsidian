---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 05 - Python Modules.pdf
created: 2026-05-02
---

# Abstract Meaning Representation (AMR)

A semantic representation that encodes meaning as a **graph of concepts and relations**, abstracting away from surface syntax. Designed to address the [[semantic-gap]]: sentences with different syntactic structure that mean the same thing should map to the same underlying representation.

## The core idea

Take two sentences:

- "The boy broke the window."
- "The window was broken by the boy."

Syntactically they differ — different parse trees, different constituent order. Semantically they are equivalent: there is an **agent** (boy), an **action** (break), and an **affected object** (window).

AMR encodes this directly:

```
(b / break-01
   :agent (boy / person)
   :patient (window / window))
```

Both surface realizations map to the same AMR graph, making **semantic equivalence explicit**.

## Trade-off

| Pros | Cons |
|---|---|
| Makes meaning explicit | Requires complex parsing and annotation |
| Captures equivalence across paraphrases | Hand-built schemas may not cover all phenomena |
| Useful for QA, MT evaluation | Does not scale as cleanly as learned embeddings |

## Where it sits

AMR is one symbolic answer to the [[semantic-gap]]. Modern neural models tackle the same problem implicitly via dense [[word-embeddings|embeddings]] and [[attention|attention]], without an explicit graph — different trade-offs (less interpretable, more scalable).
