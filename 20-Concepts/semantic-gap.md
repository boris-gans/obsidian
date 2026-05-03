---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 05 - Python Modules.pdf
created: 2026-05-02
---

# The semantic gap

The mismatch between **syntactic structure** and **meaning** in natural language. Recovering syntactic structure does not automatically yield meaning:

- Sentences with **very similar grammatical structure** may differ in interpretation
- Sentences with **different surface forms** may express the same idea (active vs passive, paraphrase, synonymy)

The gap became apparent as early syntactic parsers were applied to real language — well-formed parse trees did not, on their own, tell you what a sentence *meant*.

## Two responses to the gap

1. **Formal semantic frameworks** — compositional interpretations built using logical formalisms; syntax + a semantics-derivation rule
2. **Direct meaning representations** that abstract away from syntax altogether — e.g. [[abstract-meaning-representation|AMR]]

## The modern (implicit) response

Modern neural NLP doesn't address the semantic gap explicitly — instead, it **learns** distributed representations from data that capture syntactic and semantic regularities together. Whether this *solves* the gap or just hides it is one of the field's persistent questions, returning in [[hallucinated-responses|hallucinations]] and Session 24's discussion of challenges.

## Why it matters

The semantic gap is the formal analogue of [[eliza|ELIZA]]'s lesson: surface form (whether parse-tree well-formedness or fluent generation) does not guarantee understanding. The gap is what justifies investing in dense [[word-embeddings|embeddings]], [[attention|attention mechanisms]], and other approaches that bind structure and meaning.
