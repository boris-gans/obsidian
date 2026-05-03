---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 05 - Python Modules.pdf
created: 2026-05-02
---

# Context-free grammar

A core construct of **formal language theory** (Noam Chomsky). The idea: a finite set of grammatical **rewriting rules** generates an **infinite number of sentences** by recursively expanding a start symbol into terminals.

## Mechanics

- **Start symbol** S, plus a set of non-terminals (NP, VP, PP, Det, N, V, P, …) and terminals (concrete words)
- **Rewriting rules** of the form `S → NP VP`, `NP → Det N`, `VP → V NP PP`, `PP → P NP`, etc.
- Apply rules recursively until only terminals remain — yields a **parse tree** describing hierarchical structure

Example tree for "The man saw the car in the garage":
```
S → NP VP
NP → Det N → "The man"
VP → V NP PP
NP → Det N → "the car"
PP → P NP → "in the garage"
```

## Strength: structural well-formedness

CFGs cleanly capture **recursion** and **nested dependencies** that natural language exhibits — sentence-within-sentence structures, complex noun phrases, prepositional attachment.

## Limitation

A CFG only checks whether a sentence **can be derived from the grammar** — its primary concern is **structural well-formedness, not semantic interpretation or empirical usage patterns**. Two grammatically valid sentences may differ wildly in meaning; two sentences with very different structure may mean the same thing. This mismatch is the [[semantic-gap]].

CFGs also struggle with [[language-ambiguity|language's intrinsic ambiguity]] — many sentences have multiple valid parses, and structure alone cannot pick the right one.

## Where CFGs sit historically

CFGs were the cornerstone of [[early-symbolic-nlp|early symbolic NLP]]. They lost ground to statistical methods because handcrafted grammars **failed outside restricted domains** (Quiz I Q12). Modern NLP has largely replaced explicit parse trees with implicit structure learned by large neural models — but the conceptual machinery (constituents, dependency relations, hierarchical structure) remains foundational vocabulary in NLP.
