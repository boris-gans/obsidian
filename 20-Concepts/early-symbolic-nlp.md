---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 02 - History.pdf
created: 2026-05-02
---

# Early symbolic NLP

Early computational linguistics treated language as **a symbolic system governed by rules** (Quiz I Q11). Influenced by formal linguistics, these models assumed that understanding language meant recovering precise syntactic and semantic structures from sentences. Implementation: **handcrafted grammars, dictionaries, and transformation rules**, processing language as sequences of discrete symbols.

## Lineage of the formalization dream

- **Ramón Llull's *Ars Magna*** (13th century) — design a perfectly rational and unambiguous symbol system
- **John Wilkins's *Philosophical Language*** (17th century)
- **Gottfried Leibniz's *Characteristica Universalis*** (mid/late 17th century)

All shared the assumption that meaning could be **fully formalized** (Quiz I Q15) — that ambiguity could be eliminated through symbolic design (Quiz I Q13: Ars Magna is the canonical historical example).

## Why it didn't scale

Early symbolic / rule-based NLP **failed outside restricted domains** (Quiz I Q12). Two structural reasons:

- Constructing comprehensive rule sets required **significant human effort**
- Even small deviations from expected input caused failures

These limitations exposed [[language-ambiguity|ambiguity]] as **unavoidable**, not a fixable defect. Handcrafted rules do not scale; evaluation is nontrivial. These lessons motivated the later shift toward **probabilistic and data-driven approaches**.

## Why it still matters

Studying this history is valuable because it **reveals persistent conceptual challenges** (Quiz I Q14) — the questions about meaning, understanding, and ambiguity remain open even though the tools have changed dramatically. Modern neural NLP inherits both the strengths (the ambition to capture structure) and weaknesses (handling figurative / out-of-domain language) of its predecessors.
