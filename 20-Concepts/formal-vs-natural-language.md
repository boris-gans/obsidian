---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 01 - Introduction.pdf
created: 2026-05-02
---

# Formal vs natural language

**Formal languages** (e.g. Python, predicate logic, regex) are designed with **fixed syntax and semantics**. Each well-formed expression has exactly one meaning. The grammar is small, complete, and machine-checkable.

**Natural languages** (e.g. English, Spanish) **were not designed to be unambiguous or machine-readable**. They evolve organically, contain irregularities, and depend heavily on context.

The distinguishing property is **context dependence** — natural-language meaning depends on situation, prior knowledge, and cultural background; formal-language meaning does not.

| Property | Formal | Natural |
|---|---|---|
| Syntax | Fixed, complete | Irregular, evolving |
| Semantics | Unambiguous | Context-dependent |
| Designed for | Machines | Humans |
| Meaning over time | Stable | Drifts |

## Why this matters for NLP

Because natural language is ambiguous and context-dependent, NLP systems must cope with **uncertainty, multiple interpretations, and incomplete information** — see [[language-ambiguity]]. Early symbolic NLP assumed meaning could be **fully formalized** (Quiz I Q15); this assumption underlies projects like Ars Magna and the rule-based systems that "failed outside restricted domains" (Quiz I Q12).

Common exam framing: "Which property most clearly distinguishes natural language from formal languages?" → **context dependence** (Quiz I Q3).
