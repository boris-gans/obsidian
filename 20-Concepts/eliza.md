---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 02 - History.pdf
created: 2026-05-02
---

# ELIZA

A 1966 conversational program developed by Joseph Weizenbaum, simulating a **Rogerian psychotherapist** by responding to user input with carefully constructed questions and reflections. ELIZA was the first large-scale demonstration that **conversational behaviour alone can create a strong impression of understanding**, even when no understanding is present.

## How ELIZA worked

At a technical level: **pattern matching and substitution**.

- Detect specific **keywords** in the user's input
- Apply a corresponding **predefined transformation rule** to construct a reply
- No memory of previous interactions beyond the immediate input
- No representation of meaning, beliefs, or goals

What made ELIZA effective was not intelligence but **careful exploitation of conversational conventions**: by prompting users to elaborate and reflect on their own statements ("Tell me more about X"), it shifted the conversational burden onto the human.

## The ELIZA effect

The widespread tendency of users to **attribute understanding and intention** to ELIZA. This phenomenon highlighted a critical danger: humans are predisposed to **infer intelligence from fluent linguistic behaviour**.

The lesson: fluency alone is an unreliable indicator of intelligence or comprehension. Compare with the modern echo: **hallucinated responses** in LLMs (Quiz I Q23) are the same problem at a different scale.

## Exam-ready facts

| Fact | Source |
|---|---|
| ELIZA simulated a **psychotherapist** | Quiz I Q16 |
| ELIZA generated responses primarily using **keyword matching and substitution** | Quiz I Q17 |
| The ELIZA effect = **overestimation of system understanding** | Quiz I Q18 |
| ELIZA demonstrated that **simple mechanisms can appear meaningful** | Quiz I Q19 |
| Main lesson: **behaviour can mask lack of understanding** | Quiz I Q20 |

## Why it still matters

ELIZA forced researchers to confront the **limitations of surface-level language processing** and to reconsider how NLP systems should be evaluated. The same diagnostic question — does this system understand, or only behave as if it does? — applies to modern LLMs. See [[turing-test]] for the parallel theoretical move and [[nlp-understanding-vs-generation]] for the broader frame.
