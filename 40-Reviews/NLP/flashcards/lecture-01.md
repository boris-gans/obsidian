---
tags: [flashcards, NLP]
---

# Lecture 01 — Introduction (flashcards)

What is Natural Language Processing?
?
An interdisciplinary field at the intersection of computer science, linguistics, mathematics, and AI whose goal is to design computational systems that **analyze, interpret, and generate** natural language.

What three capabilities define the goal of NLP?
?
**Analyze** (extract structure), **interpret** (assign meaning), **generate** (produce fluent output).

What property most clearly distinguishes natural language from formal languages?
?
**Context dependence.** Natural-language meaning depends on situation, prior knowledge, and culture; formal languages have fixed syntax and semantics.

Why is ambiguity intrinsic to natural language?
?
Because words and structures **depend on context** — the same surface form can carry different meanings in different situations.

Why is ambiguity hard to eliminate computationally?
?
Because meaning depends on **context and world knowledge that is often implicit** — adding more grammar rules cannot capture the missing background.

What are the four sources of difficulty in NLP (Session 01, slide 14)?
?
Ambiguity, implicit information, figurative language (irony / metaphor / sarcasm / idioms), and domain sensitivity.

What is the central distinction when evaluating NLP systems?
?
**Fluency vs understanding** — a system can produce fluent surface output without understanding what it says.

Does fluent generation guarantee understanding?
?
No. Language can be generated using surface-level patterns; behaviour can mask lack of understanding.

Name a core NLP task and a non-NLP task.
?
Core NLP: Named Entity Recognition. Non-NLP: numerical optimization / image segmentation / graph traversal.

Give one understanding task and one generation task.
?
Understanding: sentiment analysis (or NER, QA, IE). Generation: machine translation (or summarization, dialogue, completion).

What does syntax answer vs what does semantics answer?
?
Syntax: "Is this sentence well-formed?" Semantics: "What does this sentence mean?"

What are the three things semantic analysis must do?
?
Resolve ambiguity, model relationships between words/concepts (similarity, entailment, roles, references), and connect language to internal representations (vectors, logical forms, graphs).

List the eight stages of the classical NLP pipeline.
?
Text acquisition → Normalization & preprocessing → Morphological & lexical analysis → Syntactic analysis → Semantic representation → Discourse & contextual integration → Task modelling → Output.

What step must occur before counting word frequencies?
?
**Tokenization** — you cannot count word counts before splitting the text into tokens.

What does NLP NOT imply, despite producing fluent text?
?
**True human-level understanding.** Most systems detect patterns in data rather than reason about meaning.

A widely accepted view about current NLP systems is that they…
?
…can perform language tasks effectively **without understanding**.
