---
type: study-guide-cluster
course: NLP
cluster: "01-foundations"
theme: "Foundations & history"
prerequisites: []
covers-concepts:
  - nlp-definition
  - formal-vs-natural-language
  - language-ambiguity
  - nlp-pipeline
  - semantic-analysis
  - nlp-understanding-vs-generation
  - eliza
  - turing-test
  - early-symbolic-nlp
  - context-free-grammar
  - semantic-gap
  - abstract-meaning-representation
covers-lectures:
  - lecture-01-introduction
  - lecture-02-history
  - lecture-03-fundamental-concepts
exam-weight: low
---

# Cluster 1: Foundations & history

> **The story of this cluster in one sentence.** Before we can compute *anything* over language, we have to admit *what* language is and *why* every later technique is fighting against the same enemy: **ambiguity**.

## Why this cluster exists

Every method later in the course — bag-of-words, attention, transformers — exists because language refuses to behave like a formal grammar. This cluster names that refusal. It also explains the shape of the field's history: a long detour through hand-written rules and symbolic parsers that *failed* to scale, leaving behind a deep lesson (ELIZA's "behaviour can mask understanding") that still haunts modern LLM evaluation. There are no prerequisites; this is the foundation everything stands on. Exam weight is **low** in points (mostly Quiz I single-MCQ traps about ELIZA, the Turing test, and ambiguity), but conceptually it sets up the *motivation* for every cluster that follows.

**Prerequisites you should feel solid on:**

- *None.* This is the floor.

## The arc

A walkthrough of how the concepts in this cluster build on each other.

### 1. [[nlp-definition]] — what NLP actually is

Frame NLP not as "AI for text" but as the meeting point of **linguistics, statistics, and computation**: the engineering problem of getting machines to read, write, and *understand* language. The reason this needs its own field — rather than living inside compilers or formal logic — is that natural language is the *messy* sibling of the symbolic systems CS already handles. That messiness is the through-line of the cluster.

### 2. [[formal-vs-natural-language]] — the gap that creates the field

Formal languages (programming languages, logic, regex) are designed to be unambiguous: every string has at most one meaning, every meaning has one well-formed expression. Natural languages are the opposite — they evolved for **communication under shared context**, not for parsing. This concept is the *single sentence* that justifies why every classical NLP technique you'll see ends up brittle: you're trying to apply formal-language tools to a fundamentally non-formal object. The bridge to the next concept: if natural language isn't formal, what *kinds* of trouble does that create?

### 3. [[language-ambiguity]] — the trouble, named and catalogued

**Lexical** ("bank" = riverside or financial), **syntactic** ("I saw the man with the telescope" — who has it?), **semantic** ("flying planes can be dangerous"), **pragmatic** ("can you pass the salt?" — request, not yes/no question). Every later method in the course targets one or more of these. TF-IDF doesn't help with lexical ambiguity. RNNs partially help with syntactic context. Attention helps with all four — that's what makes it powerful. Carrying a mental list of ambiguity types lets you *predict* which method addresses what.

### 4. [[nlp-pipeline]] — the canonical sequence of transformations

The classical pipeline: **raw text → tokenization → morphological analysis → syntactic parsing → semantic representation → application**. Each stage is what later clusters will name and study (Cluster 2 = the first two stages, Cluster 6 = parsing, etc.). The pipeline view also explains why deep neural models are sometimes called "end-to-end": they collapse this whole stack into one differentiable function rather than running each stage independently.

### 5. [[semantic-analysis]] — what "meaning" even is

Semantic analysis tries to map a sentence to a structured representation of *what it asserts*. This is harder than syntax because it has to commit to an *ontology* — entities, predicates, relations. The hard problem is that two surface-different sentences ("Brutus killed Caesar" / "Caesar was killed by Brutus") share semantics, while two surface-similar sentences ("the cat is on the mat" / "the dog is on the mat") differ. Foreshadows why dense embeddings are eventually useful: similarity in meaning needs to *not* be similarity in spelling.

### 6. [[nlp-understanding-vs-generation]] — two halves of the field

**NLU** maps text → structured meaning (classification, parsing, QA span extraction). **NLG** maps structured meaning → text (summarization, translation, chatbot replies). Most modern systems do both — encoder-decoder architectures literally have one of each — but historically they were studied separately, and the failure modes still differ (NLU is about *missing* meaning; NLG is about *making up* meaning, i.e. hallucination).

### 7. [[eliza]] — the surprise that gave the field a moral

ELIZA (Weizenbaum, 1966) is a regex-based chatbot impersonating a Rogerian therapist with about 200 lines of pattern rules. People reported feeling *understood* by it. The lesson — the **ELIZA effect** — is that **superficial behavioural fluency can mask the absence of understanding**. This is the most exam-tested item in the cluster (Quiz I Q13, Q16–Q23) and the central caution behind every modern LLM benchmark: passing a behavioural test ≠ comprehending the input.

### 8. [[turing-test]] — the operational definition that started the argument

Turing (1950) proposed: if a human judge in a text-only conversation can't reliably tell a machine from a person, the machine "thinks." The test sidesteps the philosophical question by replacing it with a behavioural one — and ELIZA showed exactly why that substitution is suspect. Memorize: imitation game (3 participants), text-only channel, the goal is *deception*, not understanding. Quiz I tests both the original 1950 framing and the standard objections (Searle's Chinese Room, the ELIZA effect).

### 9. [[early-symbolic-nlp]] — the road not taken (mostly)

Symbolic NLP (1950s–1990s) tried to handle language with **hand-written rules + logic + grammars**. It produced beautiful systems (SHRDLU, MARGIE) for tiny domains but couldn't scale: every new sentence type needed new rules. The lesson is the **knowledge-engineering bottleneck** — and it's why the field eventually pivoted to statistics (Clusters 2–4) and then to learned representations (Clusters 5–8).

### 10. [[context-free-grammar]] — the symbolic tool that almost worked

A context-free grammar (CFG) defines a language by **production rules** (S → NP VP, NP → Det N, …). CFGs are the formal-language ancestor of every parser. They're powerful enough to capture much of programming-language syntax — but they can't express the **long-range agreement** of natural language ("the **dogs** that he saw yesterday in the park **are** running") without ballooning. CFGs are exam-relevant as the symbolic baseline against which dependency parsing (Cluster 6) and transformers (Cluster 8) are eventually compared.

### 11. [[semantic-gap]] — the chasm symbolic methods couldn't cross

The **semantic gap** is the distance between **what's encoded in the text surface** and **what the text actually means in the world**. Symbolic NLP tried to bridge it by hand (ontologies, knowledge bases). Statistical NLP works *around* it (treat similar contexts as encoding similar meanings — the [[distributional-hypothesis|distributional hypothesis]] of Cluster 5). Naming this gap is what motivates the entire turn from symbolic → statistical → neural NLP.

### 12. [[abstract-meaning-representation]] — the symbolic dream's last gasp

AMR encodes a sentence as a **rooted, directed, acyclic graph** of concepts and relations, abstracting away from surface syntax. Two paraphrases get the same AMR. It's elegant and human-readable — and it's also what most of NLP eventually *gave up on*, because hand-annotated AMR is expensive and learned models do well enough without it. Worth knowing as an example of structured semantics, but it's a low-frequency exam item (briefly named in Quiz I).

## Connections worth seeing

- **The ELIZA effect is the same failure mode as the *accuracy trap* in Cluster 4.** ELIZA looks intelligent because we measure superficial fluency; a 99%-accurate classifier on 99%-imbalanced data looks competent because we measure the wrong number. Both are warnings that **a system passing a casual benchmark may be doing something trivial.**
- **The "knowledge-engineering bottleneck" of symbolic NLP is the same shape as the "feature engineering bottleneck" of classical ML.** Both end the same way: someone replaces hand-built structure with **representations learned from data** — distributional vectors in Cluster 5, end-to-end neural models in Clusters 7–8.
- **Ambiguity → motivation for context.** Lexical ambiguity ("bank") motivates *contextual* representations, which is what every architecture from RNN onwards is trying to provide. Each cluster from 5 onwards essentially asks: *how much context can I look at, and how cheaply?*
- **AMR vs. dependency parsing (Cluster 6) vs. attention weights (Cluster 8)** are three takes on the same question: *what's the structural relation between words in this sentence?* AMR encodes it symbolically, dependency parsing learns it as a tree, attention learns it as a soft pairwise weighting.

## Common confusions

- **NLU vs. NLG** — NLU reads, NLG writes. An encoder is NLU, a decoder is NLG, a transformer is both glued together.
- **Turing test vs. ELIZA effect** — the test is a *protocol*, the effect is a *bias* in human observers. ELIZA passed an informal Turing test in the 1960s precisely *because* of the effect.
- **Lexical vs. syntactic ambiguity** — lexical = single word has multiple senses ("bank"); syntactic = sentence has multiple parse trees ("I saw the man with the telescope").
- **CFG vs. dependency grammar** — CFG describes phrase structure (rewrite rules → parse tree); dependency grammar describes word-to-word relations directly (no internal nodes).
- **Formal vs. natural language** — formal = unambiguous by design (programming languages, logic); natural = ambiguous because evolved for *communicative use*, not *machine parsing*.

## Self-check (synthesis, not recall)

1. **The ELIZA effect** showed that fluent behaviour can mask the absence of understanding. Pick one method from a *later* cluster (e.g. extractive QA, RAG, transformer language modelling) and argue whether it is or isn't subject to the same failure mode. Be specific about *what* the system optimizes vs. *what* a human reader assumes it understands.
2. **Symbolic NLP failed because of the knowledge-engineering bottleneck.** What is the analogous bottleneck for the classical statistical methods in Clusters 2–4, and how does the move to learned embeddings (Cluster 5) get around it?
3. The four ambiguity types (lexical / syntactic / semantic / pragmatic) — match each to the *earliest* cluster in this guide whose method *partially* addresses it. Which type does no method in the course fully solve?
4. AMR captures meaning as a graph; transformers capture context as a soft attention pattern. Are these solving the same problem? Where do they disagree on what "meaning" looks like?
5. **Looking forward to Cluster 2:** if all we have is hand-written grammars and pattern-matching rules, what's the cheapest possible move that lets us start *learning* from text instead of hand-coding rules?

## If you have 10 minutes

The minimum viable review for this cluster:

1. [[language-ambiguity]] — read the four-type breakdown and one example of each
2. [[eliza]] + [[turing-test]] — understand the ELIZA effect *as the warning the field still cites*; memorize the imitation-game protocol
3. The "Common confusions" block above — the MCQ traps in Quiz I are almost entirely from this list

## Next cluster

→ [[02-text-representation]] — having admitted that language is messy and ambiguous, the cheapest thing we can do is **count the words we see**. Cluster 2 turns text into vectors of word frequencies, then immediately confronts the consequences: word order is destroyed, vocabularies are huge, and synonyms look unrelated. That's the trade we're about to make.
