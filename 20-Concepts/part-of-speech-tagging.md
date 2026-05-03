---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 14 - POS tagging.pdf
  - course: NLP
    file: 10_Part_of_Speech_tagging.ipynb
created: 2026-05-02
---

# Part-of-speech tagging

**POS tagging** is the task of assigning a **grammatical category** to each word in a sentence ([[30-Sources/NLP/pdf/Session 14 - POS tagging.pdf#page=5|slide 5]]). Categories typically include Noun, Verb, Adjective, Determiner, Preposition, Pronoun, Conjunction.

> Example: "The students study language models" → "The/DET students/NOUN study/VERB language/NOUN models/NOUN"

The blueprint flags this as **high weight**: mock Q9, Quiz III Q13 (and Model B), with Exercise 3 of the mock as a **Viterbi-on-2-state-HMM** computation (10 points).

## Position in the NLP pipeline

POS tagging provides the **grammatical layer** on which parsing and semantic interpretation depend ([[30-Sources/NLP/pdf/Session 14 - POS tagging.pdf#page=6|slide 6]]):
$$\text{Text} \to \text{Tokenization} \to \text{Morphological/Lexical} \to \boxed{\text{POS}} \to \text{Syntactic Parsing} \to \text{Semantics}$$
It is **purely grammatical annotation** — not semantic.

## Why it's a learning problem, not a rule problem ([[30-Sources/NLP/pdf/Session 14 - POS tagging.pdf#page=7|slides 7–9]])

Classical [[context-free-grammar|CFG]]-based approaches assign categories by rule derivation. Two structural failures push toward statistics:

1. **Ambiguity** ([[30-Sources/NLP/pdf/Session 14 - POS tagging.pdf#page=8|slide 8]]): "They *can* fish" (MOD) vs "a *can* of fish" (NOUN). The same word maps to different categories **depending on context** — the word alone is insufficient.
2. **Exhaustive dictionaries fail** ([[30-Sources/NLP/pdf/Session 14 - POS tagging.pdf#page=9|slide 9]]): novel words, exceptions, and language variability mean rule sets can't be made complete.

So POS is reframed as **sequence classification** ([[30-Sources/NLP/pdf/Session 14 - POS tagging.pdf#page=10|slide 10]]):
$$\hat{t}_{1:n} = \arg\max_{t_{1:n}} P(t_1, \ldots, t_n \mid w_1, \ldots, w_n)$$

Predictions are **interdependent across positions** — that's the only conceptual shift from [[naive-bayes|Naïve Bayes]] ([[30-Sources/NLP/pdf/Session 14 - POS tagging.pdf#page=11|slide 11]]).

## How it connects to Naïve Bayes ([[30-Sources/NLP/pdf/Session 14 - POS tagging.pdf#page=11|slide 11]])

| Property | Naïve Bayes | POS (HMM) |
|---|---|---|
| Models | $P(\text{class} \mid \text{document})$ | $P(\text{tag sequence} \mid \text{word sequence})$ |
| Estimation | counts + smoothing | counts + smoothing |
| Independence | features independent given class | tag depends only on previous tag (Markov) |
| Output | single label | structured (tag chain) |

The conditional-probability framework is the same — POS just adds **sequential dependency**.

## How it's solved: HMM + Viterbi

The classical solution is the [[hidden-markov-model|Hidden Markov Model]] with two assumptions:

- **Emission:** each word depends only on its tag — $P(w_i \mid t_i)$
- **Transition:** each tag depends only on the previous tag — $P(t_i \mid t_{i-1})$

The most probable tag sequence is decoded by the **[[hmm-viterbi|Viterbi algorithm]]** (dynamic programming).

## What POS tagging captures and misses ([[30-Sources/NLP/pdf/Session 14 - POS tagging.pdf#page=15|slide 15]])

**Captures:**
- Frequent **tag transitions** (DET typically precedes NOUN)
- Typical **word–category associations** ("the" → DET)
- **Local grammatical regularities**

**Misses:**
- **Long-distance syntactic dependencies** (Markov sees only the previous tag)
- **Hierarchical structure** (phrase-level grouping)
- **Semantic interpretation** (purely grammatical)

## Evaluation framing

POS taggers are evaluated by **per-token accuracy** on a held-out, gold-standard tagged corpus. The blueprint flags evaluation metrics broadly as a recurring exam pattern (Quiz III Q13 specifically targets POS evaluation framing).

## Exam framing

| Question | Answer |
|---|---|
| What is POS tagging? | Assigning a grammatical category to each word in a sentence (mock Q9; Quiz III Q13) |
| Why doesn't a word's tag come from the word alone? | **Ambiguity** — the same word maps to different categories depending on context ([[30-Sources/NLP/pdf/Session 14 - POS tagging.pdf#page=8|slide 8]]) |
| What classical model formalizes POS? | A [[hidden-markov-model|Hidden Markov Model]] — words are observable outputs, tags are hidden states ([[30-Sources/NLP/pdf/Session 14 - POS tagging.pdf#page=12|slide 12]]) |
| What's the conceptual difference from Naïve Bayes? | Predictions are **interdependent across positions** (sequence dependency); the rest of the framework is the same ([[30-Sources/NLP/pdf/Session 14 - POS tagging.pdf#page=11|slide 11]]) |
| What does POS tagging *not* capture? | Long-distance dependencies, hierarchical structure, semantics ([[30-Sources/NLP/pdf/Session 14 - POS tagging.pdf#page=15|slide 15]]) |

## Related concepts

- [[hidden-markov-model]] — the model
- [[hmm-viterbi]] — the decoder
- [[dependency-parsing]] — the next pipeline stage
- [[naive-bayes]] — conceptual ancestor

## Canonical pre-trained tagger skeleton (NLTK + spaCy)

Pre-trained POS taggers are the practical default:

```python
# NLTK approach
import nltk
from nltk import word_tokenize, pos_tag
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

s = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(s)
tags = pos_tag(tokens)   # [('The', 'DT'), ('quick', 'JJ'), ...]

# spaCy approach
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(s)
for t in doc:
    print(t.text, t.pos_, t.tag_)   # universal POS + Penn Treebank tag
```

Reference: `[NLTK + spaCy POS tagging (cells 3–10)](30-Sources/NLP/notebooks/10_Part_of_Speech_tagging.ipynb)`.

**Exam-critical library facts** (Quiz I Q35–Q44):
- `nltk.pos_tag` returns Penn Treebank tags (`DT`, `JJ`, `NN`, `VB`, ...)
- spaCy's `token.pos_` is the **universal POS** (NOUN, VERB, ADJ); `token.tag_` is the more detailed Penn-Treebank-style tag
- For HMM-from-scratch implementation, the notebook walks through the δ-table — drill against the formula-sheet Viterbi recursion
