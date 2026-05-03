---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 15 - NER.pdf
  - course: NLP
    file: 11_Named_Entity_Recognition.ipynb
created: 2026-05-02
---

# Named Entity Recognition (NER)

**NER** is the task of identifying segments of text that refer to **specific entities** in the world and assigning each segment a **semantic type** ([[30-Sources/NLP/pdf/Session 15 - NER.pdf#page=5|slide 5]]). Typical entity types: PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PERCENT, GEOPOLITICAL ENTITY (GPE).

> "Apple hired John Smith in Paris in 2024" → Apple/ORG, John Smith/PERSON, Paris/LOC, 2024/DATE.

The blueprint flags this as **high weight**: mock Q10, Quiz III Q14, Q15, Q20 (and Model B variants) — span vs type errors, span-based evaluation, POS-vs-NER contrasts.

## Why NER is not POS tagging

A named entity is **not simply a noun**. It's an *expression* that refers to a particular object, group, place, or concept ([[30-Sources/NLP/pdf/Session 15 - NER.pdf#page=6|slide 6]]). "Washington" can grammatically be a proper noun while semantically referring to a person, a state, a city, or an institution — **classification depends on context**.

NER operates at the **interface between syntax and semantics** — it uses grammatical awareness but aims at **referential interpretation**, not structural classification.

## Three structural changes from POS to NER ([[30-Sources/NLP/pdf/Session 15 - NER.pdf#page=18|slide 18]])

| Aspect | POS | NER |
|---|---|---|
| **Label space** | Grammatical (Noun, Verb, Det, …) | **Referential / semantic** (PER, ORG, LOC, DATE, …) |
| **Error type** | Tag misclassification only | **Boundary mistakes + type mistakes** |
| **Evaluation** | Per-token accuracy | **Strict span-based precision / recall** |

## From words to spans ([[30-Sources/NLP/pdf/Session 15 - NER.pdf#page=8|slide 8]])

A central difficulty: **entities are not single words**. They are spans of variable length:
- "New York" — 2 tokens, 1 LOC entity
- "Bank of America" — 3 tokens, 1 ORG entity

The system must determine **both the boundaries and the semantic type** simultaneously. This introduces structural dependencies across positions — predictions at neighbouring positions are not independent.

## The IOB representation ([[30-Sources/NLP/pdf/Session 15 - NER.pdf#page=9|slide 9]])

To turn variable-length span detection into a token-level sequence labelling problem, each token receives one of:
- **B-X** — Beginning of an entity of type X
- **I-X** — Inside (continuation of) an entity of type X
- **O** — Outside any entity

Worked example:

| Token | Barack | Obama | visited | New | York |
|---|---|---|---|---|---|
| IOB tag | B-PER | I-PER | O | B-LOC | I-LOC |

Span boundaries are recovered by scanning B/I transitions: `B-PER I-PER` → "Barack Obama" (PERSON); `B-LOC I-LOC` → "New York" (LOC).

This **per-token prediction problem with tag transitions** is exactly the shape that [[hidden-markov-model|HMM]] + [[hmm-viterbi|Viterbi]] decode.

## The HMM model ([[30-Sources/NLP/pdf/Session 15 - NER.pdf#page=11|slide 11]])

Identical mathematical structure to POS — **only the interpretation of the hidden states changes**:
- **Hidden states** = entity IOB tags (B-PER, I-PER, B-LOC, I-LOC, O, …) forming a Markov chain
- **Observable emissions** = words

Two assumptions:
- **Emission independence:** $P(w_i \mid t_i)$ — word depends only on its entity tag
- **First-order Markov:** $P(t_i \mid t_{i-1})$ — tag depends only on the previous tag

Joint factorization:
$$P(w_{1:n}, t_{1:n}) = \prod_{i=1}^{n} P(w_i \mid t_i) \cdot P(t_i \mid t_{i-1})$$

Decoding via [[hmm-viterbi|Viterbi]] gives the most likely IOB sequence, from which spans are read off.

## Strict span-based evaluation ([[30-Sources/NLP/pdf/Session 15 - NER.pdf#page=17|slide 17]])

> "A prediction is correct **only if both the boundaries and the entity type match exactly**." (Quiz III Q15)

- **Precision** = correct predicted entities / total predicted entities
- **Recall** = correct predicted entities / total true entities
- **F1** = $2 \cdot P \cdot R / (P + R)$

**Partial matches don't count.** If the system predicts "New York" but the true entity is "New York City", the prediction is **incorrect** (boundary mismatch). This strict evaluation reflects the structural nature of the task — getting the type right but the span wrong (or vice versa) is failure.

## Error sources in classical NER ([[30-Sources/NLP/pdf/Session 15 - NER.pdf#page=17|slide 17]])

- **Boundary detection mistakes** (under- or over-extending the span)
- **Type confusion** between similar categories (e.g. PER vs ORG for "Apple")
- **Data sparsity** for rare names
- **Contextual ambiguity** ("Washington")
- **Nested entities** ("Bank of America Tower" — ORG inside LOC)

These reflect the **limits of local dependencies** in the HMM model: the first-order Markov property captures only one step of context, and emission probabilities depend only on the current tag.

## Why NER motivates richer models

Classical HMM-based NER hits ceilings:
- Long-distance contextual cues (a person's role mentioned paragraphs earlier)
- Cross-sentence coreference
- Nested / overlapping entities

This motivates **discriminative alternatives** (CRFs) and ultimately **neural sequence models** — RNNs (Session 17), transformers (Session 19) — that capture richer context.

## Exam framing

| Question | Answer |
|---|---|
| What is NER? | Identifying spans referring to real-world entities and assigning them semantic types (mock Q10) |
| What two conditions must hold for a NER prediction to be correct? | **Both** the entity span (boundary) **and** the entity type must match exactly (Quiz III Q15) |
| How does NER differ from POS? | Label space is referential/semantic (not grammatical); error includes boundary mistakes; evaluation is strict span-based ([[30-Sources/NLP/pdf/Session 15 - NER.pdf#page=18|slide 18]]) |
| What encoding turns spans into per-token labels? | **IOB** — Beginning, Inside, Outside (B-X, I-X, O) |
| What model is used in classical NER? | Generative HMM with words as emissions and entity tags as hidden states ([[30-Sources/NLP/pdf/Session 15 - NER.pdf#page=11|slide 11]]) |
| What's the decoding algorithm? | [[hmm-viterbi|Viterbi]] dynamic programming |

## Related

- [[part-of-speech-tagging]] — adjacent task; same HMM machinery, different label space
- [[hidden-markov-model]] — the generative model
- [[hmm-viterbi]] — the decoder
- [[evaluation-metrics]] — extended to span-based precision/recall/F1

## Canonical pre-trained NER skeleton (spaCy + NLTK)

```python
import spacy
nlp = spacy.load("en_core_web_sm")

def spacy_ner(text: str):
    """Returns list of (span_text, label) pairs."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# spaCy NER
result = spacy_ner("Apple hired John Smith in Paris in 2024.")
# [('Apple', 'ORG'), ('John Smith', 'PERSON'), ('Paris', 'GPE'), ('2024', 'DATE')]

# NLTK NER (POS chunker)
import nltk
from nltk import word_tokenize, pos_tag
from nltk.chunk import ne_chunk

tokens = word_tokenize(text)
tagged = pos_tag(tokens)
tree = ne_chunk(tagged, binary=False)  # binary=False keeps PERSON/ORG/GPE labels
```

Reference: `[spaCy and NLTK NER (cells 5–9)](30-Sources/NLP/notebooks/11_Named_Entity_Recognition.ipynb)`.

**Exam-critical observations:**
- spaCy's `ent.label_` returns broad labels: `ORG`, `PERSON`, `GPE` (geopolitical), `DATE`, `MONEY`, `PERCENT`, ...
- spaCy treats "Paris" as `GPE` (geopolitical entity), not just `LOC` — the prof tests this distinction
- NLTK uses `ne_chunk` which produces an NLTK Tree; the binary=False flag preserves entity-type labels
