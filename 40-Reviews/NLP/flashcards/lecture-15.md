---
tags: [flashcards, NLP]
---

# Lecture 15 — Named Entity Recognition (flashcards)

What is named entity recognition (NER)?
?
The task of identifying segments of text that refer to **specific real-world entities** and assigning each segment a **semantic type** (e.g. PERSON, ORG, LOC, DATE, MONEY).

Why is NER not the same as POS tagging?
?
A named entity is **not just a noun** — it's an *expression* with a referential interpretation. "Washington" is a proper noun grammatically, but semantically it could be a PERSON, LOC, or ORG depending on context. NER operates at the **interface between syntax and semantics**.

What does the IOB representation encode?
?
Per-token labels: **B-X** (Beginning of entity X), **I-X** (Inside / continuation of entity X), **O** (Outside any entity). It turns variable-length span detection into a per-token sequence labelling problem.

Tag "Barack Obama visited New York" in IOB.
?
Barack/B-PER  Obama/I-PER  visited/O  New/B-LOC  York/I-LOC.

Where does NER sit in the NLP pipeline?
?
**After** part-of-speech tagging and **before** full information extraction. POS provides grammatical categories; NER introduces referential structure ([[30-Sources/NLP/pdf/Session 15 - NER.pdf#page=7|slide 7]]).

What's the central difficulty in NER that POS doesn't have?
?
**Entities span multiple tokens** ("New York", "Bank of America"). The system must determine both the **boundaries** and the **semantic type**, introducing structural dependencies across positions.

What model does classical NER use?
?
A **generative Hidden Markov Model**: entity tags are hidden states forming a Markov chain; words are observable emissions. Mathematically identical to the POS HMM — only the interpretation of the hidden states changes.

What two assumptions define the HMM for NER?
?
(1) **Emission independence**: each word depends only on its entity tag, $P(w_i \mid t_i)$.
(2) **First-order Markov**: each tag depends only on the previous tag, $P(t_i \mid t_{i-1})$.

What's the joint factorization of the HMM for NER?
?
$P(w_{1:n}, t_{1:n}) = \prod_{i=1}^{n} P(w_i \mid t_i) \cdot P(t_i \mid t_{i-1})$ — same as POS.

What decoding algorithm finds the most probable entity sequence?
?
**Viterbi** dynamic programming. It stores the best partial path ending in each tag at each position, and reconstructs the globally optimal sequence by walking backpointers.

What two conditions must hold for a NER prediction to be counted correct?
?
**Both** the **entity span (boundaries)** *and* the **entity type** must match exactly (Quiz III Q15). Partial matches don't count.

What's the main consequence of strict span-based evaluation?
?
Predicting "New York" when the true entity is "New York City" is **incorrect** — not partially correct. Boundary errors and type errors are both penalized fully.

What's NER precision?
?
Proportion of predicted entities that are correct: $P = \dfrac{|\text{correct predictions}|}{|\text{predicted entities}|}$.

What's NER recall?
?
Proportion of true entities that the system recovered: $R = \dfrac{|\text{correct predictions}|}{|\text{true entities}|}$.

What three things change from POS tagging to NER?
?
(1) **Label space**: grammatical → referential / semantic. (2) **Error type**: tag misclassification → boundary mistakes + type mistakes. (3) **Evaluation**: per-token accuracy → strict span-based precision/recall.

What are common error sources in classical NER?
?
**Boundary detection mistakes**, **type confusion** between similar categories, **data sparsity** for rare names, **contextual ambiguity** (e.g. "Washington"), **nested entities** ("Bank of America Tower" with ORG inside LOC).

Why does classical HMM-based NER struggle with nested entities?
?
The Markov assumption captures only the **previous tag** — it can't represent overlapping or hierarchical spans. The IOB encoding itself only allows flat spans.

What does the Viterbi recursion compute for NER?
?
$V_t(j) = \max_i [V_{t-1}(i) \cdot a_{ij}] \cdot b_j(w_t)$ — best path probability ending in entity tag $j$ at position $t$. Backpointers $\psi_t(j) = \arg\max_i V_{t-1}(i) \cdot a_{ij}$ allow trajectory reconstruction.

What are the two HMM problems, and which one Viterbi solves?
?
**Decoding** (find the most likely hidden sequence given parameters) — solved by **Viterbi**.
**Learning** (estimate parameters from observed sequences alone) — solved by **Baum–Welch / EM**.

For NER on a labelled corpus, which HMM problem do you need to solve in addition to Viterbi?
?
None of the EM-flavoured ones — labelled corpora let us estimate emissions and transitions by **MLE counts + smoothing**, exactly like Multinomial Naïve Bayes. Baum–Welch is needed only when tags are unobserved.

If a system predicts B-LOC I-LOC for "New York" but the gold annotation is B-PER I-PER, how does the strict evaluation count this?
?
Span is correct (both tokens grouped into one entity), but **type is wrong** — it counts as **incorrect** under strict span-based precision/recall.

Why are NER predictions at different positions not independent?
?
Because adjacent IOB labels constrain each other: an `I-PER` cannot follow an `O` or a `B-LOC` in a well-formed sequence — the label at one position constrains the label at the next.

What's the conceptual continuity between POS tagging and NER?
?
Same mathematical machinery: HMM with first-order Markov transitions and emission independence, decoded by Viterbi. Same evaluation framework family. **Only the label space and the strictness of evaluation change** — POS predicts grammatical categories with token accuracy; NER predicts entity types with strict span F1.
