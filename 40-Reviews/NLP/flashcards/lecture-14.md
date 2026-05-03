---
tags: [flashcards, NLP]
---

# Lecture 14 — POS Tagging (flashcards)

What is part-of-speech tagging?
?
The task of assigning a **grammatical category** (Noun, Verb, Adjective, Determiner, …) to each word in a sentence.

Why can't a word's POS tag be determined from the word alone?
?
**Ambiguity.** "They *can* fish" → MOD; "a *can* of fish" → NOUN. The same word maps to different categories depending on context.

What's the formal objective of POS tagging as a learning problem?
?
Find $\hat{t}_{1:n} = \arg\max_{t_{1:n}} P(t_1, \ldots, t_n \mid w_1, \ldots, w_n)$ — the most probable tag sequence given the word sequence.

What are the two assumptions of an HMM?
?
**Emission**: each word depends only on its tag, $P(w_i \mid t_i)$.
**Transition** (first-order Markov): each tag depends only on the previous tag, $P(t_i \mid t_{i-1})$.

How does the HMM factorize the joint probability of words and tags?
?
$P(w_{1:n}, t_{1:n}) = \prod_{i=1}^{n} P(w_i \mid t_i) \cdot P(t_i \mid t_{i-1})$ — product of per-step (emission · transition).

What's the per-step contribution to the HMM joint probability?
?
$P(\text{tag} \mid \text{prev tag}) \cdot P(\text{word} \mid \text{tag})$ — **multiply** them, not add (Quiz III Q12).

How is the transition probability estimated from a labelled corpus?
?
$P(t_i \mid t_{i-1}) = \dfrac{\text{count}(t_{i-1} \to t_i)}{\text{count}(t_{i-1})}$ — counts of consecutive tag pairs over count of the previous tag.

How is the emission probability estimated from a labelled corpus?
?
$P(w_i \mid t_i) = \dfrac{\text{count}(w_i \cap t_i)}{\text{count}(t_i)}$ — counts of (word, tag) pairs over count of the tag.

Why apply smoothing to HMM probabilities?
?
To **avoid zero probabilities** for tag transitions or word–tag pairs unseen in training. Same idea as in Multinomial Naïve Bayes (Laplace / add-α).

What's the Viterbi initialization step?
?
$\delta_1(j) = P(t_1 = j) \cdot P(w_1 \mid t_1 = j)$ — prior probability of starting in tag $j$ times the emission probability of the first word given $j$.

What's the Viterbi recursion?
?
$\delta_t(j) = \max_i [\delta_{t-1}(i) \cdot P(t_t = j \mid t_{t-1} = i)] \cdot P(w_t \mid t_t = j)$ — best path ending in tag $j$ at time $t$.

What single operator distinguishes Viterbi from the forward algorithm?
?
**`max`** instead of `Σ`. Viterbi takes the max over previous states (picking the single best path); forward sums over all paths (computing total probability).

Why does Viterbi store backpointers?
?
The `max` picks one predecessor at each step. To reconstruct the full best path at termination, we need to remember **which predecessor** was chosen — that's the backpointer $\psi_t(j) = \arg\max_i [\delta_{t-1}(i) \cdot P(j \mid i)]$.

What's the time complexity of Viterbi vs brute force?
?
Brute force: $|T|^n$ paths. Viterbi: $O(n \cdot |T|^2)$ — quadratic in the tag set, linear in the sentence.

What's the conceptual difference between POS tagging and Naïve Bayes classification?
?
The framework (counts, conditional probabilities, smoothing, likelihood) is the same. The only difference is that POS predictions are **interdependent across positions** — a structured output (sequence of tags), not a single label.

What does an HMM POS tagger capture?
?
**Frequent tag transitions** (DET → NOUN), **typical word–category associations** ("the" → DET), and **local grammatical regularities**.

What does an HMM POS tagger NOT capture?
?
**Long-distance syntactic dependencies** (Markov sees only one step back), **hierarchical structure** (phrases like NP/VP), and **semantic interpretation** (POS is purely grammatical).

What does a dependency parser output?
?
**A tree of syntactic relations between words** — each non-root word has exactly one head, and edges are labelled with relations like `nsubj`, `obj`, `det` (mock Q11).

Where does POS tagging sit in the NLP pipeline?
?
Between **morphological/lexical analysis** and **syntactic parsing** — provides the grammatical layer on which parsing and semantic interpretation depend.

Why do rule-based POS taggers fail?
?
Lexicons + handcrafted disambiguation rules can't be made complete: language variability, novel words, and exceptions multiply faster than rules can be added ([[30-Sources/NLP/pdf/Session 14 - POS tagging.pdf#page=9|slide 9]]). Hence the move to statistical models.

In the HMM graphical model, what are the hidden states and the observations for POS?
?
**Hidden states = tags** ($t_1, t_2, \ldots$) which form a Markov chain. **Observations = words** ($w_1, w_2, \ldots$) emitted from each hidden state.

For a 2-state HMM with tags ∈ {N, V}, prior P(N)=0.6 P(V)=0.4, and a sentence "fish run" — what's $\delta_1(N)$ if P(fish|N)=0.3?
?
$\delta_1(N) = P(t_1=N) \cdot P(\text{fish} \mid N) = 0.6 \cdot 0.3 = 0.18$. (And $\delta_1(V)$ is computed analogously with $P(t_1=V) \cdot P(\text{fish} \mid V)$.)

Why is sequence modelling needed for "They can fish"?
?
Tagging each word independently is ambiguous — "can" could be MOD or NOUN. The HMM compares full sequences (PRON-MOD-VERB vs PRON-NOUN-NOUN) using transition + emission probabilities, so **context resolves ambiguity statistically**.
