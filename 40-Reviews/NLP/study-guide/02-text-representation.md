---
type: study-guide-cluster
course: NLP
cluster: "02-text-representation"
theme: "Tokens, bag-of-words, n-grams"
prerequisites: [01-foundations]
covers-concepts:
  - tokenization
  - morpheme-and-subword-tokenization
  - stemming-vs-lemmatization
  - stop-words
  - bag-of-words
  - n-gram-model
  - nlp-libraries
  - one-hot-encoding
  - term-document-matrix
covers-lectures:
  - lecture-04-advanced-concepts
  - lecture-05-python-modules
exam-weight: high
---

# Cluster 2: Tokens, bag-of-words, n-grams

> **The story of this cluster in one sentence.** Cluster 1 said language is messy; this cluster says **count it anyway** — chop text into discrete units and represent documents as vectors of frequencies, accepting that you'll lose word order in exchange for being able to do *any* math at all.

## Why this cluster exists

Symbolic NLP failed because hand-written rules don't scale (Cluster 1's lesson). The cheapest replacement is to give up on understanding sentences as *structures* and treat them as **multisets of tokens**. The conceptual move is brutal: throw away order and syntax, keep only what occurred and how often. Almost every method later in the course is a reaction to one of the things we lose here. This is also the cluster where the **Code 1** exam question lives — building a TF matrix with `Counter` from `bag-of-words` is the canonical 10-pt code idiom.

**Prerequisites you should feel solid on:**

- [[language-ambiguity]] — to understand *why* we accept the lossiness of bag-of-words: we're trading one ambiguity for another (we lose syntactic disambiguation in exchange for tractability)
- [[nlp-pipeline]] — tokenization and morphological normalization are the *first two* stages of the classical pipeline

## The arc

### 1. [[tokenization]] — turning text into a sequence of discrete units

Tokenization is the first irreversible decision: where does one "word" end and the next begin? Whitespace works for English ("the cat sat") and breaks for almost everything else (Chinese has no spaces; English has "don't" → 1 or 2 tokens?; URLs and emoji confuse simple rules). The output is a **list of token strings** — that's the `tokenize → vocab → counts` chain that every later representation builds on. Every Code 1 question on the exam starts with the assumption that tokenization has already happened.

### 2. [[morpheme-and-subword-tokenization]] — going below the word

Once you realize "running", "runs", "ran" share a stem, you can either **normalize them to one form** (next concept) or **chop them into smaller units** (subword tokenization, the modern HuggingFace default). Subword tokenization (BPE, WordPiece) is what lets transformers handle out-of-vocabulary words and rare languages — the choice you make here determines what your model can even *see*. Conceptual link forward: HF's `tokenizer = AutoTokenizer.from_pretrained(...)` (Code 2) hides this whole choice behind one line.

### 3. [[stemming-vs-lemmatization]] — normalizing morphology

Two ways to collapse "running" / "runs" / "ran" to one form. **Stemming** is rule-based, fast, and approximate (Porter stemmer chops suffixes; "running" → "run", "studies" → "studi" — yes, that's wrong but close enough). **Lemmatization** is dictionary-based, slow, and accurate (uses POS to know "saw" → "see" as verb, "saw" → "saw" as noun). The exam tests the distinction (Quiz I Q5) and the trade-off: speed vs. correctness, both reduce vocabulary size at the cost of some semantic information.

### 4. [[stop-words]] — throwing out the noise

"The", "a", "of" appear in *every* document and so contribute almost no discriminative information. **Removing them** shrinks the vocabulary and speeds up retrieval — but it also throws away genuine signal in some tasks (negation cues, function words that change meaning). This is the *discrete* version of the idea that TF-IDF (Cluster 3) implements *continuously*: down-weight high-frequency words. Knowing that connection is one of the cluster-3 "aha" moments.

### 5. [[one-hot-encoding]] — the simplest possible word vector

Assign each word a unique index in a `|V|`-dim vector that is `1` at its index and `0` elsewhere. This is the **representation we'll spend the rest of the course replacing** because it has two fatal flaws: (a) every pair of distinct words has cosine similarity 0 — "cat" and "kitten" look as different as "cat" and "telephone"; (b) `|V|` is in the tens of thousands, so vectors are enormous and sparse. Mock Q23 and Q25 are entirely about the contrast between one-hot and dense embeddings — this concept exists to be the *bad baseline*. The bridge: if every word is its own dimension, we can't discover *similarity*. We need a way to make similar words share structure.

### 6. [[bag-of-words]] — a document is a vector of token counts

The **bag-of-words (BoW)** representation drops one-hot vectors over each token and **sums** them: a document becomes a `|V|`-dim count vector where entry `i` is "how many times word `i` appears." Word order is gone — that's why it's called a "bag." This is the workhorse representation for classical text classification (Cluster 4) and the Code 1 exam idiom: build a vocabulary, count tokens per document, output a matrix. The blueprint flags this as **very high weight** for Code 1 (the `Counter` skeleton).

### 7. [[term-document-matrix]] — stacking BoW vectors into a matrix

Stack the BoW vector of every document as a row (or column, depending on convention). You now have a **term-document matrix** of shape `|V| × N`. This single matrix is the substrate for almost every classical method that follows: TF-IDF reweights its entries (Cluster 3), LSA factorizes it (Cluster 5), classifiers consume its rows. *The matrix is the door.* Spotting it lets you connect distant-looking methods.

### 8. [[n-gram-model]] — putting some local order back in

BoW threw away word order. **N-grams** put a tiny bit back: instead of single words, count sequences of `n` consecutive tokens. A bigram model represents "I love NLP" as the bag {(I, love), (love, NLP)}. This restores **local context** (good: "not happy" stays together; bad: vocabulary explodes — `|V|^n` possible n-grams). The MLE estimate `c(w_{t-1}, w_t) / c(w_{t-1})` is on the formula sheet; Quiz I Q30–Q34 drill counting bigrams and trigrams from a small corpus by hand.

### 9. [[nlp-libraries]] — the practical toolkit

Four libraries do most of the work in modern NLP, and the exam wants you to know which is for what:

- **NLTK** — educational, broad, slow. Used in classroom examples.
- **spaCy** — industrial, fast, pre-trained pipelines. Production tokenizer + POS + NER.
- **gensim** — topic models and embeddings (LSA, Word2Vec, fastText). Cluster 5 uses it.
- **HuggingFace `transformers`** — pre-trained transformer models, the substrate for Code 2 and most of Cluster 9.

Quiz I Q35–Q44 is a library-purpose MCQ block. These are *cheap* exam points if you can map task → library on sight.

## Connections worth seeing

- **Stop-word removal is the discrete version of TF-IDF down-weighting.** Both penalize high-frequency words; one does it with a hard cutoff (in/out), the other does it continuously (`log(N/DF)`). When you understand that, IDF stops being a magical formula and becomes "smooth stop-word removal."
- **One-hot → BoW → TF-IDF → embeddings is the same story told four times with smoother coordinates.** Each step keeps the "document is a vector" frame and changes only *how the entries are computed*. The final reveal in Cluster 5 is that the *axes themselves* can be learned.
- **The term-document matrix shows up under three different names in this course:** raw `term-document-matrix` here, the **TF matrix** in Cluster 3 (after TF-IDF reweighting), and the **input matrix `X`** in Cluster 5 (LSA factorizes it as `UΣVᵀ`). Same object, different lenses.
- **Subword tokenization is a defence against the same problem as Laplace smoothing (Cluster 4).** Both deal with **unseen vocabulary** — Laplace smooths over zero counts, subword tokenizers ensure no token is truly unseen by reducing to a known piece inventory.

## Common confusions

- **Stemming vs. lemmatization** — stemming chops suffixes by rule (fast, approximate); lemmatization uses a dictionary + POS (slow, exact).
- **Bag-of-words vs. n-gram model** — BoW = unigrams = `n=1`; n-grams generalize to local sequences.
- **One-hot vs. BoW** — one-hot represents a *single word*; BoW represents a *document* by summing one-hot vectors of its words.
- **Term-document matrix vs. document-term matrix** — same data, transposed. The course mostly uses term-document (rows = words, columns = docs); some libraries use the transpose. Always check axes when reading code.
- **Tokenization vs. parsing** — tokenization just splits the string; parsing assigns *structure* (POS tags, dependency tree). They're different stages of the pipeline.

## Self-check (synthesis, not recall)

1. **Why does removing stop words help retrieval but hurt sentiment classification?** Tie your answer to *what each task asks the words to do.*
2. The bag-of-words model loses word order. Trace forward through the next four clusters and identify, for each, *one* method that puts some order back in. Are they doing it the same way?
3. A vocabulary of 50 000 words gives a one-hot vector with 50 000 dimensions. **What is the storage and computation cost** of representing 1 million documents this way, vs. as 100-dim dense embeddings (Cluster 5)? How does this connect to mock Q25's "dense vs sparse" framing?
4. **You are about to write the Code 1 answer for the mock exam (the `Counter`-based TF matrix).** Sketch the 8 lines from memory, then check `[BoW with Counter (cells 2–5)](30-Sources/NLP/notebooks/03_BoW_modelling.ipynb)`. Which line is most likely to trip you up under exam pressure?
5. **Looking forward to Cluster 3:** BoW gives every word a count. But "the" appears in every document and "Heisenberg" appears in two — both get a count. What weighting scheme reflects the second word's higher information content?

## If you have 10 minutes

1. The `Counter` skeleton in [[bag-of-words]] — write it from memory until you can produce it without consulting the file
2. [[stemming-vs-lemmatization]] — three differences in one paragraph
3. [[nlp-libraries]] — match the four libraries to their tasks (NLTK / spaCy / gensim / HF) — pure MCQ points

## Next cluster

→ [[03-patterns-and-retrieval]] — having turned documents into count vectors, two questions remain: *how do I find the documents I want* and *how do I weight rare words higher than common ones?* Cluster 3 answers both with regex, IR ranking, TF-IDF, and cosine similarity — and motivates Cluster 5 by exposing what TF-IDF *can't* do (vocabulary mismatch).
