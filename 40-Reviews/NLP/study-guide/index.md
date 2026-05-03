---
type: study-guide-index
course: NLP
created: 2026-05-03
---

# NLP Study Guide

A narrative layer above the lecture notes, concept notes, and flashcards. Each cluster re-tells one chapter of the course as a continuous story — the lecture/concept files give you the *what*, this guide gives you the *why* and the *connective tissue*.

## How to use this guide

Read the clusters in order on a first pass — each one opens by referencing the previous cluster's unsolved problem, so the spine of the course only emerges if you follow the chain. For review, drop into a single cluster and use the "If you have 10 minutes" block. A few days after you've read a cluster, try the **synthesis self-checks** without notes: those questions ask you to *connect* concepts, which is what the exam's contrast-style MCQs and 10-pt exercises actually test.

## The shape of the course in one paragraph

The course is a chain of representations, each invented to fix what the previous one couldn't see. We start with **what language is and why it's hard** — ambiguity, the symbolic dream, the ELIZA effect (Cluster 1). To get any traction we **chop text into tokens and count what we see**, accepting that word order is gone (Cluster 2: tokenization, BoW, n-grams). Counting alone won't *find* relevant text or weight rare-but-informative words, so we add **patterns and weighting** (Cluster 3: regex, IR ranking, TF-IDF, cosine — and the *vocabulary-mismatch* wound that never quite heals). With weighted document vectors we can **learn classifiers** and the metrics that keep them honest (Cluster 4: NB, LR, SVM, the accuracy trap). Those classifiers still treat "car" and "automobile" as orthogonal — so we **place words in a continuous space where geometry encodes semantics** (Cluster 5: distributional hypothesis → LSA, Word2Vec). But many tasks need a *label per token* with neighbours mattering, so we add **explicit sequence structure** (Cluster 6: HMM, Viterbi, POS, NER, dependencies). HMMs use discrete-tag transition tables; we replace them with **learned continuous functions and recurrent hidden states** (Cluster 7: FFN → RNN → LSTM/GRU → encoder-decoder, ending on the fixed-context bottleneck). Attention dissolves that bottleneck and, once it exists, lets us drop recurrence entirely — **transformers** (Cluster 8). Everything modern is then a fine-tuned transformer wired to a task head: classification, generation, summarization, extractive QA, RAG (Cluster 9). Reading the cluster intros in order should let you tell that story without notes — that's the test.

## Clusters

1. [[01-foundations]] — Foundations & history — lectures 01–03 — exam weight: low (cheap MCQ points)
2. [[02-text-representation]] — Tokens, bag-of-words, n-grams — lectures 04–05 — exam weight: high (Code 1)
3. [[03-patterns-and-retrieval]] — Regex, IR, TF-IDF, cosine — lectures 06–07 — exam weight: very high (Exercise 2 + regex MCQs)
4. [[04-classical-classifiers]] — Naïve Bayes, logistic regression, SVM, evaluation — lectures 09, 10, 12 — exam weight: very high (Quiz II spine)
5. [[05-word-meaning-in-vector-space]] — Distributional semantics, LSA, LDA, Word2Vec, PMI — lecture 13 — exam weight: high
6. [[06-sequence-labeling]] — HMM, Viterbi, POS, NER, dependencies — lectures 14–15 — exam weight: very high (Exercise 3)
7. [[07-neural-sequence-models]] — FFN → RNN → LSTM/GRU — lectures 16–18 — exam weight: high (Quiz IV first half)
8. [[08-attention-and-transformers]] — Attention, self-attention, multi-head, transformers — lecture 19 — exam weight: very high (Exercise 1 + Quiz IV second half)
9. [[09-modern-applications]] — Sentiment, classification, generation, QA, RAG — lectures 11, 24 — exam weight: very high (Code 2)

## Concept index

Alphabetical map of every concept note → its home cluster. Use this when you're reviewing a single concept and want to know which family it lives in.

- [[abstract-meaning-representation]] → cluster 1
- [[accuracy-trap]] → cluster 4
- [[activation-function]] → cluster 7
- [[attention]] → cluster 8
- [[backpropagation]] → cluster 7
- [[bag-of-words]] → cluster 2
- [[bayes-formula]] → cluster 4
- [[bptt]] → cluster 7
- [[causal-masking]] → cluster 8
- [[confusion-matrix]] → cluster 4
- [[context-free-grammar]] → cluster 1
- [[cosine-similarity]] → cluster 3
- [[cross-attention]] → cluster 8
- [[cross-entropy]] → cluster 7
- [[decision-threshold]] → cluster 4
- [[dependency-parsing]] → cluster 6
- [[distributional-hypothesis]] → cluster 5
- [[early-symbolic-nlp]] → cluster 1
- [[eliza]] → cluster 1
- [[embedding-matrix]] → cluster 5
- [[encoder-decoder]] → cluster 7
- [[evaluation-metrics]] → cluster 4
- [[extractive-question-answering]] → cluster 9
- [[formal-vs-natural-language]] → cluster 1
- [[generative-vs-discriminative]] → cluster 4
- [[gru]] → cluster 7
- [[hidden-markov-model]] → cluster 6
- [[hmm-viterbi]] → cluster 6
- [[huggingface-summarization]] → cluster 9
- [[huggingface-text-classification]] → cluster 9
- [[hyperspace-analogue-to-language]] → cluster 5
- [[language-ambiguity]] → cluster 1
- [[laplace-smoothing]] → cluster 4
- [[latent-semantic-analysis]] → cluster 5
- [[lda]] → cluster 5
- [[log-odds]] → cluster 4
- [[logistic-regression]] → cluster 4
- [[lstm]] → cluster 7
- [[maximum-entropy]] → cluster 4
- [[morpheme-and-subword-tokenization]] → cluster 2
- [[multi-head-attention]] → cluster 8
- [[multilayer-perceptron]] → cluster 7
- [[n-gram-model]] → cluster 2
- [[naive-bayes]] → cluster 4
- [[named-entity-recognition]] → cluster 6
- [[negative-sampling]] → cluster 5
- [[nlp-definition]] → cluster 1
- [[nlp-libraries]] → cluster 2
- [[nlp-pipeline]] → cluster 1
- [[nlp-understanding-vs-generation]] → cluster 1
- [[one-hot-encoding]] → cluster 2
- [[part-of-speech-tagging]] → cluster 6
- [[perceptron]] → cluster 7
- [[pointwise-mutual-information]] → cluster 5
- [[positional-encoding]] → cluster 8
- [[recurrent-neural-network]] → cluster 7
- [[regex-greedy-vs-lazy]] → cluster 3
- [[regex-lookarounds]] → cluster 3
- [[regular-expressions]] → cluster 3
- [[relevance-feedback]] → cluster 3
- [[retrieval-augmented-generation]] → cluster 9
- [[scaled-dot-product-attention]] → cluster 8
- [[self-attention]] → cluster 8
- [[semantic-analysis]] → cluster 1
- [[semantic-gap]] → cluster 1
- [[sentiment-analysis]] → cluster 9
- [[sigmoid]] → cluster 4
- [[skip-gram-and-cbow]] → cluster 5
- [[softmax]] → cluster 7
- [[stemming-vs-lemmatization]] → cluster 2
- [[stop-words]] → cluster 2
- [[support-vector-machine]] → cluster 4
- [[term-document-matrix]] → cluster 2
- [[text-classification]] → cluster 9
- [[text-generation-sampling]] → cluster 9
- [[tf-idf]] → cluster 3
- [[tokenization]] → cluster 2
- [[transformer]] → cluster 8
- [[turing-test]] → cluster 1
- [[vanishing-exploding-gradients]] → cluster 7
- [[vocabulary-mismatch]] → cluster 3
- [[word-embeddings]] → cluster 5
- [[word2vec]] → cluster 5

## Gaps found

Topics that the exam blueprint or lectures reference but for which no `20-Concepts/` note currently exists:

- *None.* Previously-flagged gaps (LDA, PMI) were resolved on 2026-05-03 by creating [[lda]] and [[pointwise-mutual-information]] and adding both to cluster 5. Sampling strategies are covered in [[text-generation-sampling]] (all four — greedy, beam, temperature, top-k — with Quiz IV exam framing); cross-entropy as the LR loss is covered in [[cross-entropy]] (line: "Training logistic regression… minimizes cross-entropy with observed labels — equivalently, maximum likelihood").

## Coverage check

- **84** concept notes in vault, **84** placed in clusters, **0** orphans
- **19** lecture notes in vault, **19** referenced by at least one cluster
- Every lecture in the corpus appears as `covers-lectures` in exactly one cluster (lecture 18 covers LSTM/GRU/encoder-decoder, all in cluster 7; lecture 11 sentiment-analysis lives in cluster 9 with the modern applications because the prof frames sentiment as a discourse signal applied via HF, per Quiz III Q2 / Q16.B)
- Cluster sizes: 12, 9, 8, 13, **10**, 5, 12, 8, 7 — cluster 4 (classical classifiers + evaluation) and cluster 7 (neural sequence models) sit at the upper end because they are the *spine* of the course; everything before cluster 5 builds toward them, everything after cluster 7 reacts to them
