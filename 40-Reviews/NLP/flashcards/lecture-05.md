---
tags: [flashcards, NLP]
---

# Lecture 05 — Python Modules (flashcards)

What are the two fundamentally different perspectives on natural language?
?
**Statistical** (language as observable data; regularities inferred from usage) and **symbolic** (language as a structured system governed by explicit rules).

What did Chomsky's formal language theory show about a finite set of grammatical rules?
?
That they can generate an **infinite number of sentences** through recursive rewriting from a start symbol.

What does a context-free grammar primarily concern itself with?
?
**Structural well-formedness** — whether a sentence can be derived from the grammar — not semantic interpretation or empirical usage.

What is the semantic gap?
?
The mismatch between **syntactic structure** and **meaning** — recovering structure does not automatically yield meaning. Similar structures can differ in meaning; different structures can express the same meaning.

What is Abstract Meaning Representation (AMR)?
?
A representation that encodes sentence meaning as a **graph of concepts and relations**, abstracting away from surface syntax so paraphrases map to the same structure.

Which library is most commonly used for educational NLP experiments?
?
**NLTK.** Reflects a classical, linguistically motivated view; exposes tokenization, tagging, parsing, corpora explicitly.

How is spaCy best described?
?
As **an industrial NLP pipeline** — fast, end-to-end, with pretrained models; emphasizes robustness over transparency.

What is gensim primarily associated with?
?
**Word embeddings** and vector-space / topic models (BoW, TF-IDF, LSA, LDA, Word2Vec at scale).

What do HuggingFace Transformers mainly provide?
?
**Pretrained language models** — high-level interfaces to large neural models that embed syntactic and semantic regularities implicitly.

Which task is NOT typically handled directly by spaCy?
?
**Training transformers.** spaCy uses pretrained ones; training transformers is HuggingFace's domain.

What does `doc = nlp(text)` do conceptually in spaCy?
?
**Processes text through an NLP pipeline** — tokenization, tagging, parsing, NER, etc.

What does `word_tokenize('NLP is fun')` return?
?
A **list of tokens** (e.g. `['NLP', 'is', 'fun']`).

Why does `tokens = nltk.word_tokenize` not tokenize any text?
?
**The function is not called** — that line assigns the function object to `tokens`, it doesn't invoke the tokenizer.

Why do modern NLP libraries reduce the need for manual coding?
?
Because they **provide reusable pipelines** — not because they eliminate ambiguity, avoid computation, or guarantee understanding.

What's the key trade-off when adopting an NLP library?
?
You **adopt its view** of what matters in language processing — its assumptions about representation, preprocessing, and pipeline structure travel with the library.

What's the conceptual difference between NLTK and spaCy?
?
NLTK exposes the building blocks (tokenize, tag, parse) for learning and inspection; spaCy hides them in an end-to-end pipeline that gains efficiency at the cost of transparency.

What's the common limitation of symbolic / rule-based approaches that pushed NLP toward statistical methods?
?
They are **interpretable but require strong assumptions** and **struggle with ambiguity, noise, and scale**.
