---
tags: [flashcards, NLP]
---

# Lecture 11 — Sentiment Analysis (flashcards)

What kind of NLP task is sentiment analysis?
?
A **text classification** task — assign a polarity label (positive/negative/neutral) or numeric score to a document, sentence, or aspect.

Write the lexical / dictionary-based sentiment score formula.
?
$S(d) = \dfrac{1}{n}\sum_{i=1}^n \text{score}(w_i)$ — average per-word polarity scores from a sentiment lexicon over the document's tokens.

What's the simplest learned approach to sentiment?
?
**Logistic regression on BoW or TF-IDF features** — coefficient sign reveals each word's polarity contribution.

How is sentiment typically interpreted when analyzed over a sequence of documents or time points?
?
As a **dynamical signal evolving across observations** — not a static lexical property and not a topic distribution.

What can sentiment signals over time reveal in discourse analysis?
?
**Narrative shifts in discourse** — how the framing or mood of a story changes through the corpus.

What does sentiment-over-time correlate with in financial news?
?
**Discourse dynamics in narratives** — sentiment trajectories that may correlate with market signals.

What does a positive logistic-regression coefficient for a word mean for sentiment classification?
?
The word **pushes the document toward the positive polarity class** — odds of being positive are multiplied by $e^{w}$ when the word is present.

What's the main limitation of lexical sentiment scoring?
?
It **ignores word order and negation** — "not good" and "good" produce similar averages without negation handling. Sarcasm, irony, and domain shift also break the lexical approach.

Why might a learned classifier outperform a lexical baseline for sentiment?
?
Because it learns **which words signal which polarity from data** — including domain-specific polar terms (e.g. financial: "downgrade", "earnings beat") that generic lexicons miss.

What downstream NLP techniques address the limitations of BoW sentiment?
?
Dense [[word-embeddings|word embeddings]] (Word2Vec, GloVe) and contextual transformer-based models — they capture word order, negation scope, and contextual meaning that BoW cannot.
