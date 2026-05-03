---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 04 - Adnvaced Concepts.pdf
created: 2026-05-02
---

# TF-IDF

**Term Frequency–Inverse Document Frequency**: a weighting scheme that reweights raw [[bag-of-words|BoW]] counts to **highlight terms that are discriminative** within a corpus while down-weighting common words.

The blueprint flags this as **very high weight** — Exercise 2 of the mock exam was a TF-IDF computation on a 3-document corpus.

## Formula (on the sheet)

$$w_{i,j} = \mathrm{TF}_{i,j} \cdot \log\!\left(\frac{N}{n_i}\right) = \mathrm{TF}(w_i, d_j) \cdot \mathrm{IDF}(w_i)$$

| Quantity | Definition |
|---|---|
| $\mathrm{TF}(w, d)$ | count of $w$ in $d$ |
| $\mathrm{DF}(w) = n_i$ | number of documents containing $w$ |
| $\mathrm{IDF}(w) = \log(N / \mathrm{DF}(w))$ | inverse document frequency |
| TF-IDF | TF · IDF |

## Two intuitions (slide 74)

1. A term that appears **frequently in one document** likely reflects that document's meaning strongly → high TF.
2. A term that appears **in too many documents** carries little discriminative information → low IDF.

The **logarithm** dampens IDF so that frequent-everywhere terms get small but non-zero weight.

> Mock Q3: "TF-IDF assigns higher weight to **rare but informative terms**." (correct)

## Why IDF down-weights very common words

Common words (*the*, *of*) appear in almost every document, so $N/n_i \to 1$ and $\log(\ldots) \to 0$. They **carry little discriminative information** (Quiz II.M2 Q15) — the same intuition behind [[stop-words|stop-word removal]], expressed continuously.

## Worked example (mock Q27 / slide 76)

Corpus: D1 "nlp is fun", D2 "nlp is powerful", D3 "learning nlp is fun". $N = 3$.

For "fun":
- $\mathrm{TF}(\text{fun}, D_3) = 1$ — appears once in $D_3$
- $\mathrm{DF}(\text{fun}) = 2$ — appears in $D_1$ and $D_3$
- $\mathrm{IDF}(\text{fun}) = \log(3/2) \approx 0.405$
- $\mathrm{TF\text{-}IDF}(\text{fun}, D_3) = 1 \cdot \log(3/2) = \log(3/2)$

For "models" appearing in all 3 of `[D1: students study language models, D2: students study models, D3: language models]`: $\mathrm{IDF} = \log(3/3) = 0$ — that word is so common it gets **zero weight**.

## Why it doesn't solve everything

TF-IDF still operates at the **surface level**: it assumes words are **independent features** (mock Q22 / Q12 logic). Two texts about the same topic using different vocabularies appear unrelated; synonyms never co-occur and stay disconnected. **Vocabulary mismatch** (Quiz II.M2 Q11) limits classical IR recall.

> TF-IDF improves representation, but it does not introduce structure into the space. It sharpens the coordinates, but does not change the geometry.

This motivates [[latent-semantic-analysis|LSA]] (global latent topics via SVD) and [[hyperspace-analogue-to-language|HAL]] (local context via co-occurrence) — and ultimately dense [[word-embeddings]].

## Pairing with cosine

[[cosine-similarity]] is the standard similarity over TF-IDF vectors, because cosine **removes document-length effects**. The TF-IDF + cosine combination is the foundation of classical [[information-retrieval-ranking|vector-space retrieval]].
