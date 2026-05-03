---
type: study-guide-cluster
course: NLP
cluster: "03-patterns-and-retrieval"
theme: "Regex, IR, TF-IDF, cosine"
prerequisites: [01-foundations, 02-text-representation]
covers-concepts:
  - regular-expressions
  - regex-lookarounds
  - regex-greedy-vs-lazy
  - information-retrieval-ranking
  - vocabulary-mismatch
  - relevance-feedback
  - tf-idf
  - cosine-similarity
covers-lectures:
  - lecture-06-regular-expressions
  - lecture-07-information-retrieval
exam-weight: high
---

# Cluster 3: Pattern matching and retrieval

> **The story of this cluster in one sentence.** Cluster 2 turned documents into count vectors; this cluster gives us the two practical things we now want to do with text — *find* a string that matches a pattern (regex) and *rank* documents by relevance to a query (IR + TF-IDF + cosine).

## Why this cluster exists

Once text is a discrete object you can count, two enterprise-shaped problems become tractable: pattern extraction (telephone numbers, email addresses, ING-words for a stylometric study) and document retrieval (the search engine problem — given a query, return the most relevant docs). Both turn into substantial exam content. **Regex is a recurring MCQ block** (mock Q4; Quiz II Q4, Q13, Q19; Quiz II.M2 Q4, Q12; Quiz II.M3 Q11) targeting the constructs that *aren't* on the formula sheet's symbol table — lookbehinds, lookaheads, lazy quantifiers. **TF-IDF is Exercise 2** of the mock — 10 points of by-hand computation on a 3-document corpus. **Cosine similarity** is the standard retrieval scoring function and gets numerical MCQs (Quiz III Q1).

**Prerequisites you should feel solid on:**

- [[bag-of-words]] — TF-IDF reweights BoW counts; cosine similarity scores BoW vectors
- [[term-document-matrix]] — IR + TF-IDF live in this matrix; understand the axes
- [[stop-words]] — removing them is the *discrete* version of what TF-IDF does *continuously*

## The arc

### 1. [[regular-expressions]] — patterns, not vectors

Regex is the *other* way to interrogate text — instead of building a count vector and doing math, you write a **declarative pattern** that matches strings (or doesn't). The full symbol table (`.`, `\d`, `\w`, `\s`, `^`, `$`, `*`, `+`, `?`, `{n,m}`, `[a-z]`, `()`, `|`) is on the formula sheet, so the exam doesn't test memorization of the basics — it tests whether you can *read* a pattern under time pressure (mock Q4: `\b\w+ing\b` matches what?). The two MCQ-trap categories are *what's not on the sheet* — and that's the next two concepts.

### 2. [[regex-lookarounds]] — zero-width assertions

Lookarounds let a pattern match a position based on what's around it *without consuming characters*: `(?<=X)Y` matches `Y` only if preceded by `X`; `(?!Z)` matches a position not followed by `Z`. The "zero-width" part is the trap — the matched substring doesn't include the lookaround context. Quiz II Q4, II.M2 Q4 / Q12, II.M3 Q11 all live here. Memorize the four flavors: `(?=X)` `(?!X)` `(?<=X)` `(?<!X)`.

### 3. [[regex-greedy-vs-lazy]] — quantifier appetite

Regex quantifiers (`*`, `+`, `?`, `{n,m}`) are **greedy by default** — they consume as much as possible while still allowing the overall match to succeed. Adding a `?` makes them **lazy** (`*?`, `+?`) — consume as little as possible. The exam-favorite trick: `.*?` in Python returns **the empty string** as its first match because zero characters is the smallest valid quantity (Quiz II Q19). This is the kind of MCQ that takes 10 seconds if you've internalized "lazy = match as little as the regex engine can get away with" and 5 minutes of guesswork otherwise.

### 4. [[information-retrieval-ranking]] — the search-engine problem

Given a corpus of documents and a query, return documents ranked by relevance. Classical IR's answer: represent both query and documents as **vectors over the vocabulary**, weight the entries somehow (TF-IDF), and score them by a similarity function (cosine). The whole pipeline is term-document matrix → TF-IDF reweight → query vectorization → cosine ranking. The exam tests it conceptually rather than mechanically (Quiz II Q3, Q12; Quiz II.M2 Q7, Q11, Q15) — what does TF-IDF *do for ranking*, what does cosine *normalize for*, what *limits* recall.

### 5. [[vocabulary-mismatch]] — the recall-killer

The Achilles heel of classical IR: a query says "car" and the document says "automobile" and the vector-space match returns *nothing*. There is no axis in the term-document matrix where "car" and "automobile" overlap, because each is a distinct dimension. **Vocabulary mismatch** is the single most exam-named limitation of bag-of-words IR (Quiz II.M2 Q11) — and it's the *exact* problem that distributional embeddings (Cluster 5) were invented to fix. Note this connection now; it's the backbone of the cluster-3 → cluster-5 bridge.

### 6. [[relevance-feedback]] — patching mismatch with user signal

If vocabulary mismatch is killing recall, one classical fix is to **let the user tell you which results were good** and use those to expand or reweight the query (Rocchio: push the query toward the centroid of relevant docs, away from non-relevant). This is a band-aid — it requires user labels and only addresses mismatch at query time — but it's a clean illustration of the broader idea that **representations can be updated from feedback**. Foreshadows ML training (Cluster 4) and contrastive learning more generally.

### 7. [[tf-idf]] — the weighting that makes ranking work

`TF-IDF(w, d) = TF(w, d) · log(N / DF(w))`. Two ideas in one formula: **frequent within a document means topical** (TF), **frequent across documents means uninformative** (low IDF). Common words ("the", "of") get IDF ≈ 0; rare words ("Heisenberg") get high IDF. This is **Exercise 2 of the exam** (10 pts) — the prof gives you a 3-document toy corpus and asks for `TF`, `DF`, `IDF`, `TF-IDF` by hand. The formula is on the sheet; what you have to bring is *fluency with `log(N / DF)` as a number on a 3-doc corpus*. Drill the worked example in the [[tf-idf]] note until you can do `log(3/2)` and `log(3/3) = 0` without thinking.

### 8. [[cosine-similarity]] — the scoring function that ignores length

`cos(x, y) = (x · y) / (‖x‖ ‖y‖)`. The dot product divided by the norms — geometrically, the cosine of the angle between the vectors. The reason cosine is the default IR similarity (and not, say, Euclidean distance) is that **it normalizes for document length**: a long document has bigger vector entries everywhere, but cosine cares only about *direction*. Numerical Quiz III Q1 / Q9: given two 2-D vectors, compute the cosine. Skill = compute dot product, two norms, divide. Practice this until it's 30 seconds.

## Connections worth seeing

- **TF-IDF is "smooth stop-word removal"** (cf. Cluster 2). A stop-word list is a hard cutoff (`is_stopword(w) ∈ {0, 1}`); `log(N / DF(w))` is a continuous version that smoothly down-weights words proportional to how common they are. When you see this, IDF stops being magic.
- **Cosine similarity reappears throughout the course.** Here it scores TF-IDF vectors. In Cluster 5 it scores Word2Vec vectors. In Cluster 8 the un-normalized cousin `Q · K^T` is the **attention score**. The pattern: **whenever you want "how related are these two things in vector space," reach for a dot product**, then decide whether to normalize for magnitude (cosine) or not (raw attention score, then softmax-normalized).
- **Vocabulary mismatch motivates Cluster 5.** Every dense-embedding method we'll see (Word2Vec, LSA) was created to give "car" and "automobile" *similar coordinates* so that a query for one retrieves documents about the other. Hold this thought; Cluster 5 is the answer.
- **Regex and the rest of this cluster are doing different jobs.** Regex matches *exact patterns*; IR + TF-IDF + cosine match *bags of words.* They aren't competitors — they're complementary tools for "do something useful with text." Don't try to unify them conceptually; they live on different shelves.

## Common confusions

- **Greedy vs. lazy regex** — greedy = match as much as possible; lazy = as little. `.*` greedy, `.*?` lazy. On `"abcabc"`, `.*?` first matches `""`, then `"a"`, etc.
- **Lookahead vs. lookbehind** — both are zero-width; `(?=X)` checks what *follows*; `(?<=X)` checks what *precedes*.
- **TF vs. DF vs. IDF** — TF counts within a single doc; DF counts how many *docs* contain the word; IDF = `log(N / DF)`.
- **Cosine vs. Euclidean similarity** — cosine ignores magnitude (normalizes for length); Euclidean penalizes magnitude differences. For text, you almost always want cosine.
- **Stop-word removal vs. TF-IDF** — both reduce the influence of common words; one is hard binary, the other is continuous. Doing both is fine but slightly redundant.

## Self-check (synthesis, not recall)

1. **TF-IDF assigns weight zero to a term appearing in every document.** What does this assumption *fail at* — i.e., when would a word that appears in every document actually carry useful information? Think about a corpus of customer-support emails.
2. The pattern `\b\w+ing\b` matches "running" but not "string". *Why?* Then: how would you change the pattern to match only words ending in `-ing` that are *not* preceded by `r` (so "ring" would match but "string" would not)?
3. **Cosine similarity ignores vector magnitude.** Construct two 2-D vectors that have a high cosine similarity but very different Euclidean distances. What does this scenario look like in document-retrieval terms?
4. **Vocabulary mismatch limits recall** (you ask for "car," you don't find "automobile"). Pick *two* methods from later clusters that address this — one classical (Cluster 5), one neural (Cluster 8) — and explain in one sentence each how they fix it.
5. **Looking forward to Cluster 4:** TF-IDF gives each document a vector. We now have a labeled corpus (positive vs. negative reviews) and want to *learn a classifier*. What's the minimal probabilistic assumption that lets us turn the BoW vector into a label? (Hint: the next cluster names two of them.)

## If you have 10 minutes

1. [[tf-idf]] worked example — compute TF-IDF for `D₁ = "nlp is fun"`, `D₂ = "nlp is powerful"`, `D₃ = "learning nlp is fun"` from memory; this is the Exercise 2 drill
2. [[regex-greedy-vs-lazy]] — internalize that `.*?` matches the empty string first
3. [[regex-lookarounds]] — memorize the four lookaround syntaxes; these are guaranteed MCQ points
4. [[cosine-similarity]] formula — practice on a 2-D vector pair until you can compute it in 30 seconds

## Next cluster

→ [[04-classical-classifiers]] — we now have weighted document vectors. The next move is **labeling** them: spam vs. ham, positive vs. negative, this topic vs. that. Cluster 4 introduces the two great classifier families (generative Naïve Bayes and discriminative logistic regression) and the metrics for telling whether they actually work — including the trap of judging an imbalanced classifier by accuracy alone.
