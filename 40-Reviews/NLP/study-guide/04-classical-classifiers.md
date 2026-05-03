---
type: study-guide-cluster
course: NLP
cluster: "04-classical-classifiers"
theme: "Naïve Bayes, logistic regression, SVM, evaluation"
prerequisites: [02-text-representation, 03-patterns-and-retrieval]
covers-concepts:
  - generative-vs-discriminative
  - bayes-formula
  - naive-bayes
  - laplace-smoothing
  - logistic-regression
  - sigmoid
  - log-odds
  - maximum-entropy
  - decision-threshold
  - support-vector-machine
  - confusion-matrix
  - evaluation-metrics
  - accuracy-trap
covers-lectures:
  - lecture-09-naive-bayes
  - lecture-10-logistic-regression
  - lecture-12-deepdive-svm
exam-weight: high
---

# Cluster 4: Classical classifiers and evaluation

> **The story of this cluster in one sentence.** Take the BoW + TF-IDF vector from Cluster 3, attach a label, and learn a function from vectors to labels — first the *generative* way (Naïve Bayes), then the *discriminative* way (logistic regression and SVM), then the metrics that tell you it isn't fooling you.

## Why this cluster exists

This is where NLP becomes machine learning. Cluster 3 gave us document vectors; Cluster 4 turns them into **decisions**. The two great families — **generative** (model `P(x, y)` and use Bayes' rule) and **discriminative** (model `P(y | x)` directly) — define a contrast that recurs in every later cluster (RNN classifiers are discriminative, language models are generative). Evaluation is folded in here because *the same imbalanced-data trap shows up everywhere* and you need the vocabulary (precision, recall, F1, AUC) to talk about whether your classifier is actually doing anything.

The blueprint flags this as **very high weight**: Quiz II is essentially this cluster end-to-end (NB independence, LR sigmoid evaluations, log-odds, coefficient interpretation, accuracy trap). The formula sheet provides `σ(z)`, `z = wx + b`, NB's `P(y|x) ∝ P(y)·∏P(xᵢ|y)`, and the metrics — **what's not on the sheet is the meaning**: when does NB's independence assumption hurt? Why is `+1.5` a "strong positive coefficient"? Why is accuracy the wrong metric on imbalanced data? That's the memorization burden.

**Prerequisites you should feel solid on:**

- [[bag-of-words]] / [[tf-idf]] — the input vectors these classifiers consume
- [[cosine-similarity]] — for SVM and the geometric framing of "margin"
- [[language-ambiguity]] — to understand why a classifier built on word counts can never *fully* solve sentiment

## The arc

### 1. [[generative-vs-discriminative]] — the framing that organizes the cluster

Two ways to learn a classifier. **Generative**: model `P(x, y)` (the joint), use Bayes' rule to recover `P(y | x)`. **Discriminative**: model `P(y | x)` directly. Naïve Bayes is generative. Logistic regression is discriminative. The discriminative approach uses fewer assumptions (it doesn't model how the data was generated, only how to classify it), which usually wins on accuracy but **doesn't let you generate new data** the way a generative model can. Quiz II Q1 is a definition MCQ on this distinction; Quiz II.M3 Q12 tests it in disguise. Internalize the contrast — it shows up again as encoder vs. decoder in Cluster 7.

### 2. [[bayes-formula]] — the inversion that powers Naïve Bayes

`P(y | x) = P(x | y) · P(y) / P(x)`. The denominator is the same for every class, so for *classification* (pick the most likely class), the rule simplifies to `argmax_y P(x | y) · P(y)`. This is the engine inside NB and the link between the generative model `P(x | y)` (which you trained) and the prediction `P(y | x)` (which you want). Memorize the inversion — every generative classifier is a recipe for using Bayes' rule.

### 3. [[naive-bayes]] — the bold independence assumption

NB assumes `P(x | y) = ∏ᵢ P(xᵢ | y)` — features are **conditionally independent given the label**. For text, this means "given that the email is spam, the words in it are statistically independent of each other." This is *clearly false* (real text has co-occurrence structure), and yet NB still works astonishingly well on text classification — this is the ML lore worth knowing. The blueprint flags the *trap*: NB does **not** assume joint independence (`P(x) = ∏ P(xᵢ)`); it assumes **conditional** independence given `y` (Quiz II Q18, II.M3 Q12). Memorize the difference.

### 4. [[laplace-smoothing]] — the patch for unseen words

NB multiplies probabilities. If even one `P(xᵢ | y) = 0` (the word never appeared in any spam email), the entire product is zero — the classifier is *certain* it's not spam, no matter what the other words say. **Laplace smoothing** adds a small constant to every count: `P(xᵢ | y) = (count + α) / (total + α · |V|)`. Same idea as adding a tiny `ε` to a denominator: avoid the discrete zero. Conceptually parallel to subword tokenization in Cluster 2 — both are defenses against the **unseen-vocabulary problem**.

### 5. [[logistic-regression]] — discriminative, linear, the workhorse

LR models `P(y = 1 | x) = σ(w · x + b)`. The score `z = w · x + b` is linear in features; the sigmoid squashes it to `[0, 1]`. *What's linear in the features is the **log-odds**, not the probability* — Quiz II Q11 is exactly this trap. LR makes **no independence assumption** — it just fits the best linear decision boundary and lets the data speak. Trained by **minimizing cross-entropy** (= maximum likelihood) — *not* 0-1 error, *not* margin width (Quiz II Q9). The blueprint flags this as **very high weight**: Quiz II Q5–Q11, Q15–Q16, Q20 (and Model 2/3 mirrors) are essentially LR end-to-end.

### 6. [[sigmoid]] — the squashing function

`σ(z) = 1 / (1 + e^(-z))`. Maps `z ∈ ℝ` to `(0, 1)`, monotonic, S-shaped, `σ(0) = 0.5`. The exam asks you to evaluate `σ(z)` for small `z` values (Quiz II Q10, II.M2 Q5, II.M3 Q6) — memorize `σ(0) = 0.5`, `σ(1) ≈ 0.73`, `σ(2) ≈ 0.88`, and that they're symmetric: `σ(-z) = 1 - σ(z)`. Sigmoid also reappears as the LSTM gate activation in Cluster 7 — same function, different role.

### 7. [[log-odds]] — what LR is actually linear in

`log(p / (1 - p)) = z = w · x + b`. The probability isn't linear in `x`; the **logit** is. This unlocks LR's coefficient interpretation: a coefficient `wᵢ = +1.5` means "presence of feature `i` adds 1.5 to the log-odds of `y = 1`," equivalently "multiplies the *odds* by `e^1.5 ≈ 4.5`." This is the *meaning* the formula sheet doesn't give you and the exam keeps asking for (Quiz II Q15: "+1.5 coefficient ⇒ ?"; Quiz II.M2 Q20: "near-zero coefficient ⇒ ?").

### 8. [[maximum-entropy]] — why LR is "the least committed" classifier

LR can be derived as the **maximum-entropy distribution** consistent with observed feature expectations. In English: among all distributions that match the training data's feature averages, LR picks the *flattest* (highest-entropy) one — i.e., the one that assumes the least beyond what the data forces. This is why LR generalizes well: it doesn't invent structure the data didn't provide. The connection rarely shows up directly on the exam but it's the cleanest justification of *why* LR is the right default.

### 9. [[decision-threshold]] — the knob you actually tune

LR outputs a probability; the default rule "predict `1` if `σ(z) ≥ 0.5`" is just one choice. **Lowering the threshold** increases recall (you flag more positives) at the cost of precision (you flag more false positives). **Raising it** does the opposite. The exam tests the trade-off by asking "for class-imbalanced data, which threshold tuning helps?" The threshold lives at the intersection of LR (where probabilities come from) and evaluation (where the trade-off is measured) — it's *the* lever between the two halves of this cluster.

### 10. [[support-vector-machine]] — the geometric alternative

SVM finds the **hyperplane that maximizes the margin** between classes — the gap between the boundary and the nearest training example. The math gives the same linear `w · x + b` decision rule as LR, but trained to maximize a *geometric* margin instead of a likelihood. SVM is more robust to outliers (only the support vectors matter); LR uses every point. For NLP, SVMs were dominant pre-2015 (text is high-dim and linearly separable in BoW space — SVM thrives there). The exam treats SVM as the third member of the linear-classifier family; the contrast you need is "max-margin (SVM) vs. max-likelihood (LR)."

### 11. [[confusion-matrix]] — the 2×2 table everything is built on

For binary classification: TP / FP / FN / TN. Every metric in the next two concepts is a ratio of these four numbers. Memorize the layout cold — exam questions like "given TP=8, FP=2, FN=3, TN=87, compute precision and recall" reduce to copying numbers into the right slots. Quiz III Q15 is the canonical version.

### 12. [[evaluation-metrics]] — the right number for the task

Four core metrics, each on the formula sheet:

- **Accuracy** = `(TP + TN) / total` — fraction correct. Useless on imbalanced data.
- **Precision** = `TP / (TP + FP)` — of the things I flagged positive, what fraction really are?
- **Recall** = `TP / (TP + FN)` — of the things that really are positive, what fraction did I catch?
- **F1** = `2 · P · R / (P + R)` — harmonic mean; balances P and R; punishes either being near zero.

The exam tests these as *number computations* (Quiz III Q15) and as *which-metric-when* MCQs (Quiz II Q14, II.M2 Q19). For imbalanced classes (e.g. fraud detection — 1% positives), use precision/recall/F1, not accuracy. AUC summarizes ranking quality across all thresholds — useful when the threshold isn't fixed.

### 13. [[accuracy-trap]] — the warning the cluster ends on

If 99% of your test data is class A, a classifier that always predicts A gets 99% accuracy and *catches zero positives* (recall = 0). This is the **accuracy trap**, and Quiz II / Quiz III drill it relentlessly because it's a real production failure mode. The deeper lesson: **the metric you optimize for should match the cost structure of the task.** Same shape as the **ELIZA effect** in Cluster 1 — a system can ace a casual benchmark while doing something useless.

## Connections worth seeing

- **Naïve Bayes ∶ logistic regression ∷ ELIZA ∶ a real chatbot.** NB makes a strong (and false) independence assumption that's "good enough" to look impressive on simple data; LR drops the assumption and lets the data fit itself. Both pairs illustrate the trade between *cheap pretense* and *honest modelling*.
- **The accuracy trap (here) is the ELIZA effect (Cluster 1).** Both are warnings that **a number that looks good on a casual measurement may correspond to a system doing nothing useful.** Carry this pattern: "what metric / behavioural signal could I trivially game?" — it shows up again in BLEU for translation, perplexity for LMs, and HellaSwag for LLMs.
- **`σ(z)` here is the same `σ` that gates LSTM cells in Cluster 7 and the same building block of softmax (the multi-class sigmoid) used in attention in Cluster 8.** One function, three roles: classification probability, gate, soft selector.
- **Log-odds linearity is the same template as PMI in distributional semantics.** Both are "linear-in-log-space" representations of associations — LR for `feature → label`, PMI for `word → context`. Word2Vec (Cluster 5) is famously equivalent to factorizing the PMI matrix; the connection runs deeper than it looks.
- **LR's maximum-entropy derivation foreshadows the MaxEnt principle in language modelling.** The same "fewest commitments consistent with constraints" logic underlies n-gram smoothing and modern model regularization.

## Common confusions

- **Generative vs. discriminative** — generative models `P(x, y)` (you can sample new data); discriminative models `P(y | x)` (you can only classify).
- **NB's independence assumption** — *conditional* on the label, not joint. "Given class, features are independent."
- **Linear in features (probability) vs. linear in features (log-odds)** — LR is the second, not the first.
- **SVM vs. LR** — same decision rule, different training objective: max-margin vs. max-likelihood.
- **Accuracy vs. F1** — accuracy is fine when classes are balanced; F1 is the imbalanced-data answer.
- **Precision vs. recall** — precision asks "of what I caught, how much was real?"; recall asks "of what was real, how much did I catch?"

## Self-check (synthesis, not recall)

1. **NB's independence assumption is empirically false for text** — words co-occur in obvious ways (the / is, learning / curve). Yet NB performs well. Why? What does this say about *the gap between a model's assumptions and its performance*?
2. **Logistic regression and Naïve Bayes are both linear classifiers in BoW features.** What's the *practical* reason a discriminative LR almost always outperforms NB on the same training data?
3. **The mock exam doesn't ask you to compute LR coefficients — it asks you to *interpret* them.** Given a sentiment classifier with `w_great = +2.1`, `w_terrible = -1.8`, `w_movie ≈ 0.0`: in plain English, what does each coefficient say about the corresponding word? Connect each to its impact on log-odds and odds.
4. **Pick a real-world task** (medical diagnosis, fraud detection, spam filtering) and decide which evaluation metric matters most. Justify your choice in one sentence and explain why accuracy alone would mislead.
5. **Looking back to Cluster 3:** TF-IDF gave us weighted vectors that emphasize rare informative words. Does NB or LR benefit *more* from TF-IDF weighting (vs. raw counts)? Hint: think about which classifier *implicitly* re-weights features through training.
6. **Looking forward to Cluster 5:** every classifier in this cluster treats words as independent feature dimensions. What problem from Cluster 3 (vocabulary mismatch) does *no* classifier here solve, no matter how well it's trained? What kind of input representation would?

## If you have 10 minutes

1. [[logistic-regression]] — sigmoid evaluations table + coefficient-interpretation table
2. [[naive-bayes]] — the conditional-independence formula and the *one-line* reason it's "naïve"
3. [[evaluation-metrics]] + [[accuracy-trap]] — the formulas (memorize TP/FP/FN/TN positions) and the imbalanced-data warning
4. [[generative-vs-discriminative]] — be able to say which family each classifier in this cluster belongs to

## Next cluster

→ [[05-word-meaning-in-vector-space]] — every classifier we just built treats "car" and "automobile" as orthogonal features. Cluster 5 fixes this by **placing words in a continuous vector space where geometric closeness = semantic similarity** — Word2Vec, LSA, the distributional hypothesis. The vocabulary-mismatch wound from Cluster 3 finally gets stitched.
