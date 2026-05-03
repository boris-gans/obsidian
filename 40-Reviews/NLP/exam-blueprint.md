---
tags: [exam-blueprint, NLP]
course: NLP
created: 2026-05-02
sources:
  - mock-exams/Final Exam - proposal.pdf
  - mock-exams/Final Exam - proposal - Solutions.pdf
  - quizzes/quiz I - Solutions.pdf
  - quizzes/quiz II - Solutions.pdf
  - quizzes/quiz III - Model A - Solutions.pdf
  - quizzes/quiz III - Model B - Solutions.pdf
  - quizzes/quiz IV - Model A - Solutions.pdf
  - quizzes/quiz IV - Model B - Solutions.pdf
---

# NLP Exam Blueprint

Structured map of what the final tests, derived from the mock exam (with full solutions) and four quizzes covering Sessions 1–24. This is the high-leverage study target — every concept note, flashcard, and notebook skeleton in this course should trace back to a row here.

## Format

- **Duration:** 1 hour 20 minutes
- **Closed book**, but a 3-page formula sheet is stapled to the booklet (cannot unstaple)
- **Total: 100 points**, three sections:
  - Multiple Choice — 50 pts (25 questions × 2 pts)
  - Exercises — 30 pts (3 questions × 10 pts)
  - Code — 20 pts (2 questions × 10 pts)
- **Backup MCQs:** 5 backup questions appear at the end of the booklet, used in fixed order *only if* any of Q1–Q25 are invalidated. Backups apply class-wide once triggered.
- **Multi-select on an MCQ → that question is invalidated.** Only grid bubbles count; annotations on the question itself are ignored.
- No electronic devices.

## Marks distribution

| Section | Points | # Questions | Per-question |
|---|---|---|---|
| Multiple Choice | 50 | 25 | 2 |
| Exercise 1 | 10 | 1 | 10 |
| Exercise 2 | 10 | 1 | 10 |
| Exercise 3 | 10 | 1 | 10 |
| Code 1 (predict-output) | 10 | 1 | 10 |
| Code 2 (fill-in-blanks) | 10 | 1 | 10 |
| **Total** | **100** | **30 graded** | — |

Average pace ≈ 2:40 per question. MCQs should be ≤1 min each to leave time for the 10-pt exercises and code.

## Recurring patterns

### Multiple Choice (25 × 2 pts)

Recurring shapes, with mock + quiz examples:

- **Definition** — "X is best described as…" / "X is…": mock Q1, Q11, Q19, Q22; Quiz I Q1, Q3, Q4, Q11, Q21
- **True / False (which is FALSE / NOT correct)**: mock Q2, Q16; Quiz II Q18, Q18.M2, Q18.M3; Quiz IV Q2, Q11
- **Best-describes / formal description**: mock Q3, Q7, Q8, Q11, Q22; Quiz I Q9; Quiz II Q1, Q3
- **Contrast / comparison** (X vs Y): mock Q23 (one-hot vs dense), Q24 (RNN/Transformer vs BoW), Q25 (dense vs sparse generalization); Quiz III Q19–Q20 (POS vs NER)
- **Applied syntax — regex**: mock Q4 (`\b\w+ing\b`); Quiz II Q4 (negative lookbehind), Q13 (lazy quantifier), Q19 (lazy `.*?`); Quiz II.M2 Q4 (positive lookbehind), Q12 (zero-width assertions); Quiz II.M3 Q11 (negative lookbehind for digits)
- **Numerical computation** — answer is a number derived from a small calculation:
  - Cosine similarity on 2D vectors (Quiz III Q1)
  - Embedding-matrix size `V × d` (Quiz III Q6)
  - Sigmoid evaluation `σ(z)` for given z (Quiz II Q10, Q5.M2; Quiz II.M3 Q6)
  - Softmax weight on small score vector (Quiz IV Q6.M-B, Q6.M-A)
  - Precision / recall from raw counts (Quiz III Q15)
  - HMM contribution `P(t|t_prev)·P(w|t)` (Quiz III Q12)
  - Trigram MLE counts (Quiz I Q30, Q31, Q32)
  - Token-count counting (Quiz I Q25, Q26)
  - Entropy `−Σ p log p` for uniform binary (Quiz II Q2)
- **HuggingFace / library specifics**: mock Q18 (`num_labels`), Q19 (extractive QA), Q20 (padding/truncation purpose); Quiz I Q35–Q44 (NLTK / spaCy / gensim / HF)

### Exercises (3 × 10 pts)

The mock confirms the three exercise slots map to three canonical algorithmic-compute formats. Expect one of each:

| Slot | Pattern | Mock instance | Inputs | Algorithm steps tested |
|---|---|---|---|---|
| Ex. 1 | **Attention with 2×2 matrices** | Q26 | Q, K, V given as small matrices | (a) compute QK^T, (b) softmax → α, (c) output αV |
| Ex. 2 | **TF-IDF on a 3-document corpus** | Q27 | Plain-English mini-corpus | TF, DF, IDF = log(N/DF), TF-IDF |
| Ex. 3 | **Viterbi on a 2-state HMM** | Q28 | Initial / transition / emission tables | Initialization δ₁, recursion (max over predecessors), backtracking |

These three slots are stable. The course manifest's "exercises are micro-scale" warning is reinforced — every exercise uses 2×2 / 3-doc / 2-state inputs that can be computed by hand using the formula sheet.

### Code (2 × 10 pts)

Two slots, distinct patterns:

| Slot | Pattern | Mock instance | What's tested |
|---|---|---|---|
| Code 1 | **Predict-output** on a Python snippet using a classical NLP idiom | Q29 | Read code → write exact `print()` output. The mock used `from collections import Counter` to build a TF matrix. Source notebook: `03_BoW_modelling.ipynb`. |
| Code 2 | **Fill-in-blanks** on a HuggingFace pipeline + explain the NLP task | Q30 | The mock used the QA pipeline. Blanks targeted: tokenizer/model `from_pretrained(...)`, `return_tensors="pt"`, `with torch.no_grad():`, `outputs.start_logits`, `outputs.end_logits`. Source notebook: `22_Question_Answering.ipynb`. |

The fill-in-blanks pattern requires reproducing the canonical 5–15 line skeleton from memory. The notebooks listed under "What the formula sheet does NOT provide" are the source — concept notes for those notebooks must include the skeleton verbatim.

## Topic coverage map

Each topic → which question types touch it → traceable lecture / notebook source. The "weight" column synthesizes how often the topic shows up across mock + quizzes.

| Topic | MCQ presence | Exercise | Code | Lecture | Notebook | Weight |
|---|---|---|---|---|---|---|
| NLP definition / ambiguity / formal vs natural | Quiz I Q1–Q4, Q8 | — | — | Session 01, 03 | — | low |
| ELIZA / Turing test / ELIZA effect | Quiz I Q13, Q16–Q23 | — | — | Session 02 | — | low |
| Tokenization (purpose, output type, count) | mock Q1; Quiz I Q25, Q41–Q43; Quiz I Q5 (lemmatization) | — | — | Session 04 | — | medium |
| Bag-of-Words assumptions & limits | mock Q2, Q24; Quiz I Q27–Q29 | — | — | Session 04 (intro), 08 | `03_BoW_modelling.ipynb` | high |
| n-gram MLE (unigram / bigram / trigram) | Quiz I Q30–Q34 | — | — | Session 04, 08 | — | medium |
| Python NLP libraries (NLTK / spaCy / gensim / HF) | Quiz I Q35–Q44; mock Q18 | — | (Code 2 uses HF) | Session 05 | all notebooks | medium |
| Regular expressions (basic + lookaround + lazy) | mock Q4; Quiz II Q4, Q13, Q19; Quiz II.M2 Q4, Q12, Q13; Quiz II.M3 Q11 | — | — | Session 06 | — | high |
| Information retrieval / ranking / vocabulary mismatch | Quiz II Q3, Q12; Quiz II.M2 Q7, Q11, Q15; Quiz II.M3 Q3, Q13, Q16, Q18 | — | — | Session 07 | — | high |
| TF-IDF (definition, weighting rationale, cosine) | mock Q3; Quiz I (implicit); Quiz II Q3 | **Ex. 2** (mock Q27) | (Code 1 builds TF matrix) | Session 07, 08 | `03_BoW_modelling.ipynb` | very high |
| Cosine similarity (formula + numerical) | mock Q12; Quiz III Q1, Q9 | — | — | Session 07, 13 | — | high |
| Naïve Bayes (independence assumption, generative) | mock Q21; Quiz II Q1 (generative), Q18; Quiz II.M3 Q12 | — | — | Session 09 | `05_Naive_Bayes_Classifier.ipynb` | high |
| Logistic regression (z, σ, log-odds, training, coefficient interpretation) | Quiz II Q5–Q11, Q15–Q16, Q20; Quiz II.M2 Q5, Q9, Q18, Q20; Quiz II.M3 Q5, Q6, Q10, Q15, Q19, Q20 | — | — | Session 10 | `06_Sentiment_Analysis.ipynb` | very high |
| Evaluation metrics (acc / prec / rec / F1 / AUC / ROC / accuracy trap) | Quiz II Q6, Q14, Q17; Quiz II.M2 Q14, Q17, Q19; Quiz II.M3 Q4, Q9, Q14, Q17; Quiz III Q15 | — | — | Sessions 09–11, 15 | — | very high |
| Sentiment as time-series / discourse signal | Quiz III Q2, Q16 (Model B) | — | — | Session 11 | `06_Sentiment_Analysis.ipynb` | medium |
| Sparse vs dense embeddings; one-hot vs dense | mock Q23, Q25 | — | — | Session 13 | `08_Word_Embeddings.ipynb` | high |
| Word2Vec (CBOW / skip-gram / context prediction) | mock Q6; Quiz III Q7–Q9, Q18; Quiz III.B Q7, Q8, Q9, Q18 | — | — | Session 13 | `09_Word2Vec_Embedding.ipynb` | high |
| Negative sampling | Quiz III Q7, Q18; Quiz III.B Q7, Q18 | — | — | Session 13 | `09_Word2Vec_Embedding.ipynb` | medium |
| Embedding matrix `V × d` size + lookup | Quiz III Q6, Q6.B; Quiz IV Q13, Q13.B | — | — | Session 13, 16 | — | medium |
| LSA / SVD `X = UΣV^T` | mock Q7; Quiz III Q3, Q4; Quiz III.B Q3, Q4 | — | — | Session 13 | — | medium |
| LDA (topics, Dirichlet prior, document-topic, topic-word) | mock Q8; Quiz III Q5, Q17; Quiz III.B Q5, Q17 | — | — | Session 13 | — | medium |
| HMM / Viterbi (init / recursion / backtrack / argmax) | mock Q9; Quiz III Q10, Q11, Q19; Quiz III.B Q10, Q11, Q19 | **Ex. 3** (mock Q28) | — | Session 14 | `10_Part_of_Speech_tagging.ipynb` | very high |
| POS tagging (hidden states, evaluation, errors) | mock Q9; Quiz III Q13; Quiz III.B Q13 | — | — | Session 14 | `10_Part_of_Speech_tagging.ipynb` | high |
| Named Entity Recognition (boundaries + types, sequence labeling) | mock Q10; Quiz III Q14, Q15, Q20; Quiz III.B Q14, Q15, Q20 | — | — | Session 15 | `11_Named_Entity_Recognition.ipynb` | high |
| Dependency parsing output | mock Q11 | — | — | Session 14 (likely) | — | low |
| Neural networks basics (neuron, layer, sigmoid, softmax, cross-entropy) | Quiz IV (formula header); foundational for Quiz IV | — | — | Session 16 | — | medium |
| RNN (hidden state role, vanishing / exploding gradients, BPTT) | mock Q13; Quiz IV Q1–Q2, Q1.B–Q2.B | — | — | Session 17 | `16_Elman_RNNs.ipynb` | high |
| LSTM / GRU (forget / input / cell / reset / update gates) | Quiz IV Q3, Q4, Q3.B, Q4.B | — | — | Session 18 | — | medium |
| Encoder–decoder fixed-context bottleneck → attention motivation | Quiz IV Q5, Q5.B | — | — | Session 18, 19 | — | medium |
| Attention scores `S = QK^T` / softmax / α / output | mock Q14, Q15; Quiz IV Q6–Q9, Q19; Quiz IV.B Q6–Q9, Q19 | **Ex. 1** (mock Q26) | — | Session 19 | — | very high |
| Scaled dot-product `√d_k` rationale | Quiz IV Q9; Quiz IV.B (implicit) | — | — | Session 19 | — | medium |
| Self-attention (V vectors, parallelism, quadratic complexity) | Quiz IV Q8, Q10, Q18; Quiz IV.B Q8, Q9, Q18 | — | — | Session 19 | — | high |
| Multi-head attention | mock Q17; Quiz IV Q11, Q11.B | — | — | Session 19 | — | medium |
| Positional encoding (why needed) | mock Q16; Quiz IV Q14, Q14.B | — | — | Session 19 | — | medium |
| Transformer (false statement: NOT sequential; long-range deps) | mock Q16, Q24 | — | — | Session 19 | — | high |
| `num_labels` / token IDs / attention mask / `start_logits` / `end_logits` | mock Q18, Q19 | — | **Code 2** (mock Q30) | Session 19 | `19_Text_Classification.ipynb`, `22_Question_Answering.ipynb` | very high |
| Padding & truncation purpose | mock Q20 | — | (Code 2 uses `return_tensors`) | Session 19 | `19_Text_Classification.ipynb` | medium |
| Causal masking / cross-attention / encoder-decoder | Quiz IV Q12, Q20; Quiz IV.B Q10, Q12, Q20 | — | — | Session 19 | — | medium |
| Generation: temperature / beam search / top-k | Quiz IV Q15, Q16, Q17; Quiz IV.B Q15, Q16, Q17 | — | — | Session 19+ (Beyond RNNs) | `20_Text_Generation.ipynb` | medium |
| Extractive QA (answer span; argmax start/end logits) | mock Q19 | — | **Code 2** (mock Q30) | Session 19+ | `22_Question_Answering.ipynb` | very high |
| Retrieval-based / RAG-style systems | mock Q22 | — | — | Session 24 | — | low |
| BoW with `Counter` (build vocab + TF matrix) | — | — | **Code 1** (mock Q29) | Session 04, 08 | `03_BoW_modelling.ipynb` | very high |
| Text generation pipeline | — | — | (potential Code) | — | `20_Text_Generation.ipynb` | medium |
| Text summarization pipeline | — | — | (potential Code) | — | `21_Text_Summarization.ipynb` | medium |
| Sentiment-analysis pipeline (HF) | — | — | (potential Code) | Session 11 | `06_Sentiment_Analysis.ipynb` | medium |

## Prof tells

- **Multi-select = 0 points.** If two bubbles are filled, that MCQ is dead — and triggers the backup MCQ machinery for the *whole class*.
- **5 backup MCQs** appear at the end. Used **in the order printed**, only when at least one of Q1–Q25 is invalidated for everyone. Don't skim past them — they exist as drop-in replacements.
- **Annotations on the question are ignored.** Grid bubbles only.
- **Closed book + stapled formula sheet you can't unstaple.** Keep the formula sheet flipped open beside the booklet for the whole exam — every exercise is solvable directly from formulas it provides.
- **No electronic devices.** Do exercise arithmetic by hand. Inputs are deliberately small (2×2, 3-doc, 2-state) so this is feasible.
- **Pace:** ~3 min/question on average; in practice MCQs should take ≤1 min so the 10-pt exercises and code each get ~5–7 min.

## What the formula sheet provides

Reproduced from the mock exam's appended formula sheet (pp. 11–13). **Do not waste flashcard slots on these — recognize them on the sheet, don't memorize.**

- **Vector representations:** dot product `x·y = Σ xᵢyᵢ`, norm `‖x‖ = √Σ xᵢ²`, cosine similarity `(x·y)/(‖x‖‖y‖)`
- **Bag-of-words / n-grams:** vocabulary, BoW vector, term count `c(w,d)`, binary BoW indicator, n-gram tuple, count `T−n+1`, bigram example, unigram model `P(w₁…w_T) = Π P(wₜ)`, bigram model `Π P(wₜ|wₜ₋₁)`, MLE estimate `c(wₜ₋₁,wₜ)/c(wₜ₋₁)`
- **TF-IDF:** TF, DF, **IDF = log(N/DF(w))**, TF-IDF = TF·IDF
- **Cross-entropy:** `−Σ yᵢ log ŷᵢ`; binary case `−[y log p + (1−y) log(1−p)]`
- **Logistic regression:** linear score `z = w·x + b`, sigmoid `σ(z) = 1/(1+e^−z)`, prediction `P(y=1|x) = σ(z)`, decision rule `y=1 if σ(z)≥0.5`
- **Naïve Bayes:** classification rule `P(y|x₁…x_n) ∝ P(y) Π P(xᵢ|y)`
- **HMM / Viterbi:** init `δ₁(i) = P(z₁=i)·P(x₁|z₁=i)`, recursion `δₜ(j) = maxᵢ[δₜ₋₁(i)·P(zₜ=j|zₜ₋₁=i)]·P(xₜ|zₜ=j)`, backtracking via stored argmax pointers, full δ-table layout
- **Transformer input:** `[CLS] question [SEP] context [SEP]`, token IDs `input_ids`, attention mask (`1=token, 0=padding`), output for QA = `start_logits, end_logits`
- **Word embeddings:** lookup `xw = E[w]`, similarity `cos(xw, xw')`
- **Softmax:** `softmax(xᵢ) = e^xᵢ / Σⱼ e^xⱼ`
- **Attention:** `Q = XW_Q`, `K = XW_K`, `V = XW_V`; scores `S = QK^T`; scaled scores `QK^T/√d_k`; weights `α = softmax(S)`; output `Attention(Q,K,V) = αV`; row-wise softmax `αᵢ = e^Sᵢ / Σⱼ e^Sⱼ`
- **RNN update:** `hₜ = tanh(W xₜ + U hₜ₋₁)`
- **LSA:** `X = UΣV^T`, dimensionality reduction by keeping top-k singular values, low-rank approximation `X_k = U_k Σ_k V_k^T`
- **LDA:** Dirichlet prior `Dir(α₁…α_K)`, expectation `E[θ_k] = α_k/Σα_j`, document-topic `θ_d ~ Dir(α)`, topic-word `φ_k ~ Dir(β)`, word generation `z ~ θ_d, w ~ φ_z`, posterior counts `θ_k ∝ n_k + α_k`, word distribution `φ_w ∝ n_w + β_w`
- **Evaluation metrics:** Accuracy `(TP+TN)/(TP+TN+FP+FN)`, Precision `TP/(TP+FP)`, Recall `TP/(TP+FN)`, F1 `2·PR/(P+R)`
- **Regular expressions** — full symbol table: `.`, `\d`, `\w`, `\s`, `^`, `$`, `*`, `+`, `?`, `n`, `n,m`, `[a-z]`, `( )`, `|`

## What the formula sheet does NOT provide

The actual memorization burden. **This is the high-leverage list — every flashcard's reason for existing.**

### Conceptual / definitional (MCQ-tested, no formula given)

- BoW assumption: word order ignored; consequences (loss of semantics, sparse high-dim vectors)
- Tokenization purpose, lemmatization vs stemming
- One-hot vs dense vector tradeoffs (dimensionality, semantic similarity, generalization)
- ELIZA / Turing test / ELIZA effect / "behaviour can mask understanding"
- Ambiguity in language, formal vs natural, syntactic vs semantic
- spaCy = industrial pipeline; NLTK = educational; gensim = word embeddings / topic models; HF = pretrained models. Library purposes are exam fodder.
- Generative (`p(x,y)`) vs discriminative (`p(y|x)`) distinction
- Naïve Bayes assumes **conditional independence of features given the label** (not joint independence)
- Logistic regression: log-odds linear in features; coefficient `+w` ⇒ odds multiplied by `e^w`; near-zero coefficient ⇒ no influence
- Accuracy trap on imbalanced data (majority-class predictor inflates accuracy with near-zero recall)
- Why AUC > accuracy when thresholds vary; ROC summarizes ranking quality
- Vocabulary mismatch is the main recall-limiter in classical IR
- Cosine similarity normalizes for document length

### Word embeddings & topic models

- Word2Vec architectures: CBOW vs **Skip-gram with negative sampling** (predict context vs predict target; negative sampling avoids full-vocab softmax cost)
- Negative samples drawn from a noise distribution over vocabulary
- Word2Vec is a **shallow neural network with one hidden layer**
- Distributional hypothesis: words in similar contexts → similar vectors
- LSA = SVD of term-document matrix → keep top-k singular values for dimensionality reduction
- LDA: documents are mixtures of topics; topics are distributions over words; Dirichlet prior controls **sparsity of topic proportions**
- Embedding matrix has **`V × d`** parameters; computed by hand for small V/d in MCQs

### Sequence models

- POS tagging = HMM where words are observed (visible chain), tags are hidden states; Markov assumption: tag depends only on previous tag
- NER = sequence labeling problem assigning entity labels to each token; evaluation requires correct **entity span + entity type** (not just span)
- Dependency parser output = **tree of syntactic relations** between words
- HMM Viterbi mechanics: Viterbi replaces summation over previous states with **max** (not sum); backpointer stores the argmax predecessor
- HMM probability contribution per step = `P(tag|prev_tag) · P(word|tag)` — multiply, not sum

### Neural / deep / transformer

- RNN hidden state encodes past info into a **fixed-size representation**
- BPTT: gradients are repeated compositions of Jacobians → vanishing / exploding for long sequences
- LSTM gates: forget `fₜ` scales previous cell state; input `iₜ` regulates new candidate; cell update `cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ`
- GRU reset gate controls how much past info is ignored in candidate; update gate combines past with candidate
- Encoder–decoder fixed context vector forces compression of entire input → motivated attention
- **Attention motivation: pairwise similarity Q·K, softmax → weights, weighted sum of V**
- Scaling by `√d_k` prevents large dot-products from saturating softmax
- Multi-head attention: each head uses **separate** Q/K/V projections, captures different relational subspaces; outputs concatenated then projected
- Self-attention has **quadratic complexity O(L²)** in sequence length
- **Transformers process tokens in parallel** (false-statement traps say "strictly sequentially"); positional encoding restores order
- Causal masking in decoder prevents attending to future tokens
- Cross-attention lets decoder attend to encoder representations
- Sampling: temperature ↑ flattens distribution (more diverse); ↓ sharpens (more deterministic, peaked on argmax); beam size ↑ better quality but more compute; beam=1 = greedy; top-k restricts to k most probable

### HuggingFace pipeline literals (Code 2)

The exam tests the canonical 5–15 line skeleton from memory. These tokens MUST be reproducible:

- `from transformers import AutoTokenizer, AutoModelForQuestionAnswering` (and analogously `AutoModelForSequenceClassification`)
- `import torch`
- `tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")`
- `model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")`
- `inputs = tokenizer(question, context, return_tensors="pt")` (also `padding=True, truncation=True` for batched/text-classification)
- `with torch.no_grad():` — disables gradient tracking for inference
- `outputs = model(**inputs)`
- `start_idx = torch.argmax(outputs.start_logits)`
- `end_idx = torch.argmax(outputs.end_logits) + 1`
- `tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx])`
- For text classification: `num_labels` parameter = number of output classes
- `outputs.start_logits` / `outputs.end_logits` are the per-token scores for span start/end
- `attention_mask`: 1 for real tokens, 0 for padding

### Regex — beyond the symbol table on the sheet

The symbol table covers basics. These advanced constructs are tested in MCQs and NOT on the sheet:

- **Lookbehind:** `(?<=X)` positive (preceded by X), `(?<!X)` negative (not preceded by X) — zero-width, doesn't consume characters
- **Lookahead:** `(?=X)` positive, `(?!X)` negative
- **Lazy quantifiers:** `*?`, `+?` match as little as possible (regex is otherwise greedy). On `"abcabc"` the pattern `.*?` matches **the empty string** as its first match
- **`\b` word boundary** (used in `\b\w+ing\b`)

### BoW with `Counter` skeleton (Code 1)

Reproducible idiom for predict-output:

```python
from collections import Counter
tokens = [["nlp","is","fun"], ["nlp","is","powerful"], ["learning","nlp","is","fun"]]
vocab = sorted({t for sent in tokens for t in sent})
tf_matrix = []
for sent in tokens:
    counts = Counter(sent)
    tf_matrix.append([counts[w] for w in vocab])
print(vocab)         # ['fun','is','learning','nlp','powerful']
print(tf_matrix)     # [[1,1,0,1,0],[0,1,0,1,1],[1,1,1,1,0]]
```

The exam-ready insight: `Counter[w]` returns 0 for missing keys (no `KeyError`); set comprehension `{t for sent in ... for t in sent}` flattens; `sorted()` on a set returns a list in alphabetical order.

### Sentiment / discourse

- Sentiment treated as **a dynamical signal evolving across observations** (time-series interpretation in Quiz III), not a static lexical property
- Sentiment-over-time used to study discourse dynamics / narrative shifts / market signals

### Loss & training (beyond what's on the sheet)

- Logistic regression training **minimizes cross-entropy** (= negative log-likelihood) — not 0–1 error, not margin
- For binary classification with uniform p(y=1) = p(y=0) = 0.5, entropy = `log 2` (max for binary)
- Probabilities near 0.5 ⇒ high entropy / max uncertainty
- A near-zero coefficient ⇒ the word has little influence; large positive ⇒ strong positive evidence; +1.5 coefficient ⇒ odds × `e^1.5`

---

## Study priorities (derived from this blueprint)

1. **Memorize the HF QA + classification skeletons cold** — Code 2 is 10 points, formula sheet gives zero help.
2. **Drill the three exercise patterns by hand**: 2×2 attention, 3-doc TF-IDF, 2-state Viterbi. Each is 10 points and the formula sheet *does* provide the equations — you just need fluency.
3. **Internalize Word2Vec + LSA + LDA conceptually** — the formula sheet gives the math but not what the math means. MCQs target the meaning.
4. **Lookbehind / lookahead / lazy quantifiers** — the regex symbol table on the sheet stops at the basics; these get MCQ traps.
5. **Multi-head attention, positional encoding, causal masking, cross-attention** — concept-level only, no formula support.
6. **Library purposes**: spaCy = industrial pipeline, NLTK = educational, gensim = embeddings/topics, HF = pretrained models. Cheap MCQ points.
