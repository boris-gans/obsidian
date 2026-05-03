---
course-code: NLP
display-name: Natural Language Processing
semester: TBD
exam-date: TBD
exam-format: |
  Closed-book final, 1h20, 100 points total.
  - Multiple-choice (50 pts): 25 questions × 2 pts each. 5 backup MCQs at the end of the booklet, used in fixed order if any of Q1–Q25 are invalidated. Multi-select invalidates a question; only grid bubbles count.
  - Exercises (30 pts): 3 questions × 10 pts each. Algorithmic compute on micro inputs (e.g. attention with 2×2 matrices, TF-IDF on a 3-doc corpus, Viterbi on a 2-state HMM).
  - Code (20 pts): 2 questions × 10 pts each. Predict-output on a small Python snippet AND/OR fill-in-blanks on a canonical pipeline pattern from the notebooks (e.g. HuggingFace QA).
  Formula sheet (3 pages) is appended to the exam booklet — students cannot unstaple. See `40-Reviews/NLP/exam-blueprint.md` for what's on it.
diagram-categories:
  preprocessing: "#5b8def"     # tokenization, regex, normalization, padding
  classical: "#52a675"         # BoW, TF-IDF, LSA, LDA, Naïve Bayes
  embeddings: "#a366d9"        # Word2Vec, dense vs sparse, cosine
  sequence: "#e8a04a"          # RNN, HMM, PoS, NER, dependency parsing
  transformer: "#d65a5a"       # attention, multi-head, transformers
  applications: "#3aa6a6"      # classification, generation, summarization, QA
---

# Natural Language Processing

## Course quirks

- **Code questions come from the notebooks** in `30-Sources/NLP/notebooks/`. Treat each notebook as an algorithm reference — capture the canonical pipeline skeleton (5–15 lines: imports + load + tokenize-with-kwargs + run + decode) in the matching concept note. Don't drift into library-API trivia.
- **Generous formula sheet**: vector ops, BoW, TF-IDF, cross-entropy, logistic regression, Naïve Bayes, HMM Viterbi (with δ-table layout), transformer input format, softmax, attention (Q/K/V, scores, scaled, weights, output), RNN update, LSA, LDA, evaluation metrics, full regex symbol table. **Memorization effort goes to what's NOT on the sheet** — see `40-Reviews/NLP/exam-blueprint.md` for both lists.
- **HuggingFace specifics tested in MCQs and code** (`num_labels`, `start_logits`, `end_logits`, `input_ids`, `return_tensors`, `with torch.no_grad()`, `padding`, `truncation`). Notebooks 19, 20, 21, 22 are the sources.
- **Exercises are micro-scale.** 2×2 matrices, 2-state HMMs, 3-document corpora. Run them by hand; the formula sheet is the cheat code.
- **Slide filenames use `Session NN - Topic.pdf`** (not `NN_topic.pdf`). Some have hyphenated suffixes (`-1`) and typos (`Adnvaced`, `Beyonf`). Use Glob to find the exact filename — don't infer.
- **Slide and notebook numbering are NOT aligned**. E.g. `Session 14 - POS tagging.pdf` ↔ `10_Part_of_Speech_tagging.ipynb`. Map by topic, not by number, when linking.

## Sources at hand

- [x] Slides (`pdf/`, `slides-png/`, `images/`) — 19 PDFs (`Session 01` through `Session 24`, with gaps)
- [x] Notebooks (`notebooks/`) — 12 `.ipynb` files: BoW, Naïve Bayes, Sentiment, Word Embeddings, Word2Vec, PoS, NER, Elman RNN, HF Text Classification, Text Generation, Text Summarization, QA
- [x] Mock exam (`mock-exams/`) — `Final Exam - proposal.pdf` + solutions
- [x] Quizzes (`quizzes/`) — 4 sets (I, II, III A/B, IV A/B), all with solutions
