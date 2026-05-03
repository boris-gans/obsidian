---
type: study-guide-cluster
course: NLP
cluster: "09-modern-applications"
theme: "Sentiment, classification, generation, QA, RAG"
prerequisites: [04-classical-classifiers, 07-neural-sequence-models, 08-attention-and-transformers]
covers-concepts:
  - text-classification
  - sentiment-analysis
  - huggingface-text-classification
  - text-generation-sampling
  - huggingface-summarization
  - extractive-question-answering
  - retrieval-augmented-generation
covers-lectures:
  - lecture-11-sentiment-analysis
  - lecture-24-challenges-and-trends
exam-weight: high
---

# Cluster 9: Modern applications and open problems

> **The story of this cluster in one sentence.** Cluster 8 built the transformer; this cluster is the **catalogue of what people fine-tune it to do** — classify, generate, summarize, answer — wrapped in HuggingFace pipelines, with the Code 2 exam question (HF QA fill-in-blanks) hiding inside.

## Why this cluster exists

Almost every modern NLP product is a transformer plus a task-specific head. This cluster names the **canonical task interfaces** so you can recognize each pipeline and reproduce its 5–15 line skeleton from memory. It also names what's *still hard* — vocabulary mismatch (which never fully went away), hallucination, long context, factual grounding — and the modern attempt to patch them with retrieval-augmented generation.

The blueprint flags this as **very high weight** because **Code 2 — fill-in-blanks on the HuggingFace QA pipeline — is 10 points** and lives entirely in this cluster. The pipeline tokens (`AutoTokenizer.from_pretrained`, `return_tensors="pt"`, `with torch.no_grad()`, `outputs.start_logits`, `outputs.end_logits`, the argmax decode) must be reproducible cold. MCQs cover sentiment as a discourse signal (Quiz III Q2, Q16.B), HF specifics like `num_labels` (mock Q18), padding/truncation (mock Q20), generation strategies temperature/beam/top-k (Quiz IV Q15–Q17 + B), extractive QA span semantics (mock Q19), retrieval-augmented systems (mock Q22).

**Prerequisites you should feel solid on:**

- [[transformer]] — every pipeline in this cluster is a fine-tuned transformer
- [[softmax]] — the output activation for classification heads
- [[logistic-regression]] / [[evaluation-metrics]] — sentiment is a binary classification task; metrics from Cluster 4 still apply
- [[attention]] — extractive QA's start/end logits are an attention-style soft pointer over input positions

## The arc

### 1. [[text-classification]] — the umbrella task

Map a piece of text to one of `K` classes. Classical recipe: BoW/TF-IDF vector → LR or NB → softmax. Modern recipe: transformer → `[CLS]` token's last-layer hidden vector → linear head → softmax over `K` classes. Same task, different representation. The modern version dramatically improves accuracy on hard examples (sarcasm, negation, domain shift) because the transformer produces *contextual* embeddings; the classical version is faster, more interpretable, and competitive on simple tasks. Both still use the **evaluation metrics from Cluster 4**: precision, recall, F1, accuracy with the imbalanced-data caveat.

### 2. [[sentiment-analysis]] — the canonical text classification task

Sentiment analysis assigns text a polarity (positive / negative, sometimes neutral or fine-grained 1–5 stars). It's the canonical text-classification benchmark — IMDB reviews, Amazon reviews, Twitter polarity. The blueprint's interesting twist: Quiz III treats sentiment as a **time-series signal**, where you watch sentiment evolve across documents (e.g. across news articles over a quarter, or across reviews ordered by time) to study **discourse dynamics or market signals**. The takeaway is that the *output of a sentiment classifier is itself data* that can be analyzed downstream — sentiment isn't a static lexical property of words but a **dynamical signal** (Quiz III Q2). For the exam: know both the per-doc classification framing and the discourse / time-series framing.

### 3. [[huggingface-text-classification]] — the production pipeline

The HF text-classification pipeline:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=-1)
```

Memorize the kwargs: `num_labels=K` (= number of output classes — mock Q18), `padding=True` and `truncation=True` (make variable-length inputs into a fixed-shape tensor — mock Q20), `return_tensors="pt"` (return PyTorch tensors instead of lists), `with torch.no_grad()` (disable gradient tracking for inference). These are the *exam tokens* — Code 2 might be QA, but classification uses the same skeleton with three swapped names.

### 4. [[text-generation-sampling]] — temperature / beam / top-k

Once you have a transformer language model that outputs a probability distribution over the next token, you need a **sampling strategy** to actually pick one. Three to know:

- **Temperature** — divide logits by `T` before softmax. `T → 0` = deterministic argmax (always pick the most probable token). `T → ∞` = uniform random. `T = 1` = sample from the model's natural distribution. Higher = more creative / diverse, lower = more focused / repetitive.
- **Beam search** — keep the top-`k` partial sequences at each step rather than committing to one. `beam = 1` = greedy. Higher beam = better likelihood but more compute and *less* diverse output (a known beam-search pathology).
- **Top-k sampling** — restrict each step's sampling to the `k` most probable tokens. Avoids the fat-tail problem of pure sampling (where rare nonsense tokens occasionally get chosen).

Quiz IV Q15–Q17 + B drill all three. Memorize the *direction of effect* of each knob.

### 5. [[huggingface-summarization]] — the encoder-decoder application

Summarization is a sequence-to-sequence task: long text → short summary. It's the **canonical encoder-decoder transformer** application — encoder reads the source, decoder generates the summary token-by-token using cross-attention to the encoder. Modern models (BART, T5, Pegasus) are pre-trained on a mix of denoising and reconstruction objectives, then fine-tuned on summary corpora (CNN/DailyMail, XSum). The HF pipeline is `from transformers import pipeline; summarizer = pipeline("summarization")`. Two failure modes worth knowing: **extractive bias** (the model copies source spans rather than abstracting) and **hallucination** (the model generates fluent text not entailed by the source). These connect to Cluster 1's ELIZA-effect lesson: fluency ≠ understanding.

### 6. [[extractive-question-answering]] — predict a span by its endpoints

Extractive QA: given a (question, context) pair, return the **substring of the context** that answers the question. The transformer takes input `[CLS] question [SEP] context [SEP]`, processes it, and outputs **two logit vectors over context positions**: `start_logits` and `end_logits`. The predicted answer span is `context[argmax(start_logits) : argmax(end_logits) + 1]`. This is **Code 2 of the mock exam (10 pts)**:

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

inputs = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

start_idx = torch.argmax(outputs.start_logits)
end_idx   = torch.argmax(outputs.end_logits) + 1
answer    = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx])
```

The five blanks the prof targets (per the mock): the two `from_pretrained` lines, `return_tensors="pt"`, `with torch.no_grad():`, and the `start_logits` / `end_logits` access. Memorize this skeleton **cold** — formula sheet provides nothing for it.

### 7. [[retrieval-augmented-generation]] — fixing the parametric-knowledge limit

A pure transformer LM stores all its knowledge **in its parameters** — which means (a) it can't be updated without retraining, and (b) it confidently makes things up when it doesn't know (hallucination). **Retrieval-augmented generation (RAG)** fixes both by letting the model **query an external document store** at inference time: a retriever fetches relevant docs, the transformer generates conditioned on both the query *and* the retrieved passages. This is the modern cure for vocabulary mismatch (Cluster 3's old wound — finally cured at the application level: even if the model never saw your domain in training, it can read your domain at inference). Mock Q22 names this directly. Open problems remain — retrieval quality, citation accuracy, long-context efficiency — and they're the live research frontier the course's "challenges and trends" lecture surveys.

## Connections worth seeing

- **The HF pipelines are all the same shape with three slots swapped.** `tokenizer = AutoTokenizer.from_pretrained(...)`, `model = AutoModelFor<TASK>.from_pretrained(...)`, `outputs = model(**inputs)`, `decode_or_argmax(outputs)`. Once you've memorized one pipeline, you can write the others by changing the model class and the output extraction. This is *the* high-leverage observation for Code 2.
- **Extractive QA's `start_logits` / `end_logits` are a special case of attention.** They're soft pointers over input positions — the same template as attention weights. The model is essentially asking "which input position is the start of the answer?" via softmax over context tokens.
- **Sampling temperature is structurally identical to softmax temperature in classification.** Same `softmax(z / T)` dial; in classification it controls the confidence of the prediction, in generation it controls the diversity of sampling. *Same knob, two roles.*
- **RAG is the application-layer answer to vocabulary mismatch (Cluster 3).** TF-IDF couldn't fix it. Word2Vec partially fixed it by giving similar-meaning words similar coordinates. RAG completes the fix by putting the actual relevant text *in the prompt* — synonyms don't matter if the source doc is in the context window.
- **The accuracy trap (Cluster 4) and the ELIZA effect (Cluster 1) are both alive in this cluster.** A summarizer with high ROUGE can be hallucinating; a QA model with high SQuAD F1 can fail on out-of-distribution questions; a chatbot can pass casual benchmarks while making up facts. *The same evaluation warnings apply, just at a higher level of capability.*
- **Sentiment as a time series** (lecture 11) is the modern reincarnation of the **discourse dynamics** that Cluster 1's symbolic NLP couldn't quite capture. We've come back to discourse, but now with a learned classifier instead of hand-written rules.

## Common confusions

- **Extractive QA vs. generative QA** — extractive returns a span of the input; generative writes a free-form answer. Mock Q19 specifies extractive; the canonical model is BERT for QA, not GPT.
- **`num_labels` is for classification, not for QA** — QA models don't have a classification head; they have *two logit outputs* (start, end) over input positions.
- **Temperature direction** — high T = more diverse, low T = more focused. *Easy to flip under exam pressure.*
- **Beam search direction** — bigger beam = better likelihood, more compute, *less* diverse. (Counterintuitive: greedy is often more diverse than beam.)
- **`with torch.no_grad():`** — disables gradient tracking. Use for inference. Forgetting it doesn't break anything but wastes memory.
- **`return_tensors="pt"`** — returns PyTorch tensors. `"tf"` for TensorFlow. The default returns Python lists — useless for `model(**inputs)`.
- **Padding vs. truncation** — padding lengthens short inputs; truncation shortens long inputs. Both are needed because the model expects fixed-shape tensors.
- **RAG vs. fine-tuning** — RAG looks up new info at inference; fine-tuning bakes new info into parameters. Different update granularity.

## Self-check (synthesis, not recall)

1. **Code 2 cold-write drill** — without looking at any file, write the 5–15 line HuggingFace QA pipeline from memory. Then check it against [[extractive-question-answering]]. Which lines did you forget? Practice those specific tokens until they're automatic.
2. **Beam search produces high-likelihood, often boring text.** Pure sampling produces diverse but sometimes nonsensical text. Top-k sits in between. Pick a real-world generation task (chatbot reply, news headline, code completion) and justify which strategy you'd default to.
3. **Sentiment classification is a Cluster 4 problem with a Cluster 8 backbone.** Trace the data flow: input text → tokenization (Cluster 2) → embedding lookup (Cluster 5) → transformer encoder (Cluster 8) → `[CLS]` representation → linear head → softmax → label. Where does the *training signal* enter, and what loss is used?
4. **Hallucination in summarization** vs. **the ELIZA effect** (Cluster 1) — argue whether they're the same phenomenon or different. What's the diagnostic test that would tell them apart?
5. **RAG vs. simply retraining the model on the new data.** What's the trade-off? Think about update latency, parameter cost, and what happens when the underlying knowledge base changes weekly.
6. **Tying everything together:** if you had to teach the entire course in one sentence, what would it be? Try this answer and refine it: *"NLP is a sequence of attempts to give machines a representation of meaning that overcomes [X], where X starts as ambiguity, becomes high-dim sparsity, then becomes lack-of-context, then becomes the encoder-decoder bottleneck, and currently is parametric-knowledge brittleness."*

## If you have 10 minutes

1. **[[extractive-question-answering]] skeleton** — write the 5–15 lines from memory. This is **10 exam points**. Drill until automatic.
2. [[huggingface-text-classification]] — memorize `num_labels`, `padding`, `truncation`, `return_tensors="pt"`, `with torch.no_grad()`
3. [[text-generation-sampling]] — direction of temperature, beam size, top-k effects
4. [[retrieval-augmented-generation]] — what it adds to a vanilla LM and why (mock Q22)
5. The HF-pipeline-template observation in "Connections worth seeing" — see the shape that's common across all three pipelines

## Next cluster

→ *None — this is the end of the course.* Re-read [[index]]'s "shape of the course" paragraph as a final sanity check that you can name the through-line: ambiguity (1) → counts (2) → patterns + retrieval (3) → classification (4) → meaning-as-geometry (5) → sequence labeling (6) → neural sequence models (7) → attention + transformers (8) → applications (9). Each cluster opens a door the next one walks through.
