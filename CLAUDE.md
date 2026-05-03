# Study Vault Instructions

This vault holds study materials for multiple university courses. The structure and workflow are course-agnostic; per-course details live in each course's manifest at `30-Sources/{course}/_course.md`.

## Folders

- `00-Inbox/` — raw captures, untriaged
- `10-Courses/{course}/` — your work for the course (lecture notes, problem sets, syllabus)
- `20-Concepts/` — atomic concept notes, **shared across all courses**, one idea per file, kebab-case filenames
- `30-Sources/{course}/` — slides, notebooks, mock exams, quizzes, and other raw materials for that course (see [Source layout](#source-layout-per-course))
- `40-Reviews/{course}/` — flashcards, study guides, clusters, practice mocks for that course
- `99-Templates/` — Templater templates, prompt templates

`{course}` is the course code, e.g. `Computer-Vision`, `History-101`. Use kebab-case, no spaces.

## Per-course manifest

Each course has a manifest at `30-Sources/{course}/_course.md` that captures course-specific details:

```yaml
---
course-code: Computer-Vision      # exact folder name used in paths
display-name: Computer Vision     # human-readable
semester: Spring 2026
exam-date: 2026-04-29
exam-format: |
  Free-form description of the exam structure, marks, duration
diagram-categories:               # optional, for Mermaid color coding
  category-name: "#hex-color"
---

# {Display Name}

Notes about this course's quirks: prof's grading style, common pitfalls, abbreviations used in slides, etc.
```

**Always read `30-Sources/{course}/_course.md` before working on any artifact for that course.** It tells you the exam format, the prof's idiosyncrasies, and any course-specific conventions.

If a course manifest doesn't exist when the user references a new course, copy `99-Templates/_course.md` to `30-Sources/{course}/_course.md`, fill in everything you can infer from the user's request and the source materials at hand, then ask the user only for fields you genuinely can't infer (typically `semester` and `exam-date`). Don't blockingly interview them on every field.

## Concept notes are shared across courses

`20-Concepts/` is a single shared library across all courses — concepts genuinely overlap (Bayes' theorem appears in stats, ML, and epistemology; gradient descent in optimization and deep learning). Frontmatter on every concept note must include a `courses:` array listing every course that touches the concept:

```yaml
---
tags: [concept]
courses: [Computer-Vision, Deep-Learning]
sources:
  - course: Computer-Vision
    file: 16_CNNs.pdf
  - course: Deep-Learning
    file: 03_optimization.pdf
created: YYYY-MM-DD
---
```

When ingesting from a new course's slides, **search 20-Concepts/ first**. If a concept exists, EXTEND it with this course's framing, add the course to the `courses:` array, and add the new source. Don't duplicate.

## Rules for note creation

- **Atomic** — one concept per file in `20-Concepts/`
- **Search before creating** — extend or link, don't duplicate
- **Link aggressively** with `[[wikilinks]]`; orphans are failures
- **YAML frontmatter required** on every note
- **Never invent facts.** If a source doesn't say it, don't write it.
- When extracting from `30-Sources/`, cite the exact source filename in the note

## When generating flashcards

Format: **multi-line Spaced Repetition** — question on its own line, then `?` alone on the next line, then the answer. Cards separated by a blank line:

```
question text
?
answer text
```

Rules:

- Do NOT use the older `Q:: / A::` inline format
- Atomic recall, not paragraph dumps
- Equations stay as LaTeX (`$...$` inline, `$$...$$` display); preserve verbatim
- If a card has an existing `<!--SR:!date,interval,ease-->` review-history comment, place it on the line directly after the answer
- Save to `40-Reviews/{course}/flashcards/lecture-{NN}.md`
- Frontmatter: `tags: [flashcards, {course}]`

## Style

- Plain language. Define jargon on first use.
- Equations as LaTeX (`$...$` inline, `$$...$$` display).

---

## Source layout per course

`30-Sources/{course}/` holds all raw materials for one course. Standard subfolders:

- `_course.md` — the per-course manifest (see above)
- `pdf/` — slide decks converted to PDF (auto-filled by `bin/ingest-slides.sh`)
- `slides-png/{lecture-name}/slide-NN.png` — full-slide screenshots (auto-filled)
- `images/{lecture-name}/img-NNN.png` — extracted slide bitmaps (auto-filled)
- `notebooks/` — Jupyter / Colab notebooks (`.ipynb`)
- `mock-exams/` — past or sample exams provided by the prof
- `quizzes/` — short quizzes (graded or practice)

Drop `.pptx` decks at the top level of `30-Sources/{course}/`; `bin/ingest-slides.sh {course}` converts and routes them into `pdf/`, `slides-png/`, `images/`. Drop notebooks, mock exams, and quizzes directly into their own subfolders — no script processing required, Claude reads them in place.

Not every course will have every subfolder. Create only what the course actually has.

---

## Slide ingest workflow

When the user asks to ingest a lecture slide deck:

**Pre-flight (every time):**

1. Identify the course from the user's request or the slide path. Read `30-Sources/{course}/_course.md`. If it doesn't exist, ask for course details and create it.
2. Confirm the slides are PDF + slides-png pre-rendered. If not, run `bin/ingest-slides.sh {course}` first.

**Per-slide-deck:**

1. Read the PDF in `30-Sources/{course}/pdf/{NN}_{topic}.pdf`
2. Create ONE lecture note in `10-Courses/{course}/lectures/lecture-{NN}-{topic}.md`
   - Frontmatter:
     ```yaml
     tags: [lecture]
     course: {course}
     source: {pdf-filename}
     lecture: NN
     slides: "[[30-Sources/{course}/pdf/{pdf-filename}|Open slides ↗]]"
     created: YYYY-MM-DD
     ```
   - The `slides` value is a vault-root wikilink so Obsidian renders it as a clickable "Open slides ↗" link. Use the exact source filename including any `(1)` suffix and the `.pdf` extension.
   - Sections: Overview, Key concepts (linked), Equations, Diagrams, Open questions
3. Extract atomic concepts to `20-Concepts/{kebab-case-name}.md`
   - One concept per file
   - Frontmatter must include: `tags: [concept]`, `courses: [{course}]`, a `sources:` entry referencing `{pdf-filename}`, `created: YYYY-MM-DD`
   - **Search before create.** Before writing any new concept note, run `Glob 20-Concepts/*.md` and check for the slug *and* obvious synonyms (e.g. for "self-attention", also check `attention.md`, `multi-head-attention.md`, `scaled-dot-product-attention.md`). If a match exists, EXTEND it — don't duplicate. Add this course to its `courses:` array if missing, and append a new entry to `sources:`.
4. Create flashcards in `40-Reviews/{course}/flashcards/lecture-{NN}.md` (multi-line `?` format)
   - Atomic recall pairs, exam-oriented
   - Cover definitions, when-to-use, common pitfalls
   - 15–25 cards per lecture

Show me a summary of what you'll create BEFORE writing any files. Wait for "go".

---

## Notebook ingest workflow

When the user asks to ingest a notebook from `30-Sources/{course}/notebooks/`:

**Pre-flight:** Read `30-Sources/{course}/_course.md`. If it doesn't exist, create it from `99-Templates/_course.md`.

### Mental model: notebooks are algorithm references, not API documentation

In code-heavy courses, exams test whether the student can **reproduce the canonical pipeline shape** on a closed-book exam — `from X import Y` → load → tokenize/transform with the right kwargs → run → decode by argmax — not trivia about specific library functions.

Concept notes derived from notebooks should capture the **reproducible 5–15 line skeleton** a student would write under exam pressure. Library-specific identifiers (`num_labels`, `start_logits`, `return_tensors="pt"`) are part of the skeleton; vendored numpy/pandas helper calls usually aren't.

**Litmus test:** if the exam asks "fill in the blanks of a HuggingFace QA pipeline", which lines must the student type from memory? Those lines belong in the concept note. Everything else is context.

### Per-notebook

1. Read the `.ipynb` directly with the Read tool — cells, outputs, and plots come back together.
2. Identify the **canonical pipeline pattern** the notebook teaches (e.g. "BoW with `collections.Counter`", "HuggingFace QA: tokenizer + `AutoModelForQuestionAnswering` + argmax over start/end logits"). Name it.
3. Extract atomic concepts to `20-Concepts/`. **Search before create**: `Glob 20-Concepts/*.md` for the slug and obvious synonyms. If a match exists, EXTEND it — append the new source and add this course to `courses:` if missing.
   - Source entry shape:
     ```yaml
     - course: {course}
       file: {notebook-name}.ipynb
     ```
   - Each concept note must include the canonical code skeleton in a fenced ```` ```python ```` block — imports + key calls + decode pattern. Not a function dictionary.
4. Promote a code idiom to its own concept note only when it's broadly reusable across multiple notebooks/lectures.
5. If the notebook accompanies a specific lecture, link it from the lecture note under a `## Notebooks` section using the citation format below.

### Notebook citation format

Use a **vault-root-absolute markdown link** with the cell number in the link text. Obsidian opens the notebook in its native viewer; the cell number guides the reader to the right spot.

`[Description (cell N)](30-Sources/{course}/notebooks/{file}.ipynb)`

Examples:

- `[QA pipeline skeleton (cell 4)](30-Sources/NLP/notebooks/22_Question_Answering.ipynb)`
- `[BoW with Counter (cells 2–5)](30-Sources/NLP/notebooks/03_BoW_modelling.ipynb)`

Use the same format from chat output and from inside lecture/concept notes — vault-root paths resolve from anywhere in Obsidian. Do NOT use `[[wikilinks]]` for notebooks; markdown links open in the embedded notebook viewer, wikilinks don't.

### No flashcards by default

Code recall doesn't fit spaced repetition well — the canonical skeleton lives in the concept note. Add cards only on request.

Show me a summary of what you'll create BEFORE writing any files. Wait for "go".

---

## Mock-exam and quiz ingest workflow

Mock exams (`30-Sources/{course}/mock-exams/`) and quizzes (`30-Sources/{course}/quizzes/`) are the prof's signal about what they test. They are inputs to review artifacts, not standalone notes.

When the user asks to ingest a mock or quiz:

1. **Read** the file (PDF, ipynb, or markdown — use Read directly).
2. **Update the exam blueprint.** Append findings to `40-Reviews/{course}/exam-blueprint.md` per the schema below; create the blueprint if missing. Cite source filenames in the `sources:` frontmatter.
3. **Surface gaps** — for any topic tested but not already a concept note, create or extend the `20-Concepts/` entry (apply the search-before-create rule), citing the mock/quiz as the source.
4. **Generate practice mocks on request** — write derived mocks to `40-Reviews/{course}/practice-mocks/mock-N-questions.md` and `mock-N-answer-key.md`. Glob existing files first to pick the next free `N`. Paraphrase in the prof's style — never copy past questions verbatim.

Show me a summary of what you'll create BEFORE writing any files. Wait for "go".

### Exam blueprint schema

`40-Reviews/{course}/exam-blueprint.md` is the structured record of what the exam tests and how. One per course. Update incrementally as new mocks/quizzes arrive — append to relevant sections, don't rewrite from scratch.

Frontmatter:

```yaml
---
tags: [exam-blueprint, {course}]
course: {course}
created: YYYY-MM-DD
sources:
  - mock-exams/{filename}.pdf
  - quizzes/{filename}.pdf
---
```

Required sections, in order:

- `## Format` — duration, sections, marks distribution, closed/open book, whether a formula sheet is provided.
- `## Marks distribution` — table mapping section → points → question count → per-question value.
- `## Recurring patterns` — for each section, the recurring question shapes. Examples: MCQ (definition / true-false / formal-description / contrast / applied-syntax); exercises (algorithmic compute / proof); code (predict-output / fill-blanks / write-from-scratch).
- `## Topic coverage map` — table mapping each topic to where it appears (which MCQ #s, which exercises, which code questions) and which notebook/lecture it traces back to.
- `## Prof tells` — quirks: backup-question rules, electronic-device policy, partial-credit conventions, anything in the exam preamble.
- `## What the formula sheet provides` — list every formula given. Notes don't need to drill these.
- `## What the formula sheet does NOT provide` — formulas, idioms, and one-liners the student must memorize. The high-leverage section.

---

## Partial ingest

When the user asks to ingest only part of a source — "just the section on Word2Vec", "ingest cells 3–7 of the QA notebook", "the Hough transform slides only" — do NOT regenerate full lecture artifacts.

**Slides:** Read only the named slide range. Extract concepts. If the lecture note already exists, append to its relevant section; create the lecture note skeleton (Overview + just the named section) only if missing.

**Notebooks:** Read only the named cells. Extract / extend concepts. Don't recreate the lecture's `## Notebooks` link if it's already there.

**Quizzes / mocks:** Update only the relevant rows in the blueprint's `## Topic coverage map` and any new patterns in `## Recurring patterns`. Don't rewrite the blueprint.

In all cases: still apply search-before-create, still show a summary before writing, still wait for "go".

---

## Diagrams: Mermaid first, always

**Default to Mermaid.** Every diagram in a slide must first be attempted as Mermaid. Image embeds are a fallback, not a convenience.

### Process for every diagram

1. Identify what the diagram represents: architecture, flow, hierarchy, state, sequence, comparison, or visualization?
2. Pick the Mermaid type:
   - Sequential pipelines, network architectures, layer stacks → `flowchart LR` or `flowchart TD`
   - Algorithms, training loops, data pipelines → `flowchart TD`
   - Class hierarchies, model genealogies → `classDiagram` or `flowchart`
   - State machines, training phases → `stateDiagram-v2`
   - Forward/backward sequences → `sequenceDiagram` or `flowchart`
   - Comparisons → markdown table, never an image
3. Write the Mermaid using actual labels and dimensions from the slide — not generic "Layer 1." If the slide says "Conv 3×3, 64 filters → ReLU → MaxPool 2×2", the Mermaid nodes say exactly that.
4. Embed an image only if the content matches the allowlist below.

### Image embed allowlist (narrow)

Only permitted for:

- Pixel-level visualizations (filter kernels, learned weights, feature map activations)
- Photographic examples (sample images, real photos demonstrating an effect)
- Plotted graphs with specific data (loss curves, accuracy plots — where shape/values matter)
- Spatial geometry that loses meaning when abstracted (receptive fields on real images, bounding box examples)
- Mathematical visualizations on real data (Fourier spectrum images, frequency domain plots)

If it's not on this list, it must be Mermaid. "It would be complex" is not a reason. ">10 nodes" is not a reason — Mermaid handles 40+ nodes fine.

### Forbidden

- NEVER describe a diagram in prose ("the figure shows three boxes connected by arrows…")
- NEVER embed a slide image as a substitute for a diagram you could Mermaid
- NEVER skip a diagram silently — if you can't Mermaid it and it's not on the allowlist, ask

### Mermaid quality bar

- Use real labels from the slide, including dimensions (e.g. `224×224×3`)
- Group related nodes with `subgraph` when the slide groups them
- Add edge labels for non-obvious transitions: `-->|"stride 2"|`
- For repeated blocks, use one subgraph + a note about repetition rather than copy-pasting
- Split diagrams exceeding ~40 nodes into two views
- **Direction:** prefer `flowchart LR` for sequential pipelines (data → model → output), `flowchart TD` for hierarchies and decision flows. Match the slide's natural reading direction.
- **Use subgraphs to chunk visual complexity.** Group related layers (e.g. "Encoder", "Decoder", "Classifier head") into named subgraphs with `subgraph Name [Display Label]`.
- **Edge labels only when meaningful.** `-->|"stride 2"|` is useful; `-->|"then"|` is noise.
- **Node shape signals type.** Use:
  - `[Rectangle]` for layers / operations
  - `([Stadium])` for inputs/outputs
  - `{Diamond}` for decisions or branching
  - `[(Cylinder)]` for data stores / datasets
  - `((Circle))` for sums / merge points (e.g. ResNet skip-add)
- **One concept per diagram.** If a slide tries to show training loop + architecture + loss in one figure, split into 2-3 Mermaid blocks each focused on one aspect.
- **Annotate dimensions on the edge, not the node**, when feasible: `Conv1 -->|"112×112×64"| Pool1`. Keeps node labels short.
- **Skip connections with explicit merge nodes:** for ResNet-style skips, draw `Input --> Conv1 --> Conv2 --> Add((+)); Input --> Add; Add --> Output`. Don't fake skips with curved arrows — they don't render predictably.

### Course-specific diagram styling

If the course manifest defines `diagram-categories`, use those colors via `classDef`. Otherwise use a calm default palette (avoid neon).

### Image sources

Two pre-extracted sources per lecture:

- `30-Sources/{course}/slides-png/{lecture-name}/slide-NN.png` — full slide screenshots. Default when an image is needed.
- `30-Sources/{course}/images/{lecture-name}/img-NNN.png` — extracted bitmaps. Use only when an isolated clean figure is materially better than the full slide. Indices do not correspond to slide order — identifying the right one requires opening the PDF.

### Image embed format

```markdown
![[30-Sources/{course}/slides-png/{lecture-name}/slide-NN.png]]
*Figure: {what it shows} (slide N).*
```

Use vault-root-absolute paths (no `../../` walks). Always caption with what it shows + slide number.

### Where diagrams go

- Lecture notes: Mermaid liberally throughout. Equations as LaTeX. Images sparingly, only per the allowlist.
- Concept notes: at most one diagram per note. Mermaid strongly preferred. Image only when the concept is fundamentally visual (e.g. "feature map" warrants an image; "convolution operation" warrants Mermaid).

---

## Mehrmaid for markdown-rich diagrams

For diagrams where node labels need full markdown (LaTeX, tables, internal links, multi-line lists), use ```` ```mehrmaid ```` instead of ```` ```mermaid ````. Same syntax; node content wrapped in `("...")` is rendered as Obsidian markdown.

Default to plain `mermaid`. Reach for `mehrmaid` only when the content genuinely needs markdown features. Don't use it to paper over bad diagram structure — restructure first.

### Mermaid syntax constraints (forbidden inside `mermaid` blocks)

These break rendering:

- `>` blockquote lines
- `**bold**`, `_italic_`, `==highlight==`
- `- bullet` or `* bullet` lines
- `# header` lines
- `<!-- HTML comments -->` (use `%% Mermaid comments` instead)
- Raw `$LaTeX$` inside unquoted node labels

For emphasis, use HTML inside quoted labels: `A["<b>bold</b>"]`. For line breaks: `A["line one<br>line two"]`.

---

## Templates vs. ingest output

`99-Templates/lecture.md` and `99-Templates/concept.md` are live-capture stubs for in-class use — intentionally lightweight. They are NOT the structure for slide-ingest output. The slide-ingest workflow above produces a richer post-hoc reference artifact (Overview / Key concepts / Equations / Diagrams / Open questions). Don't reshape ingest output to match the live-capture templates; they serve different audiences (future-you reviewing vs. in-class-you taking notes).

---

## When processing 00-Inbox/

- Split into atomic concept notes in `20-Concepts/`
- Move source material (PDFs, slides, transcripts) to `30-Sources/{course}/`
- Link new notes to existing related concepts
- Show me a diff or summary before writing

---

## Answering study questions

When the user asks a conceptual or content question, **explain it clearly first** — read the relevant lecture and concept notes (and the underlying PDF if needed), then teach the concept in flowing prose.

**Format:**

- Write naturally, like teaching a peer studying for an exam. Full sentences, normal paragraph flow.
- Use markdown freely where it helps: **bold** for key terms, *italics* for emphasis or asides, equations in `$...$` and `$$...$$`, numbered/bulleted lists for steps, Mermaid diagrams for flows.
- **No `> blockquote` formatting in the body** — it fragments the flow.

**Inline source citations — must be Obsidian wikilinks with `#page=N`, not plain text or Markdown links:**

Markdown links to PDFs (`[text](path#page=N)`) break in Obsidian — clicking them produces a "folder already exists" error because Obsidian treats the unresolved path as a request to create a new note. **Use wikilink syntax instead** — Obsidian handles PDFs natively, including spaces in filenames and the `#page=N` fragment.

**Wrong:** `[slide 7 of 10_fourier_transform2.pdf]`
**Wrong:** `[slide 7](30-Sources/Computer-Vision/pdf/10_fourier_transform2.pdf#page=7)` — Markdown link breaks in Obsidian.
**Right:** `[[30-Sources/Computer-Vision/pdf/10_fourier_transform2.pdf#page=7|slide 7]]`

**Path:** always vault-root-relative inside the wikilink (no `../`), regardless of where the citing note lives. Obsidian resolves wikilink paths from the vault root.

**Citation patterns:**

- Single slide: `[[30-Sources/{course}/pdf/{file}.pdf#page=14|slide 14]]`
- Slide range: `[[30-Sources/{course}/pdf/{file}.pdf#page=14|slides 14–17]]` (anchor to first page)
- Multiple slides: stack — `[[…#page=14|slide 14]] [[…#page=22|slide 22]]`
- Synthesis across many slides: `[[…#page=14|synthesis of slides 14–22]]`
- Lecture note section: `[[lecture-12-feature-detectors-2#Hough Transform]]` (no `#page=`)
- Concept note: `[[hough-transform]]`

**Compact text preferred.** The display alias after `|` should be `slide N` when the lecture is clear from context. Use `slide N of 10_fourier_transform2.pdf` only when citing across multiple lectures and the filename is needed for disambiguation.

**Filename rules:**

- Use the EXACT PDF filename, including any `(1)` suffix and any typos in the source filename (e.g. `Beyonf` not `Beyond`)
- Don't strip `.pdf`, don't URL-encode anything (wikilinks use literal characters)
- Don't invent slide numbers — verify by checking the matching slide PNG if uncertain

**Slide-number scheme:** the cited number is the **PDF page index** (1-N within the deck), not any cumulative footer the prof prints. If a deck shows footer "327" on its 20th page, cite it as `[[…#page=20|slide 20]]`, not slide 327. Visible alias must equal the `#page=N` fragment.

**Always prefer slide references** over concept-note references when possible — slides are the original source. Concept notes were derived from slides.

If a claim has no source support, mark it inline as `[not in source]`. Never invent specifics.

**End every grounded answer with three short sections:**

1. **The source's exact framing** — short paragraph in *italics* (no blockquote), capturing how the lecture phrases the concept. End with a clickable slide link.
2. **Remember this** — one-line atomic fact for flashcards.
3. **Beyond the lecture** — what you added that wasn't in the source, plain prose. Skip if nothing.

**How to find slide numbers:**

The slides are pre-rendered as PNGs at `30-Sources/{course}/slides-png/{lecture-name}/slide-NN.png`. To verify or locate a claim, view the relevant slide PNG. The PDF in `30-Sources/{course}/pdf/{lecture-name}.pdf` is also readable directly. When unsure which slide a claim came from, scan the slide PNGs for that lecture rather than guessing.

**Hard rules:**

- Stay grounded — vague hand-waves are fine, fabricated specifics aren't.
- For exam prep, lead with the lecture's framing. Flag textbook discrepancies.
- Diagrams welcome — use Mermaid per the rules above.
