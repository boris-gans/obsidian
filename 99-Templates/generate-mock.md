# Generate practice mock exam

Use this prompt to generate a fresh practice exam in this prof's style.

---

Generate a new practice exam matching the format and abstraction level of the original mock — but explicitly varying the open-ended topics to cover the rest of the course.

**Inputs to read first:**

- `40-Reviews/Computer-Vision/exam-style-guide.md` — format and rubric
- `40-Reviews/Computer-Vision/exam-blueprint.md` — original mock's MCQ topic distribution
- `30-Sources/Computer-Vision/mock-exams/mock-exam.pdf` and `mock-exam-graded.pdf` — calibration
- `10-Courses/Computer-Vision/lectures/` and `20-Concepts/` — source material; every question must be answerable from these
- `40-Reviews/Computer-Vision/practice-mocks/` — list existing practice mocks so you don't repeat their topics

---

## MCQ section (25 × 0.3 = 7.5 pts)

Match the original mock's topic distribution from `exam-blueprint.md` — this is the part that should feel familiar topic-wise. Same lectures, same coverage proportions, but new angles on each topic. No reused questions.

Distribute roughly:
- Fourier & sampling: ~5 questions
- Transformers & attention: ~3 questions  
- PyTorch (broadcasting, autograd, tensor ops): ~4 questions
- CNN basics (pooling, receptive field, dimension math): ~3 questions
- Edge detection & gradients: ~3 questions
- Segmentation/detection (IoU, Dice): ~2 questions
- Corner/keypoint detection: ~2 questions
- Hough transform: ~1 question (the big one is in the algorithmic section)
- Other (CLIP, Inception, ICP, shape descriptors): ~2 questions

Keep difficulty distribution similar: most medium, a few easy recall, a few traps.

---

## Open-ended section — VARY THE TOPICS

The open-ended questions reuse the *shape* of the originals but **on different topics from across the course**. The graded mock teaches you the prof's preferred abstraction level, length, and structure — not which topic to ask about. Use any lecture (10–20) as fair game, but pick topics not yet tested in the original mock or previous practice mocks.

### Two conceptual open-ended questions (2 × 0.75 = 1.5 pts)

For each, match the original conceptual questions' shape:
- Same length expectation (per `exam-style-guide.md`)
- Same structure of model answer (definition → mechanism → why it matters → example, or whatever pattern the style guide identifies)
- Same depth — neither too recall nor too research-y
- Same use of equations/diagrams as the originals

But pick topics from lectures that the **original mock under-tested** or didn't cover at all. Candidates (cross-check against blueprint and previous mocks):
- Lecture 13: binary object characterization (connectivity, region labelling, shape descriptors beyond solidity)
- Lecture 15: intro to deep learning (gradient descent variants, regularization, overfitting)
- Lecture 18: segmentation/detection beyond IoU and Dice (e.g. anchor boxes, NMS in detection, mask vs box loss)
- Lecture 19: self-supervised learning beyond CLIP (contrastive vs predictive, SimCLR, masked modelling)
- Lecture 20: autoencoders (vanilla vs variational, denoising, latent space, reconstruction loss)
- Lecture 14: PyTorch concepts beyond simple ops (computational graph, training loop structure, common pitfalls)
- Less-tested angles in covered lectures (e.g. active contours from Lecture 11, ICP edge cases from Lecture 12)

Examples of question shape (don't reuse, but match the abstraction level):
- *"Explain what positional encoding is and describe two schemes for implementing it."* → New: *"Explain what an autoencoder learns to do and describe two architectural variants."*
- *"Describe how to downsample an image by 2 in the Fourier domain. Why is blurring necessary first?"* → New: *"Describe how a denoising autoencoder is trained. Why is the reconstruction objective alone insufficient without the noise injection?"*

### One algorithmic open-ended question (1 × 1.0 = 1.0 pt)

Match the original's shape: requires both a conceptual explanation AND an algorithmic component (pseudocode or step-by-step procedure), with the breakdown of marks roughly mirroring the original.

The original's algorithmic question was Hough transform line detection. Don't reuse Hough. Pick a different algorithm from the course where a similar concept-plus-pseudocode answer makes sense:

- ICP for point cloud alignment
- Canny edge detector full pipeline
- SIFT keypoint detection or matching
- Non-maximum suppression in object detection
- Mean shift or k-means in unsupervised contexts
- Gradient-based edge detection from scratch
- Active contour energy minimization loop
- A specific CNN training step with backprop

Match the original's depth: the prof's graded answer length, the prof's level of pseudocode detail, the prof's mix of motivation + algorithm + complexity/edge-case discussion.

---

## Output two files

Use `{N}` = next integer based on what's in `practice-mocks/`.

1. **`40-Reviews/Computer-Vision/practice-mocks/mock-{N}-questions.md`** — questions only. Frontmatter: `tags: [practice-mock, Computer-Vision]`, `generated: YYYY-MM-DD`. Include a "my answer" blank under each question. Match the original's question numbering and marks notation.

2. **`40-Reviews/Computer-Vision/practice-mocks/mock-{N}-answer-key.md`** — full worked answers in the prof's style:
   - Each MCQ: correct answer + one-sentence justification + brief note on why each distractor is wrong
   - Each conceptual: a model answer at the length and structure specified by `exam-style-guide.md`
   - The algorithmic: full worked solution at the depth of the prof's original
   - Each answer cites lecture(s) and concept note(s) it draws from
   - Frontmatter: `tags: [practice-mock-key, Computer-Vision]`

---

## Before writing

Show me a topic plan as a table:

| Q# | Marks | Topic | Lecture | Question shape |

Then list which topics from the original mock you're consciously *not* repeating, and which under-tested topics you're surfacing. Wait for "go".

---

## Variation across mocks

If `practice-mocks/` already contains generated mocks:
- Read them and avoid topic overlap on the open-ended questions especially
- Rotate algorithmic questions across mocks (mock-1: Canny; mock-2: ICP; mock-3: SIFT) so I drill different procedures
- Avoid repeating MCQ angles even if the topic recurs (each Fourier MCQ tests a different facet)

After 2-3 mocks, you'll have hit most of the course; further mocks should remix more aggressively.
