# Evaluation dataset

`questions.json` is the Phase 0 ground-truth set. It contains 15 questions in the category distribution required by the PRD:

| Category | Count |
|---|---:|
| Exact lookup | 3 |
| Semantic | 3 |
| Date/deadline | 3 |
| Cross-document | 2 |
| Cross-domain | 2 |
| Unanswerable | 2 |

Expected source titles are human-readable, while expected source IDs provide stable keys for later automated evaluation. Expected answers describe the facts a grounded answer should contain; they are not prompts or generated results.

Relative dates are evaluated as of **2026-08-20** in `America/Los_Angeles`.
