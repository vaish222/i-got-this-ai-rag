# I Got This — What's Next?

A privacy-first personal and family knowledge assistant, developed incrementally from a controlled RAG dataset.

## Current status: Phase 0 complete

This repository currently contains only the dataset milestone described in the product requirements:

- 20 synthetic documents across school, children's activities, personal learning, volunteer, household, social, and family domains;
- YAML-style metadata on every source document;
- 15 evaluation questions with expected answers and expected source documents;
- exactly two deliberately unanswerable questions; and
- privacy exclusions for local personal documents and future Qdrant storage.

The corpus uses fictional anonymous identifiers and a fixed reference date of **August 20, 2026**, which makes relative-time evaluation reproducible. See [data/README.md](data/README.md) for dataset conventions and [evaluation/README.md](evaluation/README.md) for the evaluation category distribution.

No RAG pipeline, embeddings, vector database, generation model, LangGraph workflow, or UI has been implemented yet. Those belong to later phases.

## Repository layout

```text
data/
├── README.md
├── private/                 # ignored; local personal documents only
└── sample/                  # 20 synthetic public documents
    ├── activities/          # 3
    ├── family/              # 2
    ├── household/           # 3
    ├── learning/            # 3
    ├── school/              # 4
    ├── social/              # 3
    └── volunteer/           # 2

evaluation/
├── README.md
└── questions.json           # 15 ground-truth evaluation records
```

The full roadmap and requirements are in [product_requirements.md](product_requirements.md).
