# I Got This — What's Next?

A privacy-first personal and family knowledge assistant, developed incrementally from a controlled RAG dataset.

## Current status: Phase 1 complete

The Phase 0 controlled dataset remains unchanged:

- 20 synthetic documents across school, children's activities, personal learning, volunteer, household, social, and family domains;
- YAML-style metadata on every source document;
- 15 evaluation questions with expected answers and expected source documents;
- exactly two deliberately unanswerable questions; and
- privacy exclusions for local personal documents and Qdrant storage.

The corpus uses fictional anonymous identifiers and a fixed reference date of **August 20, 2026**, which makes relative-time evaluation reproducible. See [data/README.md](data/README.md) for dataset conventions and [evaluation/README.md](evaluation/README.md) for the evaluation category distribution.

Phase 1 is implemented as an executable notebook: [notebooks/phase_1_naive_dense_rag.ipynb](notebooks/phase_1_naive_dense_rag.ipynb). It provides:

- Markdown, TXT, and PDF loading;
- whitespace cleaning and YAML front-matter preservation as metadata;
- token-aware recursive chunking with a 500-token target and 75-token overlap;
- free local `embeddinggemma` embeddings through LangChain and Ollama;
- an on-disk local Qdrant collection using dense vectors only;
- Top-5 similarity retrieval;
- local `gemma3:1b` answer generation with explicit insufficient-information behavior; and
- answer citations linked to retrieved document and chunk metadata.

No Phase 2+ work is included: the evaluation set is not executed, and there is no hybrid search, reranking, metadata filtering, query rewriting, LangGraph, or UI.

## Run the Phase 1 notebook

This project targets Python 3.12 and also supports Python 3.13.

```bash
uv sync
cp .env.example .env
ollama pull embeddinggemma
ollama pull gemma3:1b
```

Make sure Ollama is running, open the notebook, select the repository's `.venv` kernel, and run the cells from top to bottom. On macOS, opening the Ollama app starts its local service; you can also run `ollama serve` in a separate terminal. No API key is required.

The default corpus is the synthetic `data/sample` directory. To use local personal files, change `RAG_DATA_DIR` to `data/private` in `.env`. Document chunks, questions, embeddings, and generated answers stay on the local machine. Qdrant vectors and payloads remain under the ignored local `qdrant_storage/` directory.

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

notebooks/
└── phase_1_naive_dense_rag.ipynb

pyproject.toml               # uv environment and notebook dependencies
```

The full roadmap and requirements are in [product_requirements.md](product_requirements.md).
