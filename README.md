# I Got This — What's Next?

A privacy-first personal and family knowledge assistant, developed incrementally from a controlled RAG dataset.

## Current status: Phase 2 complete

The Phase 0 controlled dataset remains unchanged:

- 20 synthetic documents across school, children's activities, personal learning, volunteer, household, social, and family domains;
- YAML-style metadata on every source document;
- 15 evaluation questions with expected answers and expected source documents;
- exactly two deliberately unanswerable questions; and
- privacy exclusions for local personal documents and secrets.

The corpus uses fictional anonymous identifiers and a fixed reference date of **August 20, 2026**, which makes relative-time evaluation reproducible. See [data/README.md](data/README.md) for dataset conventions and [evaluation/README.md](evaluation/README.md) for the evaluation category distribution.

Phase 1 is implemented as an executable notebook: [notebooks/phase_1_naive_dense_rag.ipynb](notebooks/phase_1_naive_dense_rag.ipynb). It provides:

- Markdown, TXT, and PDF loading;
- whitespace cleaning and YAML front-matter preservation as metadata;
- token-aware recursive chunking with a 500-token target and 75-token overlap;
- free local `embeddinggemma` embeddings through LangChain and Ollama;
- a hosted Pinecone serverless index using dense vectors only;
- Top-5 similarity retrieval;
- local `gemma3:1b` answer generation with explicit insufficient-information behavior; and
- answer citations linked to retrieved document and chunk metadata.

Phase 2 adds a repeatable baseline evaluation runner that:

- executes all 15 ground-truth questions against the existing Phase 1 Pinecone namespace;
- records retrieved document and chunk IDs, similarity scores, expected-source ranks, generated answers, and resolved citations;
- records retrieval, generation, and total latency for every question;
- calculates macro Recall@5 over the 13 answerable questions; and
- saves `config.json` and `results.json` with SHA-256 links to the exact dataset and configuration used.

No Phase 3+ work is included: there are no chunking or embedding experiments, hybrid retrieval, reranking, metadata filtering, query rewriting, LangGraph, or UI.

## Run the Phase 1 notebook

This project targets Python 3.12 and also supports Python 3.13.

```bash
uv sync
cp .env.example .env
ollama pull embeddinggemma
ollama pull gemma3:1b
```

Create a Pinecone API key, add it to `.env`, and make sure Ollama is running. Then open the notebook, select the repository's `.venv` kernel, and run the cells from top to bottom. On macOS, opening the Ollama app starts its local service; you can also run `ollama serve` in a separate terminal. Ollama requires no API key, but the hosted Pinecone service does.

The default corpus is the synthetic `data/sample` directory. Pinecone receives and stores chunk text, metadata, and locally generated vectors; questions are also sent to Pinecone for retrieval. Answer generation remains local through Ollama. Before changing `RAG_DATA_DIR` to `data/private`, confirm that uploading those personal documents to your Pinecone project matches your privacy requirements.

The notebook creates the configured serverless index if it does not exist and isolates this dataset in `PINECONE_NAMESPACE`. With `REBUILD_NAMESPACE = True`, rerunning the indexing cell replaces only that namespace and does not delete the Pinecone index or other namespaces.

## Run the Phase 2 baseline evaluation

First run the Phase 1 notebook through its indexing cell so the configured Pinecone namespace exists. Keep Ollama running, then execute:

```bash
uv run python evaluation/run_baseline.py
```

The runner reads `.env` with override enabled, verifies model/index dimensional compatibility, and evaluates the existing namespace without changing it. Output is written to:

```text
evaluation/results/E001_dense_baseline/
├── config.json
└── results.json
```

Generated result directories are intentionally git-ignored because they contain model outputs and can be regenerated. See [evaluation/README.md](evaluation/README.md) for the metric definition and result schema.

Run the offline Phase 2 tests with:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

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
├── questions.json           # 15 ground-truth evaluation records
├── run_baseline.py          # Phase 2 command-line runner
└── results/                 # generated experiment config and results

notebooks/
└── phase_1_naive_dense_rag.ipynb

src/i_got_this_rag/
├── baseline.py              # read-only connection to the Phase 1 pipeline
├── evaluation.py            # metrics, result schema, and persistence
└── settings.py              # typed environment configuration

tests/
└── test_phase2_evaluation.py

pyproject.toml               # uv environment and notebook dependencies
```

The full roadmap and requirements are in [product_requirements.md](product_requirements.md).
