# I Got This — What's Next?

A privacy-first personal and family knowledge assistant, developed incrementally from a controlled RAG dataset.

## Current status: Phase 5 complete

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

Phase 3 adds controlled chunk-size experiments for 250, 500, 750, and 1000 tokens. The corpus, 75-token overlap, embedding model, Pinecone index, dense Top-5 retrieval, LLM, and evaluation questions stay fixed. Each configuration uses a dedicated `phase3-*` namespace and records Recall@5, expected-source ranks, latency, and retrieval failures by question category.

Phase 4 adds a controlled local embedding-model comparison across `all-minilm`, `nomic-embed-text`, and `mxbai-embed-large`. It reuses the exact selected 500/75 chunk set and creates a separate dimension-compatible Pinecone index for every model.

Phase 5 adds configurable dense, sparse BM25, and hybrid reciprocal-rank-fusion retrieval. All three strategies use the same 20 chunks, `embeddinggemma` dense model, Top-5 output, `gemma3:1b` generation model, prompt, and evaluation dataset.

No Phase 6+ work is included: there is no candidate reranking, metadata filtering, query rewriting, LangGraph, or UI.

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

## Run the Phase 3 chunking experiments

Keep Ollama running and execute:

```bash
uv run python evaluation/run_chunk_experiments.py
```

The runner rebuilds only the four namespaces prefixed with `phase3-`; it cannot delete the Phase 1 baseline namespace. It writes one reproducible config/result pair per chunk size plus a cross-experiment comparison:

```text
evaluation/results/phase3_chunking/
├── comparison.json
├── E101_dense_chunk250/{config.json,results.json}
├── E102_dense_chunk500/{config.json,results.json}
├── E103_dense_chunk750/{config.json,results.json}
└── E104_dense_chunk1000/{config.json,results.json}
```

The verified run on the controlled corpus produced:

| Chunk size | Chunks | Recall@5 | Retrieval failures |
|---:|---:|---:|---:|
| 250 | 25 | 0.829 | 4 |
| 500 | 20 | 0.900 | 2 |
| 750 | 20 | 0.900 | 2 |
| 1000 | 20 | 0.900 | 2 |

Based on these results, the selected baseline remains **500-token chunks with a 75-token overlap**. This matches the active environment example and typed configuration defaults.

## Run the Phase 4 embedding experiments

Install the three required local models once:

```bash
ollama pull all-minilm
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
```

Then run:

```bash
uv run python evaluation/run_embedding_experiments.py
```

The runner probes each model's vector dimension and creates or validates three separate cosine indexes. It rebuilds only the namespace prefixed `phase4-` in each index. Output is written to:

```text
evaluation/results/phase4_embeddings/
├── comparison.json
├── E201_dense_all_minilm/{config.json,results.json}
├── E202_dense_nomic/{config.json,results.json}
└── E203_dense_mxbai/{config.json,results.json}
```

The verified controlled run produced:

| Embedding model | Dimension | Recall@5 | Retrieval failures |
|---|---:|---:|---:|
| `all-minilm` | 384 | 0.826 | 4 |
| `nomic-embed-text` | 768 | 0.867 | 3 |
| `mxbai-embed-large` | 1024 | 0.873 | 2 |

`mxbai-embed-large` has the strongest aggregate Phase 4 result. The active baseline embedding model is not changed automatically; selecting a new production model is a separate decision.

## Run the Phase 5 retrieval experiments

Keep Ollama running and execute:

```bash
uv run python evaluation/run_retrieval_experiments.py
```

Dense retrieval uses Pinecone, sparse retrieval uses a deterministic local BM25 index, and hybrid retrieval combines their Top-5 rankings with reciprocal-rank fusion. The final result count remains five for every strategy; Phase 5 does not retrieve 20 candidates or rerank them.

The runner rebuilds only the dedicated `phase5-*` dense namespace and writes:

```text
evaluation/results/phase5_retrieval/
├── comparison.json
├── E301_dense/{config.json,results.json}
├── E302_sparse/{config.json,results.json}
└── E303_hybrid/{config.json,results.json}
```

The verified controlled run produced:

| Strategy | Recall@5 | Exact | Semantic | Date/deadline | Cross-document | Cross-domain |
|---|---:|---:|---:|---:|---:|---:|
| Dense | **0.900** | 1.000 | 1.000 | 1.000 | 1.000 | **0.350** |
| Sparse BM25 | 0.738 | 1.000 | 0.722 | 1.000 | 0.667 | 0.050 |
| Hybrid RRF | 0.840 | 1.000 | 1.000 | 1.000 | 0.833 | 0.125 |

Dense retrieval remains the best production strategy for this corpus. Hybrid improves the mean expected-source rank for exact lookups from 1.50 to 1.25, but its lexical influence displaces too much cross-domain evidence at Top-5.

Run all offline tests with:

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

config/experiments/
├── chunk_250.yaml
├── chunk_500.yaml
├── chunk_750.yaml
└── chunk_1000.yaml

config/embedding_experiments/
├── embedding_all_minilm.yaml
├── embedding_nomic.yaml
└── embedding_mxbai.yaml

config/retrieval_experiments/
├── retrieval_dense.yaml
├── retrieval_sparse.yaml
└── retrieval_hybrid.yaml

evaluation/
├── README.md
├── questions.json           # 15 ground-truth evaluation records
├── run_baseline.py          # Phase 2 command-line runner
├── run_chunk_experiments.py # Phase 3 controlled experiment runner
├── run_embedding_experiments.py # Phase 4 controlled experiment runner
├── run_retrieval_experiments.py # Phase 5 controlled experiment runner
└── results/                 # generated experiment config and results

notebooks/
└── phase_1_naive_dense_rag.ipynb

src/i_got_this_rag/
├── baseline.py              # read-only connection to the Phase 1 pipeline
├── chunk_experiments.py     # experiment config and guarded namespace indexing
├── embedding_experiments.py # dimension-safe model/index experiments
├── evaluation.py            # metrics, result schema, and persistence
├── ingestion.py             # reusable Phase 1 loading and chunking behavior
├── retrieval.py             # dense, BM25, and RRF retrieval strategies
├── retrieval_experiments.py # Phase 5 config and guarded indexing
└── settings.py              # typed environment configuration

tests/
├── test_phase2_evaluation.py
├── test_phase3_chunking.py
├── test_phase4_embeddings.py
└── test_phase5_retrieval.py

pyproject.toml               # uv environment and notebook dependencies
```

The full roadmap and requirements are in [product_requirements.md](product_requirements.md).
