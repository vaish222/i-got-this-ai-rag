# I Got This — What's Next?

A privacy-first personal and family knowledge assistant, developed incrementally from a controlled RAG dataset.

## Current status: Phase 10 complete with Streamlit UI and experiment dashboard

The Phase 0 controlled dataset remains unchanged:

- 20 synthetic documents across school, children's activities, personal learning, volunteer, household, social, and family domains;
- YAML-style metadata on every source document;
- 15 evaluation questions with expected answers and expected source documents;
- exactly two deliberately unanswerable questions; and
- privacy exclusions for local personal documents and secrets.

The corpus uses fictional anonymous identifiers and a fixed reference date of **August 20, 2026**, which makes relative-time evaluation reproducible. See [data/README.md](data/README.md) for dataset conventions and [evaluation/README.md](evaluation/README.md) for the evaluation category distribution.

Phase 1 is available as both an executable walkthrough notebook, [notebooks/phase_1_naive_dense_rag.ipynb](notebooks/phase_1_naive_dense_rag.ipynb), and a repeatable Python runner, [evaluation/run_phase1.py](evaluation/run_phase1.py). It provides:

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

Phase 6 adds optional two-stage reranking and compares the selected dense Top-5 baseline against dense Top-20 followed by a deterministic local BM25 candidate reranker and final Top-5. Candidate evidence and latency are recorded separately from reranking outcomes.

Phase 7 adds optional metadata-aware dense retrieval. A deterministic analyzer extracts high-confidence domain, person, document-type, event-type, date, status, RSVP-status, and gift-status constraints without rewriting the question. Pinecone applies those filters first, then dense fallback fills any remaining Top-5 slots so narrow metadata cannot reduce the evidence count.

Phase 8 adds local LLM query rewriting and multi-query retrieval. It compares the original dense query, one guarded rewrite, and the original plus two guarded rewrites fused with reciprocal-rank fusion. Exact terms are preserved, newly invented dates/times/IDs are removed, and raw model output and guard repairs remain observable.

Phase 9 wraps the selected retrieval and generation components in a LangGraph `StateGraph`. It analyzes intent, builds metadata filters, retrieves dense Top-5 evidence, retains a reranking node with the Phase 6-selected disabled setting, grades evidence, permits one guarded query-rewrite retry, verifies grounding, and returns the explicit insufficient-information response when evidence or support checks fail.

Phase 10 compares all eight PRD system versions on the same 15-question dataset. It reuses the recorded results for six versions, runs the two missing end-to-end combinations, adds deterministic claim-level faithfulness and refusal scoring, measures average and p95 latency, and records per-question gains, losses, costs, and a recommendation.

The Streamlit application provides the PRD's focused question-and-answer experience plus a read-only experiment dashboard. The Ask tab uses the selected dense pipeline and displays grounded answers with resolved source citations. The Experiments tab compares the eight measured Phase 10 versions and explains what changed, what stayed constant, gains, regressions, latency cost, and whether the result justified the change.

No other post-Phase-10 work is included. The RAG developer/debug mode, deployment, and later phases remain unimplemented.

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

## Run Phase 1 from Python

The Python runner executes the same reusable loading, chunking, local embedding, dense retrieval, and grounded-generation behavior:

```bash
uv run python evaluation/run_phase1.py \
  --question "What do we need to bring to the neighborhood potluck?"
```

It creates the configured Pinecone index when missing, indexes an empty namespace, and otherwise reuses existing namespace vectors by default. Rebuild only the configured namespace explicitly with:

```bash
uv run python evaluation/run_phase1.py --rebuild-namespace
```

The runner never deletes the whole index. Its single-question artifact is written to `evaluation/results/phase1/run.json` and contains the public configuration, corpus and chunk fingerprints, indexing action, retrieved chunks, resolved citations, answer, and latency. It does not execute or score the Phase 2 evaluation dataset.

## Run the Streamlit user interface

Populate `.env`, keep Ollama running, and index the configured Pinecone namespace with the Phase 1 runner if it is empty. Then start the application from the repository root:

```bash
uv sync
uv run streamlit run app.py
```

The Ask tab reads the same typed settings and `.env` file as the Python runners. It connects only after a non-empty question is submitted, caches the read-only RAG connection for subsequent questions, and displays only sources actually cited in the answer. Connection and configuration failures are shown in the page instead of crashing it.

The Experiments tab reads `evaluation/results/phase10_final/comparison.json`; it never launches or reruns experiments. If that artifact is unavailable, the page provides the Phase 10 runner command needed to generate it. The dashboard includes the measured configuration matrix, Recall@5, strict faithfulness, average latency, the selected recommendation, and a per-experiment PRD trade-off analysis.

The application intentionally has no live configuration editor, retrieval diagnostics, or RAG developer/debug mode.

## Notebook interfaces for Phases 2–10

Every implemented phase now has a notebook under `notebooks/`. Phases 2–10 delegate to the same tested command-line runners documented below, then load the generated JSON artifacts for interactive inspection. This keeps pipeline logic in `src/i_got_this_rag/` instead of duplicating it in notebook cells.

The experiment notebooks default to `RUN_EXPERIMENT = False`. Review their parameters and configuration previews, change the switch to `True`, and run the execution cell when you are ready to connect to Ollama and Pinecone. Namespace-mutating runners retain their existing phase-prefix guards.

| Phase | Notebook |
|---:|---|
| 1 | [Naive dense RAG](notebooks/phase_1_naive_dense_rag.ipynb) |
| 2 | [Baseline evaluation](notebooks/phase_2_baseline_evaluation.ipynb) |
| 3 | [Chunk-size experiments](notebooks/phase_3_chunking_experiments.ipynb) |
| 4 | [Embedding-model experiments](notebooks/phase_4_embedding_experiments.ipynb) |
| 5 | [Retrieval-strategy experiments](notebooks/phase_5_retrieval_experiments.ipynb) |
| 6 | [Reranking experiments](notebooks/phase_6_reranking_experiments.ipynb) |
| 7 | [Metadata-aware retrieval](notebooks/phase_7_metadata_experiments.ipynb) |
| 8 | [Query transformation](notebooks/phase_8_query_transformation_experiments.ipynb) |
| 9 | [LangGraph agentic RAG](notebooks/phase_9_langgraph_agentic_rag.ipynb) |
| 10 | [Final evaluation](notebooks/phase_10_final_evaluation.ipynb) |

## Run the Phase 2 baseline evaluation

First run the Phase 1 notebook through its indexing cell or run `evaluation/run_phase1.py` so the configured Pinecone namespace exists. Keep Ollama running, then execute:

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

## Run the Phase 6 reranking experiments

Keep Ollama running and execute:

```bash
uv run python evaluation/run_reranking_experiments.py
```

The runner compares dense Top-5 without reranking against dense Top-20 candidates reranked by local BM25 to a final Top-5. It rebuilds only the dedicated `phase6-*` namespace and writes:

```text
evaluation/results/phase6_reranking/
├── comparison.json
├── E401_dense_top5/{config.json,results.json}
└── E402_dense_top20_bm25/{config.json,results.json}
```

The verified controlled run produced:

| Configuration | Candidate recall | Final Recall@5 | Reranking failures | Mean reranking latency |
|---|---:|---:|---:|---:|
| Dense Top-5, no reranker | 0.900 | **0.900** | 0 | 0 ms |
| Dense Top-20 → BM25 → Top-5 | **1.000** | 0.738 | 6 | 1.9 ms |

All expected evidence reached the Top-20 candidate set, but BM25 discarded relevant semantic, cross-document, and cross-domain sources. The tested reranker therefore remains disabled and dense Top-5 remains the selected production path.

## Run the Phase 7 metadata experiments

Keep Ollama running and execute:

```bash
uv run python evaluation/run_metadata_experiments.py
```

Both experiments search the same metadata-enriched `phase7-*` namespace. The unfiltered experiment performs the selected dense Top-5 search. The filtered experiment analyzes the original question, sends supported constraints as a separate Pinecone metadata filter, and fills short filtered result sets from the unchanged dense query. It does not rewrite or expand the query.

The runner writes:

```text
evaluation/results/phase7_metadata/
├── comparison.json
├── E501_dense_unfiltered/{config.json,results.json}
└── E502_dense_metadata_filtered/{config.json,results.json}
```

The verified controlled run produced:

| Configuration | Recall@5 | Mean expected-source rank | Mean retrieval latency | Filters applied |
|---|---:|---:|---:|---:|
| Dense Top-5, unfiltered | **0.900** | 2.423 | **194 ms** | 0/15 |
| Metadata-filtered + dense fallback | **0.900** | **2.385** | 371 ms | 13/15 |

Filtering improved expected-source ordering on three answerable questions, degraded it on four, and left six unchanged; the two unanswerable questions are unscored. It improved cross-document ranking and the overall mean expected-source rank slightly, but exact-lookup and semantic mean ranks became worse. Since it added latency without improving Recall@5, metadata filtering remains available but is not selected as the default retrieval path.

## Run the Phase 8 query-transformation experiments

Keep Ollama running and execute:

```bash
uv run python evaluation/run_query_experiments.py
```

Phase 8 compares three dense Top-5 paths over one controlled `phase8-*` namespace:

- the original user question;
- one `gemma3:1b` retrieval-focused rewrite; and
- the original plus two generated queries, fused with RRF (`rrf_k=60`).

The rewriter is guarded: anonymous IDs, dates, times, event names, domain terms, deadlines, RSVP status, and gift terminology must survive transformation. Missing terms are restored, and newly invented IDs, dates, years, or times are removed. The original question is always used for answer generation.

The runner writes:

```text
evaluation/results/phase8_query_transformation/
├── comparison.json
├── E601_dense_original_query/{config.json,results.json}
├── E602_dense_llm_rewrite/{config.json,results.json}
└── E603_dense_multi_query/{config.json,results.json}
```

The verified controlled run produced:

| Query strategy | Recall@5 | Mean found-source rank | Transformation latency | Total retrieval latency |
|---|---:|---:|---:|---:|
| Original query | **0.900** | 2.577 | 0 ms | **196 ms** |
| One LLM rewrite | 0.865 | **2.042** | 295 ms | 480 ms |
| Original + two rewrites, RRF | 0.890 | 2.440 | 430 ms | 1,036 ms |

The lower rewrite rank applies only to expected sources that were still retrieved; it does not offset the recall loss. Single-query rewriting reduced recall on Q004 and Q012. Multi-query retrieval reduced recall on Q012 and also worsened source ordering on four other questions. Original dense retrieval therefore remains the selected default.

## Run the Phase 9 LangGraph workflow

Keep Ollama running, configure Pinecone in `.env`, and execute one question through the graph:

```bash
uv run python evaluation/run_agentic_rag.py \
  --question "Which invitations still need an RSVP?"
```

The graph follows the Phase 9 path:

```text
analyze → rewrite → metadata → retrieve → rerank → grade
                                                    ├─ sufficient → generate → verify → answer/refuse
                                                    └─ weak → retry once → generate or refuse
```

The first retrieval keeps the original query and can use metadata filtering with dense fallback. If evidence is weak, the only allowed retry uses the existing guarded rewrite and broadens retrieval by removing the metadata filter. Reranking is an explicit measured no-op because the Phase 6 BM25 reranker was not selected. When the local model omits citations, a deterministic attributor adds one only if the claim's concrete facts and informative terms match a retrieved source. The grounding verifier then requires resolvable claim-level citations and rejects unsupported IDs, dates, times, numbers, or claims.

The runner rebuilds only its guarded `phase9-*` namespace and writes the state, node trace, query history, retrieval attempts, evidence grades, citations, grounding result, and latencies to:

```text
evaluation/results/phase9_agentic/run.json
```

This remains the Phase 9 single-workflow artifact. Use the Phase 10 runner for the cross-version comparison.

## Run the Phase 10 final evaluation

Phase 10 evaluates the exact eight versions named in the PRD. Keep Ollama running, configure Pinecone in `.env`, and make sure the Phase 2–8 experiment artifacts are present before running:

```bash
uv run python evaluation/run_final_evaluation.py
```

The runner re-scores six recorded experiments and executes the two combinations that did not previously have standalone artifacts: hybrid Top-20 followed by the BM25 reranker, and the LangGraph workflow over all 15 questions. It rebuilds only the guarded `phase10-*` namespace.

Faithfulness is a conservative deterministic full-answer grounding score. An answerable response receives credit only when every material claim has a valid retrieved citation and the grounding verifier accepts its facts; an unanswerable question receives credit only for the explicit refusal. It is reproducible and auditable, but it is stricter than a semantic LLM judge.

The verified run measured:

| Version | Recall@5 | Faithfulness | Avg. latency | p95 latency |
|---|---:|---:|---:|---:|
| Baseline dense RAG | **0.900** | 0.400 | **1.453 s** | 3.257 s |
| Best chunking (500/75) | **0.900** | 0.400 | 1.532 s | 3.476 s |
| Best embedding (`mxbai-embed-large`) | 0.873 | **0.467** | 1.649 s | 4.410 s |
| Hybrid retrieval | 0.840 | **0.467** | 1.499 s | 3.679 s |
| Hybrid + BM25 reranker | 0.738 | **0.467** | 1.627 s | 4.427 s |
| Metadata-aware retrieval | **0.900** | 0.400 | 1.810 s | 3.789 s |
| One guarded query rewrite | 0.865 | 0.333 | 1.704 s | 3.770 s |
| LangGraph workflow | **0.900** | 0.333 | 1.789 s | 3.496 s |

Baseline dense RAG is the current recommendation: it ties for best Recall@5, is fastest, and correctly refuses both unanswerable questions. None of the tested versions reaches the 0.90 faithfulness target, so generation grounding remains the main improvement area. Historical experiments were run at different times, so latency differences are measured costs rather than a simultaneous benchmark.

Outputs are written to:

```text
evaluation/results/phase10_final/
├── config.json
├── comparison.json
└── analysis.md
```

Run all offline tests with:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

## Repository layout

```text
app.py                      # simple Streamlit question-and-answer interface

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

config/reranking_experiments/
├── reranking_disabled.yaml
└── reranking_bm25.yaml

config/metadata_experiments/
├── metadata_unfiltered.yaml
└── metadata_filtered.yaml

config/query_experiments/
├── query_original.yaml
├── query_rewrite.yaml
└── query_multi.yaml

config/
├── agentic_rag.yaml         # Phase 9 graph and retry configuration
└── final_evaluation.yaml    # Phase 10 exact comparison matrix

evaluation/
├── README.md
├── questions.json           # 15 ground-truth evaluation records
├── run_phase1.py            # Phase 1 indexing and single-question runner
├── run_baseline.py          # Phase 2 command-line runner
├── run_chunk_experiments.py # Phase 3 controlled experiment runner
├── run_embedding_experiments.py # Phase 4 controlled experiment runner
├── run_retrieval_experiments.py # Phase 5 controlled experiment runner
├── run_reranking_experiments.py # Phase 6 controlled experiment runner
├── run_metadata_experiments.py # Phase 7 controlled experiment runner
├── run_query_experiments.py # Phase 8 controlled experiment runner
├── run_agentic_rag.py       # Phase 9 single-workflow runner
├── run_final_evaluation.py  # Phase 10 cross-version runner
└── results/                 # generated experiment config and results

notebooks/
├── phase_1_naive_dense_rag.ipynb
├── phase_2_baseline_evaluation.ipynb
├── phase_3_chunking_experiments.ipynb
├── phase_4_embedding_experiments.ipynb
├── phase_5_retrieval_experiments.ipynb
├── phase_6_reranking_experiments.ipynb
├── phase_7_metadata_experiments.ipynb
├── phase_8_query_transformation_experiments.ipynb
├── phase_9_langgraph_agentic_rag.ipynb
└── phase_10_final_evaluation.ipynb

src/i_got_this_rag/
├── baseline.py              # read-only connection to the Phase 1 pipeline
├── chunk_experiments.py     # experiment config and guarded namespace indexing
├── embedding_experiments.py # dimension-safe model/index experiments
├── evaluation.py            # metrics, result schema, and persistence
├── ingestion.py             # reusable Phase 1 loading and chunking behavior
├── retrieval.py             # dense, BM25, and RRF retrieval strategies
├── retrieval_experiments.py # Phase 5 config and guarded indexing
├── reranking.py             # optional candidate reranking pipeline
├── reranking_experiments.py # Phase 6 config and guarded indexing
├── metadata_retrieval.py    # Phase 7 facet extraction and filtered dense retrieval
├── metadata_experiments.py  # Phase 7 config and guarded indexing
├── query_transformation.py  # Phase 8 rewriting, guards, and multi-query fusion
├── query_experiments.py     # Phase 8 config, impact analysis, and guarded indexing
├── agentic_rag.py           # Phase 9 LangGraph state, nodes, and routing
├── agentic_experiments.py   # Phase 9 config and guarded namespace indexing
├── final_evaluation.py      # Phase 10 scoring, comparison, and analysis
├── experiment_dashboard.py  # measured experiment matrix and trade-off projection
├── user_interface.py        # UI-facing answer and citation projection
└── settings.py              # typed environment configuration

tests/
├── test_phase_notebooks.py
├── test_phase1_runner.py
├── test_phase2_evaluation.py
├── test_phase3_chunking.py
├── test_phase4_embeddings.py
├── test_phase5_retrieval.py
├── test_phase6_reranking.py
├── test_phase7_metadata.py
├── test_phase8_query_transformation.py
├── test_phase9_agentic_rag.py
├── test_phase10_final_evaluation.py
├── test_experiment_dashboard.py
└── test_streamlit_user_interface.py

pyproject.toml               # uv environment and notebook dependencies
```

The full roadmap and requirements are in [product_requirements.md](product_requirements.md).
