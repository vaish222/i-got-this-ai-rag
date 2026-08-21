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

## Phase 2 baseline runner

The Phase 2 runner evaluates the existing Phase 1 dense Pinecone namespace. It is read-only: indexing remains the responsibility of the Phase 1 notebook.

Prerequisites:

1. Populate `.env` and run the Phase 1 notebook through the Pinecone indexing cell.
2. Keep Ollama running with the configured embedding and chat models available.
3. Run the evaluator from the repository root:

```bash
uv run python evaluation/run_baseline.py
```

Optional experiment labels and output location can be supplied without changing retrieval behavior:

```bash
uv run python evaluation/run_baseline.py \
  --experiment-id E001_dense_baseline \
  --experiment-name "Phase 2 dense baseline" \
  --output-root evaluation/results
```

The runner always validates and executes all 15 questions. `RAG_TOP_K` must be at least 5 so Recall@5 can be calculated.

## Recorded output

Each experiment directory contains:

- `config.json`: active Pinecone, model, chunking, retrieval, dataset, and time-reference settings. Secrets are never recorded.
- `results.json`: one result per question plus aggregate metrics. It includes the SHA-256 hash of `config.json`; both files include the evaluation dataset's SHA-256 hash.

Each question result records:

- the question, category, answerable flag, expected answer, and expected sources;
- retrieved document IDs, chunk IDs, source paths, ranks, and similarity scores;
- the first rank for every expected source and the best expected-source rank;
- generated answer and citations resolved back to retrieved chunks;
- Recall@5 for answerable questions; and
- retrieval, LLM/generation, and total latency in seconds.

Aggregate Recall@5 is the macro average of per-question source recall across the 13 answerable questions:

```text
expected sources retrieved in ranks 1–5 / total expected sources
```

The two deliberately unanswerable questions have no expected sources, so their per-question Recall@5 is `null` and they are excluded from the aggregate retrieval metric. Their generated answers, citations, and latencies are still recorded.

## Phase 3 chunk-size experiments

Phase 3 compares exactly four chunk sizes configured in `config/experiments/`:

| Experiment | Chunk size | Overlap | Pinecone namespace |
|---|---:|---:|---|
| `E101_dense_chunk250` | 250 | 75 | `phase3-chunk-250` |
| `E102_dense_chunk500` | 500 | 75 | `phase3-chunk-500` |
| `E103_dense_chunk750` | 750 | 75 | `phase3-chunk-750` |
| `E104_dense_chunk1000` | 1000 | 75 | `phase3-chunk-1000` |

Overlap stays fixed so chunk size is the only content variable. The runner also locks the following controlled variables across all four experiments:

- the SHA-256 fingerprint of the same 20 source documents;
- embedding model and Pinecone index;
- dense retrieval strategy and Top-K;
- LLM and prompt behavior; and
- the SHA-256 fingerprint of the same 15-question evaluation dataset.

Run the suite from the repository root:

```bash
uv run python evaluation/run_chunk_experiments.py
```

Each run replaces only namespaces beginning with `phase3-`. A runtime guard rejects any attempt to rebuild the baseline or another unprefixed namespace. The runner writes individual `config.json` and `results.json` files under `evaluation/results/phase3_chunking/`, plus `comparison.json` containing:

- overall Recall@5, expected-source rank, and latency for every chunk size;
- category-level Recall@5 and latency;
- failed question IDs and missing expected source IDs grouped by category; and
- a question-by-question expected-source rank comparison across all four experiments.

The current controlled run found that 500, 750, and 1000 tokens tie at 0.900 Recall@5. The 250-token configuration scored 0.829 because it added cross-document misses on top of the cross-domain failures shared by every configuration. This is a measurement result only; Phase 3 does not change the production baseline based on it.

## Phase boundary

Phase 3 varies chunk size only. Alternate embeddings, sparse or hybrid retrieval, reranking, metadata-aware retrieval, query transformation, LangGraph, and UI work remain intentionally unimplemented.
