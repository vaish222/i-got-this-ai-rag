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

## Phase 4 embedding-model experiments

Phase 4 compares exactly the three local Ollama embedding models required by the PRD:

| Experiment | Model | Dimension | Pinecone index |
|---|---|---:|---|
| `E201_dense_all_minilm` | `all-minilm` | 384 | `i-got-this-all-minilm` |
| `E202_dense_nomic` | `nomic-embed-text` | 768 | `i-got-this-nomic` |
| `E203_dense_mxbai` | `mxbai-embed-large` | 1024 | `i-got-this-mxbai` |

Dimensions are probed from Ollama at runtime rather than assumed. Each model uses a separate Pinecone cosine index, preventing incompatible vector dimensions from being mixed. Existing indexes are reused only after their dimension and metric are validated.

The following remain identical across all three experiments:

- the SHA-256 fingerprint of the 20-document corpus;
- the SHA-256 fingerprint of the exact 20-chunk set;
- 500-token chunks with 75-token overlap;
- dense Top-5 retrieval;
- the `gemma3:1b` generation model and grounded prompt; and
- the SHA-256 fingerprint of the 15-question evaluation dataset.

Run the suite from the repository root:

```bash
uv run python evaluation/run_embedding_experiments.py
```

The runner writes individual experiment artifacts and `evaluation/results/phase4_embeddings/comparison.json`. The comparison records Recall@5, source ranks, retrieval latency, category summaries, failure cases, vector dimensions, and question-by-question results.

The current controlled run measured 0.826 Recall@5 for `all-minilm`, 0.867 for `nomic-embed-text`, and 0.873 for `mxbai-embed-large`. `mxbai-embed-large` had the best aggregate recall and the fewest failed questions, although all three models still had incomplete retrieval on both cross-domain questions.

## Phase 5 retrieval-strategy experiments

Phase 5 compares three configurable retrieval strategies:

- **Dense:** cosine similarity over `embeddinggemma` vectors in the controlled Pinecone namespace.
- **Sparse:** local Okapi BM25 lexical scoring with `k1=1.5` and `b=0.75`.
- **Hybrid:** reciprocal-rank fusion of the dense Top-5 and sparse Top-5 using `rrf_k=60`, returning a final Top-5.

The sparse index is deliberately local and deterministic; it operates on the exact same chunks sent to Pinecone and adds no new hosted service or dependency. Hybrid result records include the dense and sparse component ranks that produced each fused rank.

The following remain identical across the experiments:

- the SHA-256 fingerprints of the 20-document corpus and 20-chunk set;
- 500-token chunks with 75-token overlap;
- the `embeddinggemma` dense embedding model and Pinecone index;
- final Top-K of five;
- the `gemma3:1b` generation model and grounded prompt; and
- the SHA-256 fingerprint of the 15-question evaluation dataset.

Run the suite from the repository root:

```bash
uv run python evaluation/run_retrieval_experiments.py
```

The current run measured 0.900 Recall@5 for dense retrieval, 0.738 for sparse BM25, and 0.840 for hybrid RRF. All strategies achieved perfect exact-lookup and date/deadline recall. Hybrid improved exact-lookup mean expected-source rank, but both lexical strategies performed substantially worse on cross-domain questions. Dense therefore remains the selected retrieval strategy for this controlled corpus.

## Phase 6 reranking experiments

Phase 6 compares two dense retrieval paths:

- **Baseline:** retrieve and generate from Pinecone Top-5 with reranking disabled.
- **Reranked:** retrieve Pinecone Top-20, score those candidates with deterministic local BM25, and generate from the reranked Top-5.

The BM25 candidate reranker is the PRD's compatible local reranker option. It adds no hosted service or model and uses the original question to score only the dense candidate set. The generation model and grounded prompt remain unchanged.

Run the suite from the repository root:

```bash
uv run python evaluation/run_reranking_experiments.py
```

Each question result records:

- all pre-rerank candidates with dense score and candidate rank;
- the final Top-5 with candidate rank, dense score, and BM25 reranker score;
- candidate retrieval, reranking, generation, and total latency;
- candidate expected-source ranks and candidate recall; and
- whether each missing final source was absent from the candidate set or lost during reranking.

The controlled run found 1.000 candidate recall for dense Top-20, proving that all expected evidence reached the reranker. BM25 then reduced final Recall@5 from 0.900 to 0.738, producing six reranking-failure questions across semantic, cross-document, and cross-domain categories. The reranker is therefore not selected for production.

## Phase 7 metadata-aware retrieval experiments

Phase 7 compares the selected unfiltered dense Top-5 path against metadata-aware dense Top-5 retrieval. The controlled variables remain the 20-document corpus, 500/75 chunks, `embeddinggemma`, Pinecone index, `gemma3:1b`, grounded generation prompt, and 15 evaluation questions.

The metadata-aware path has three deliberately separate operations:

1. A deterministic analyzer extracts high-confidence domain, exact anonymous person ID, document type, event type, event date, general status, RSVP status, and gift status constraints.
2. Those constraints become Pinecone metadata filters while the semantic query remains byte-for-byte unchanged.
3. When a filter returns fewer than five chunks, unfiltered dense results fill the remaining slots with chunk-ID deduplication.

The indexed Phase 7 chunks retain the original metadata and add normalized filter facets. Event dates and status facets are extracted conservatively from each chunk; boolean facet keys make multi-value person, event, and status filtering unambiguous in Pinecone.

Run the controlled suite from the repository root:

```bash
uv run python evaluation/run_metadata_experiments.py
```

Per-question results record the extracted constraints, exact Pinecone filter, filter and fallback result counts, unchanged retrieval query, metadata-analysis latency, retrieval ranks, answer, citations, and existing latency metrics. `comparison.json` classifies every answerable question as improved, degraded, or unchanged using Recall@5 first and expected-source rank second.

The current run measured identical 0.900 Recall@5 for both paths. Metadata filtering improved expected-source ordering on three answerable questions, degraded four, and left six unchanged. Mean expected-source rank improved slightly from 2.423 to 2.385, while mean retrieval latency rose from 194 ms to 371 ms because filtered searches frequently needed a second dense fallback query. The filtered path remains optional rather than becoming the selected default.

## Phase 8 query-transformation experiments

Phase 8 compares exactly three query strategies while holding the corpus, 500/75 chunks, `embeddinggemma`, Pinecone index, dense Top-5 output, generation model, grounded prompt, and evaluation questions fixed:

- **Original:** retrieve with the user's question unchanged.
- **Rewrite:** ask local `gemma3:1b` for one retrieval-focused query and search only that rewrite.
- **Multi-query:** retrieve with the original plus two generated queries, then fuse the three rankings with RRF (`rrf_k=60`) to a final Top-5.

The transformation guard extracts protected anonymous IDs, dates, times, relative-date phrases, event names, domain terms, deadline language, RSVP status, and gift terminology. It restores protected terms omitted by the model and removes newly invented IDs, dates, years, and times. Multi-query parsing accepts the first valid JSON array when the small local model emits duplicate fenced arrays. These repairs and the raw model output remain in the result record.

Run the suite from the repository root:

```bash
uv run python evaluation/run_query_experiments.py
```

Each question records the original query, generated queries, actual retrieval queries, protected terms, guard repairs, raw transformation output, fusion components, transformation latency, vector-search latency, retrieved sources, answer, and citations. The comparison explicitly lists quality-reduction and recall-reduction question IDs for both transformed strategies.

The verified run measured:

| Query strategy | Recall@5 | Mean found-source rank | Transformation latency | Total retrieval latency |
|---|---:|---:|---:|---:|
| Original | **0.900** | 2.577 | 0 ms | **196 ms** |
| One rewrite | 0.865 | **2.042** | 295 ms | 480 ms |
| Original + two rewrites | 0.890 | 2.440 | 430 ms | 1,036 ms |

Single rewriting improved three answerable questions, degraded four, and left six unchanged; recall fell on Q004 and Q012. Multi-query improved two, degraded five, and left six unchanged; recall fell on Q012. Original dense retrieval remains selected because neither transformation improved aggregate recall and both added substantial latency.

## Phase 9 LangGraph workflow

Phase 9 refactors the selected components into a conditional LangGraph `StateGraph` without replacing the existing retrieval, metadata-analysis, guarded-rewrite, or grounded-generation modules. Its nodes cover query analysis, query rewriting, metadata construction, retrieval, reranking, evidence grading, generation, and grounding verification.

Run one question from the repository root:

```bash
uv run python evaluation/run_agentic_rag.py \
  --question "Which invitations still need an RSVP?"
```

The workflow uses the original query for the first dense Top-5 retrieval. Metadata constraints can filter that attempt, with dense fallback filling short result sets. Weak evidence can trigger exactly one guarded LLM rewrite and an unfiltered dense retry. The reranking node remains disabled because Phase 6 found that the tested BM25 reranker reduced final Recall@5.

Evidence grading occurs before generation. If the local model omits citations, deterministic attribution adds a source label only when the claim's concrete facts and informative terms match that source. The grounding verifier then requires valid claim-level citations and checks support for IDs, dates, times, numeric facts, and claim terms. Weak evidence after the retry or failed grounding produces the standard explicit insufficient-information response.

The runner writes `evaluation/results/phase9_agentic/run.json`, including the full serializable graph state, node trace, query and retrieval histories, evidence grades, answer, citations, grounding decision, and latency. It rebuilds only namespaces beginning with `phase9-`.

This runner validates a single Phase 9 workflow. It intentionally does not run the Phase 10 cross-version evaluation or populate Recall@5, faithfulness, and latency comparisons.

## Phase boundary

Phase 9 implements LangGraph orchestration, evidence grading, one bounded retry, and grounding-based refusal. Phase 10 final cross-version evaluation and later dashboard/UI work remain intentionally unimplemented.
