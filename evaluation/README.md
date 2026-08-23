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

## Notebook interfaces

Phases 2–10 have matching notebooks in `notebooks/`. Each notebook delegates execution to the phase's tested runner and reads the same generated JSON artifacts described in this document. The notebooks default to dry-run mode (`RUN_EXPERIMENT = False`) so opening or running exploratory cells cannot unexpectedly rebuild a Pinecone namespace or start a multi-model experiment.

Select the repository `.venv` kernel, review the parameters, set `RUN_EXPERIMENT = True`, and run the execution cell when ready. Phase-specific namespace guards and controlled-variable validation remain enforced by the underlying runner.

## Phase 2 baseline runner

The Phase 2 runner evaluates the existing Phase 1 dense Pinecone namespace. It is read-only: indexing remains the responsibility of the Phase 1 notebook or `evaluation/run_phase1.py`.

Prerequisites:

1. Populate `.env` and either run the Phase 1 notebook through the Pinecone indexing cell or run `uv run python evaluation/run_phase1.py`.
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

This runner remains the single-question Phase 9 interface. The Phase 10 runner performs the full cross-version comparison.

## Phase 10 final evaluation

Phase 10 compares the exact eight system versions required by the PRD: baseline dense RAG, selected 500/75 chunking, selected `mxbai-embed-large` embedding, hybrid retrieval, hybrid plus BM25 reranking, metadata-aware retrieval, one guarded query rewrite, and the LangGraph workflow.

Run it from the repository root after generating the Phase 2–8 source artifacts:

```bash
uv run python evaluation/run_final_evaluation.py
```

Six versions are reconstructed from their recorded per-question artifacts. The runner executes the two missing end-to-end variants over all 15 questions, using a guarded `phase10-final-500` namespace: hybrid RRF Top-20 followed by BM25 candidate reranking to Top-5, and the Phase 9 LangGraph workflow. Source artifact hashes are retained in the comparison for provenance.

Faithfulness uses `phase10-deterministic-grounding-v1`. Answerable responses score 1 only when all material claims resolve to retrieved citations and pass the deterministic grounding verifier; unanswerable responses score 1 only for the standard explicit refusal. This deliberately strict, reproducible lower-bound metric should not be interpreted as an LLM-as-judge semantic score.

The verified comparison is:

| Version | Recall@5 | Faithfulness | Avg. latency | Correct refusal rate |
|---|---:|---:|---:|---:|
| Baseline dense RAG | **0.900** | 0.400 | **1.453 s** | 1.000 |
| Best chunking (500/75) | **0.900** | 0.400 | 1.532 s | 1.000 |
| Best embedding (`mxbai-embed-large`) | 0.873 | **0.467** | 1.649 s | 1.000 |
| Hybrid retrieval | 0.840 | **0.467** | 1.499 s | 1.000 |
| Hybrid + BM25 reranker | 0.738 | **0.467** | 1.627 s | 1.000 |
| Metadata-aware retrieval | **0.900** | 0.400 | 1.810 s | 1.000 |
| One guarded query rewrite | 0.865 | 0.333 | 1.704 s | 1.000 |
| LangGraph workflow | **0.900** | 0.333 | 1.789 s | 1.000 |

Baseline dense RAG is selected because it ties for the highest Recall@5, has the lowest average latency, and refuses both unanswerable questions correctly. The embedding and hybrid variants improve strict faithfulness by 0.067 but lose Recall@5. Hybrid reranking has the largest retrieval regression. Metadata filtering preserves aggregate quality but adds latency; rewriting and LangGraph add latency without improving this dataset's aggregate scores. No version reaches the 0.90 faithfulness target.

Generated outputs:

```text
evaluation/results/phase10_final/
├── config.json       # resolved comparison specification and source hashes
├── comparison.json   # aggregate/category/question results and recommendation
└── analysis.md       # concise why-better, why-worse, and cost analysis
```

The historical source experiments were executed at different times, so the latency column is useful for observed cost comparison but is not a simultaneous benchmark.

### Current Streamlit app measurement

The historical eight-version comparison remains immutable. To measure the application after changing conversation handling, deterministic answers, citation repair, deduplication, or plain-language display, run:

```bash
uv run python evaluation/run_current_app_evaluation.py
```

This evaluates the selected dense pipeline through the same `answer_question()` path used by the Ask tab. It runs the original 15 questions for direct comparison and nine targeted UI regression scenarios, then writes `evaluation/results/phase10_current_app/results.json` for the Experiments tab. The regression set includes deterministic weekly planning and exact dated meal-plan checks.

The latest measured results are:

| System | Recall@5 | Strict faithfulness | Correct refusal | Avg. latency | P95 latency | UI regressions |
|---|---:|---:|---:|---:|---:|---:|
| Current Streamlit app | 0.900 | 0.200 | 1.000 | 1.536 s | 4.252 s | 7/9 |

Recall was unchanged versus the historical dense baseline. Strict faithfulness decreased by `0.200`, and average latency increased by `0.083 s`. The weekly summary, “Plan my week,” exact Sunday meal plan, volunteer-week answer, clarification response, additive volunteer follow-up, and birthday humanization checks passed. Pending RSVP aggregation still duplicated an open-commitments section, and the standalone “When is my next volunteer work planned?” regression returned a safe refusal instead of a sourced answer. These are reported as remaining gaps, not silently excluded from the measurement.

The expanded local corpus currently produces 28 chunks, while the active `baseline` Pinecone namespace used by this run contains 20 vectors. Rebuild that namespace before interpreting this current-app artifact as a measurement of the newly added corpus content. The historical eight-version Phase 10 matrix remains tied to its original corpus fingerprint and should not be mixed with newly generated runtime versions from a different corpus.

## Generation grounding ablation

The generation ablation keeps embeddings, Pinecone, chunking, dense Top-5
retrieval, the 15 evaluation questions, and application routing fixed. It compares:

- **A — current:** the original prose generation prompt;
- **B — strict prompt:** Pydantic structured facts with strict grounding rules; and
- **C — strict prompt + filtering:** the same structured generator after deterministic
  question-constraint extraction and direct-relevance filtering of the retrieved Top-5.

Run all three configurations with:

```bash
uv run python evaluation/run_generation_experiments.py
```

The latest controlled run measured:

| Configuration | Recall@5 | Faithfulness | Answer relevance/correctness | Correct refusal | Avg. latency | P95 latency |
|---|---:|---:|---:|---:|---:|---:|
| A — current | 0.900 | 0.200 | 0.262 | 1.000 | **1.502 s** | **4.231 s** |
| B — strict prompt | 0.900 | **0.267** | **0.329** | 1.000 | 2.036 s | 8.738 s |
| C — strict + filtering | 0.900 | **0.267** | 0.303 | 1.000 | 1.622 s | 5.343 s |

Faithfulness remains the existing deterministic full-answer citation-grounding
score. Answer relevance/correctness is deterministic token F1 between the confirmed
facts and the expected answer. Optional suggestions are excluded from both factual
sections and faithfulness scoring. C is the application default because it preserves
the quality gain and exact refusal behavior while substantially reducing B's average
and tail latency. It does not yet include a grounding-validator or retry loop.

Each configuration is saved separately under
`evaluation/results/generation_ablation/<experiment_id>/`, with a resolved
`config.json` and per-question `results.json`. The combined metric and delta table is
`evaluation/results/generation_ablation/comparison.json`.

## Generation model comparison

The model comparison uses Mode B only: the strict grounded prompt with relevance
filtering disabled. Retrieval is executed once for the 15 questions and cached in
memory, so D1, D2, and D3 receive byte-for-byte equivalent serialized Top-5 evidence.
Only the generation provider/model changes.

Configure the two hosted model names and API key in `.env`:

```dotenv
NEBIUS_API_KEY=your-key
NEBIUS_MODEL_1=your-first-model-id
NEBIUS_MODEL_2=your-second-model-id
```

The model IDs remain configuration values in
`config/generation_model_experiments/`; they are not embedded in the RAG pipeline.
Run the comparison with:

```bash
uv run python evaluation/run_generation_model_experiments.py
```

Each model writes its resolved, credential-redacted configuration and its full
per-question results separately:

```text
evaluation/results/
├── D1_current_model/
│   ├── config.json
│   └── results.json
├── D2_nebius_model_1/
│   ├── config.json
│   └── results.json
├── D3_nebius_model_2/
│   ├── config.json
│   └── results.json
└── generation_model_comparison.json
```

Every question records its generated answer, retrieved source IDs, active model,
latency, scores, and any classified model error. Missing keys, invalid configuration,
unavailable models, timeouts, and malformed structured responses are recorded without
terminating the remaining evaluation. A model is ineligible for the dashboard's best
metric highlights unless the run completed without generation failures and correct
refusal remained exactly `1.000`.

## Claim-level faithfulness audit

The claim audit explains disagreements between the original binary, full-answer
faithfulness score and explicit support at the factual-claim level. It reuses the
saved D1/D2/D3 answers and their saved Top-5 chunk IDs; it does not generate new
answers, rerun similarity search, or change the RAG pipeline.

Run it with:

```bash
uv run python evaluation/run_claim_faithfulness_audit.py
```

The audit fetches the exact indexed text for those already-saved chunk IDs and writes:

```text
evaluation/results/claim_faithfulness_audit/
├── results.json  # machine-readable per-claim support, relevance, and categories
└── report.md     # model summary plus expandable question-by-question evidence
```

Answers with no factual claims are reported as such instead of receiving an artificial
zero. A disagreement is flagged when the absolute difference between the original
score and claim-level score is at least `0.25`. Grounded information that does not
answer the question is classified as `irrelevant but grounded information`; it remains
supported for faithfulness and is tracked as a relevance problem. The same read-only
audit is available in the Streamlit RAG Lab.

## Phase boundary

Phase 10 implements final cross-version evaluation. The Streamlit application in `app.py` provides both the simple user interface and a read-only dashboard over these measured results. RAG developer/debug mode, deployment, and all later phases remain intentionally unimplemented.
