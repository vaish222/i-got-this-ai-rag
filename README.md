# I Got This — What's Next?

A privacy-conscious family knowledge assistant that connects schedules, deadlines,
invitations, activities, meals, learning commitments, volunteer work, and household
tasks in one conversational interface.

## What you can ask

Use everyday questions such as:

- “What’s coming up this week?”
- “Plan my week.”
- “What should I prepare for this weekend?”
- “Which invitations still need an RSVP?”
- “What is the meal plan for Sunday?”
- “Is there any other volunteer work?”

The assistant provides grounded answers from the configured knowledge base, shows
the supporting documents, keeps dates and deadlines prominent, and asks for
clarification when a question is too broad.

## Application experience

The Streamlit application includes:

- a personalized Ask tab with short-term conversation memory;
- suggested questions that remain available throughout the conversation;
- plain-language answers without internal document identifiers;
- one compact card per day for schedules, with pastel category and status chips;
- source attribution for supported answers;
- safe clarification and insufficient-information responses; and
- a read-only Experiments tab with saved quality and latency measurements.

Conversation history is stored only in the active Streamlit session. Starting a new
conversation or ending the session clears it.

## Knowledge domains

The included synthetic sample knowledge base covers:

- school;
- children’s activities;
- household tasks and meals;
- personal learning;
- volunteer commitments;
- social invitations and gifts; and
- family schedules and important dates.

The sample data uses fictional anonymous roles and a fixed reference date so results
remain reproducible. Dataset conventions are documented in
[data/README.md](data/README.md).

## Requirements

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- A Pinecone account and API key

Ollama runs embedding and answer-generation models locally. Pinecone is hosted and
receives document chunks, metadata, generated vectors, and retrieval queries.

## Setup

Install the project and create your environment file:

```bash
uv sync
cp .env.example .env
```

Download the configured local models:

```bash
ollama pull embeddinggemma
ollama pull gemma3:1b
```

Add your Pinecone API key to `.env` and review the remaining settings:

```dotenv
APP_USER_NAME=Vaishali

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=embeddinggemma
OLLAMA_CHAT_MODEL=gemma3:1b

PINECONE_API_KEY=your-key
PINECONE_INDEX_NAME=i-got-this-phase-1
PINECONE_NAMESPACE=baseline

RAG_DATA_DIR=data/sample
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=75
RAG_TOP_K=5
RAG_REFERENCE_DATE=2026-08-20
RAG_TIMEZONE=America/Los_Angeles
```

Keep Ollama running before indexing data or starting the application. On macOS,
opening the Ollama application starts its local service; `ollama serve` can also be
run in a separate terminal.

## Index the knowledge base

Create or reuse the configured Pinecone namespace:

```bash
uv run python evaluation/run_phase1.py
```

When local source documents have changed, explicitly replace only the configured
namespace:

```bash
uv run python evaluation/run_phase1.py --rebuild-namespace
```

This operation does not delete the Pinecone index or unrelated namespaces.

The current sample corpus produces 28 chunks, while an older `baseline` namespace
may contain only 20 vectors. Rebuild the namespace after pulling the latest data so
new activities, meals, invitations, trips, assignments, and volunteer commitments
can be retrieved.

## Run the application

From the repository root:

```bash
uv run streamlit run app.py
```

Open the URL printed by Streamlit. The RAG connection is created after the first
question and reused for later questions in the same application process.

## Privacy notes

The repository ignores `.env` and `data/private/`. Keep secrets in `.env`, never in
source files.

The default `data/sample` corpus is synthetic. Before configuring `RAG_DATA_DIR` to
point at personal documents, confirm that sending their chunks and metadata to your
Pinecone project meets your privacy requirements. Answer generation remains local
through Ollama, but retrieval uses the hosted Pinecone service.

## Current evaluation

The latest saved application measurement reports:

| Metric | Result |
|---|---:|
| Recall@5 | 0.900 |
| Strict faithfulness | 0.200 |
| Correct refusal rate | 1.000 |
| Average latency | 1.536 s |
| P95 latency | 4.252 s |
| Corrected-behavior checks | 7/9 |

Weekly planning and the exact Sunday meal-plan checks pass. Two measured gaps remain:
pending RSVP aggregation can include a duplicate commitment, and the standalone
“next volunteer work” question can return a safe refusal instead of a sourced answer.

These numbers were measured against the active 20-vector `baseline` namespace. Rebuild
that namespace before treating the results as a measurement of all 28 current sample
chunks.

Regenerate the application measurement with:

```bash
uv run python evaluation/run_current_app_evaluation.py
```

Detailed metric definitions and experiment artifacts are documented in
[evaluation/README.md](evaluation/README.md).

## Tests

Run the offline test suite with:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

## Repository overview

```text
app.py                  Streamlit chat and experiment dashboard
data/                   Synthetic sample data and ignored private-data directory
evaluation/             Evaluation datasets, runners, and generated results
notebooks/              Optional interactive interfaces
config/                 Saved experiment and workflow configuration
src/i_got_this_rag/     Application, retrieval, conversation, and evaluation code
tests/                  Offline unit and Streamlit regression tests
```
