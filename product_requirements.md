# Product Requirements Document

# I Got This — What’s Next?
### *A privacy-first, RAG-powered personal & family command center*

**Version:** 2.1  
**Primary Language:** Python 3.12  
**RAG Framework:** LangChain  
**Agentic Orchestration:** LangGraph — advanced phase  
**Vector Database:** Pinecone  
**Initial Embedding Model:** `nomic-embed-text` via Ollama  
**UI:** Streamlit  
**Initial Knowledge Base:** 20 documents  
**Evaluation Dataset:** 15 questions

---

# 1. Project Pitch

School schedules. Kids' activities. Work. Household tasks. Volunteer commitments. Personal learning. Birthdays. Invitations. Social events. RSVPs. Gifts. Community commitments. Things we've promised other people.

There is always something to remember—and it's usually scattered across different emails, documents, schedules, notes, and messages.

**I Got This — What’s Next?** brings that information together so instead of trying to remember everything, you can simply ask:

> **“What’s next?”**

Or, perhaps even more importantly:

> **“What am I forgetting?”**

The system acts as a personal knowledge and family command center—retrieving relevant information, organizing what matters, connecting events with responsibilities, and providing grounded answers with sources.

The goal is to move from:

> **“I have too much to remember.”**

to:

> **“I got this.”**

---

# 2. Project Overview

**I Got This — What’s Next?** is a privacy-first personal RAG assistant designed to reduce the mental effort required to manage everyday personal and family life.

Important information may be scattered across:

- school newsletters
- school calendars
- kids' activity schedules
- household documents
- family schedules
- course materials
- course assignments
- volunteer notes
- birthday invitations
- social events
- RSVP deadlines
- gift reminders
- community obligations
- personal notes

Instead of remembering where information was mentioned or repeatedly searching through different sources, users can ask natural-language questions such as:

- What's coming up this week?
- What do the kids have this weekend?
- When is picture day?
- What do I need to prepare for the field trip?
- What assignments do I have coming up?
- What volunteer commitments are due?
- Which invitations still need an RSVP?
- Which birthdays do I still need gifts for?
- What should I prepare for this weekend?
- What's next?
- What am I forgetting?

The application retrieves relevant information from the user's personal knowledge base and generates answers grounded in those sources.

The project intentionally evolves through:

**Basic RAG → Advanced RAG → Agentic RAG → Personal AI Command Center**

---

# 3. Problem Statement

Managing personal and family life requires continuously tracking information for multiple people across multiple systems.

A normal week may involve:

- work
- school
- homework
- extracurricular activities
- appointments
- household responsibilities
- meal planning
- personal learning
- certification/course deadlines
- volunteer responsibilities
- birthdays
- invitations
- social events
- RSVP deadlines
- gifts
- hosting commitments
- community activities
- family events

The information usually exists.

The problem is that it is **fragmented**.

The real questions become:

> Where did I see that?

> Was that this Friday or next Friday?

> Did we RSVP?

> Did I buy the gift?

> What do we need to bring?

> Didn't I promise someone we'd attend?

> What assignments are due?

> What's happening this weekend?

> What am I forgetting?

> What's next?

Traditional keyword search requires knowing both where to search and what terminology was used in the original source.

General-purpose LLMs can produce natural responses but do not inherently know a family's private information and may hallucinate missing details.

**I Got This — What’s Next?** uses Retrieval-Augmented Generation to retrieve relevant personal information first and asks an LLM to answer using that evidence.

---

# 4. Product Vision

Transform scattered information:

```text
School emails ──────────────┐
School schedules ───────────┤
Kids' activities ───────────┤
Course material ────────────┤
Course deadlines ───────────┤
Volunteer commitments ──────┤
Household information ──────┤
Birthday invitations ───────┤
Social commitments ─────────┤
Family schedules ───────────┘
                            │
                            ▼
                   I GOT THIS
                   WHAT'S NEXT?
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
          REMEMBER       ORGANIZE        RETRIEVE
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                     CONNECT CONTEXT
                            │
                            ▼
                    WHAT MATTERS NEXT?
                            │
                            ▼
                     GROUNDED ANSWER
```

The product should eventually answer four core questions exceptionally well:

### What's happening?

Retrieve relevant events, activities, deadlines, and commitments.

### What's coming next?

Organize upcoming information across different areas of life.

### What do I need to do about it?

Connect events with associated actions.

### What am I forgetting?

Identify incomplete or upcoming obligations connected to known events.

These four questions form the product's long-term North Star.

---

# 5. One-Line Use Case

> **I Got This — What’s Next? helps busy families turn scattered school, activity, household, social, volunteering, and learning information into grounded answers about what matters, what needs attention, and what's coming next.**

---

# 6. Technical Learning Goal

This project has a second goal beyond building a useful product:

> **Understand what happens between a user's question and the LLM's answer, and how every RAG architectural decision affects retrieval and generation quality.**

The project should therefore be built experimentally rather than implementing the final architecture immediately.

The learning progression is:

**Documents → Cleaning → Metadata → Chunking → Embeddings → Pinecone → Dense Retrieval → Evaluation → Sparse Retrieval → Hybrid Retrieval → Reranking → Metadata Filtering → Query Transformation → Context Engineering → Grounding → LangGraph**

---

# 7. Success Targets

Initial targets:

- **≥90% grounded faithfulness**
- **p95 response latency under 5 seconds**
- Correct supporting document retrieved for at least **80% of answerable questions within Top-5**
- Correct refusal for intentionally unanswerable questions

These are evaluation targets, not assumed results.

---

# 8. Target User

## Primary Persona

A busy parent balancing:

- full-time employment
- children at different school stages
- extracurricular activities
- household responsibilities
- meal planning
- volunteer commitments
- personal learning
- professional development
- social responsibilities
- family obligations

### Core Need

> **Don't make me remember where the information lives. Help me find what matters when I need it.**

---

# 9. Initial Scope

V1 intentionally contains only **20 documents**.

This creates a controlled RAG environment where retrieval behavior can be manually inspected.

| Domain | Documents |
|---|---:|
| School | 4 |
| Kids' activities/classes | 3 |
| Personal learning/course | 3 |
| Volunteer work | 2 |
| Household | 3 |
| Social responsibilities | 3 |
| Family planning | 2 |
| **Total** | **20** |

The architecture must allow the corpus to grow without redesigning the core pipeline.

---

# 10. Knowledge Domains

## School

- Newsletters
- Calendars
- Field trips
- Parent notices
- Forms
- School events
- Important deadlines

## Kids' Activities

- Sports
- Music
- Dance
- Weekend classes
- Activity schedules
- Equipment requirements
- Locations

## Personal Learning

- Course schedules
- Course notes
- Assignments
- Certification requirements
- Study plans
- Deadlines

## Volunteer

- Mentoring responsibilities
- Meetings
- Content commitments
- Projects
- Deliverables

## Household

- Meal plans
- Home tasks
- Maintenance
- Shopping/planning information
- Household reference documents

## Social Responsibilities

- Birthday invitations
- Dinner invitations
- Parties
- Community events
- Housewarmings
- Graduation celebrations
- RSVP deadlines
- Gift reminders
- Hosting commitments
- Potlucks
- Items to bring
- Promises made to others
- Follow-up obligations

## Family Planning

- Family schedules
- Important dates
- Appointments
- Shared commitments
- Family events

---

# 11. Events vs. Obligations

A key product concept is:

> **Event ≠ Obligation**

An event may create several actions.

```text
Birthday
   ├── RSVP
   ├── Buy gift
   └── Attend

Potluck
   ├── RSVP
   └── Bring dessert

School Trip
   ├── Complete permission form
   ├── Prepare supplies
   └── Attend

Course Deadline
   ├── Complete module
   └── Submit assignment
```

This distinction becomes particularly important for:

> **What am I forgetting?**

The long-term system should retrieve both the event and its associated unfinished responsibilities.

---

# 12. Privacy Strategy

Privacy is a core product requirement.

Real personal documents may be used locally but must never be committed to the public GitHub repository.

### Public synthetic corpus

```text
data/sample/
```

### Private corpus

```text
data/private/
```

`.gitignore`:

```text
data/private/*
.env
evaluation/private/*
```

Therefore:

> **Public Demo Corpus ≠ Private Personal Knowledge Base**

No real family names, children's information, private schedules, addresses, or sensitive documents should be committed to the repository.

---

# 13. Initial Corpus Structure

```text
data/sample/

├── school/
│   ├── middle_school_newsletter.md
│   ├── middle_school_calendar.md
│   ├── elementary_newsletter.md
│   └── school_events.md
│
├── activities/
│   ├── swimming_schedule.md
│   ├── music_schedule.md
│   └── weekend_classes.md
│
├── learning/
│   ├── course_schedule.md
│   ├── week_2_assignment.md
│   └── course_requirements.md
│
├── volunteer/
│   ├── mentor_program.md
│   └── content_commitments.md
│
├── household/
│   ├── meal_plan.md
│   ├── home_tasks.md
│   └── household_information.md
│
├── social/
│   ├── invitations.md
│   ├── birthdays_and_gifts.md
│   └── social_commitments.md
│
└── family/
    ├── family_schedule.md
    └── important_dates.md
```

The synthetic corpus should intentionally contain:

- overlapping dates
- similar event names
- multiple family members
- deadlines
- recurring activities
- incomplete obligations
- related information distributed across documents
- similar concepts expressed using different terminology

This prevents retrieval from becoming trivial.

---

# 14. Supported Document Types

### V1

- Markdown
- TXT
- PDF

### Future

- Email
- Calendar events
- Google Docs
- Spreadsheets
- Web pages
- Messages
- Images/OCR

These integrations are outside Week 2 scope.

---

# 15. Metadata Schema

Each document or chunk should retain metadata.

```text
document_id
document_title

domain
document_type

person
related_person

event_type
event_date
event_time

status
rsvp_status
gift_status

action_required
action_due_date

source
tags
updated_at
```

Example:

```json
{
  "document_id": "social_001",
  "document_title": "August Invitations",
  "domain": "social",
  "document_type": "invitation",
  "event_type": "birthday",
  "related_person": "friend_child_01",
  "event_date": "2026-08-22",
  "event_time": "15:00",
  "rsvp_status": "completed",
  "gift_status": "needed",
  "action_required": "Purchase birthday gift",
  "tags": [
    "birthday",
    "social",
    "gift"
  ]
}
```

Public sample data should use anonymous identifiers instead of real names.

---

# 16. Architectural Principle — Configuration Over Hard-Coding

Because this project is intended to teach RAG through controlled experimentation, important RAG parameters must be configurable.

Do not hard-code:

- embedding model
- embedding dimension
- chunk size
- chunk overlap
- Pinecone index
- Pinecone namespace
- retrieval strategy
- candidate count
- Top-K
- reranker
- LLM model
- generation configuration

This enables the core learning loop:

```text
CHANGE ONE VARIABLE
        ↓
RUN SAME QUESTIONS
        ↓
MEASURE
        ↓
COMPARE
        ↓
UNDERSTAND WHY
```

---

# 17. Configuration Strategy

Secrets and experiment configuration should remain separate.

## Environment Variables

Use `.env` for secrets and deployment-specific values.

```text
PINECONE_API_KEY=
PINECONE_INDEX_NAME=igotthis-nomic
PINECONE_NAMESPACE=baseline
```

`.env` must never be committed to Git.

## Typed RAG Configuration

Conceptually:

```python
class RAGConfig:
    embedding_model: str = "nomic-embed-text"

    chunk_size: int = 500
    chunk_overlap: int = 75

    pinecone_index: str = "igotthis-nomic"
    pinecone_namespace: str = "baseline"

    retrieval_strategy: str = "dense"

    candidate_k: int = 5
    top_k: int = 5

    reranker_enabled: bool = False

    llm_model: str = "..."
```

Pydantic settings, dataclasses, or another lightweight typed configuration approach may be used.

---

# 18. Configuration Validation

Before ingestion or retrieval:

```text
Embedding Model
      ↓
Embedding Dimension
      ↓
Pinecone Index Dimension
      ↓
Compatible?
   /       \
 YES        NO
  │          │
Continue   Fail Clearly
```

The application must not silently insert incompatible vectors.

Example:

> **Embedding model configuration is incompatible with the selected Pinecone index.**

The active configuration should also be validated for:

- positive chunk size
- valid chunk overlap
- Top-K greater than zero
- configured Pinecone index
- valid retrieval strategy
- available embedding model
- required API credentials

---

# 19. Ingestion Pipeline

```text
Documents
     ↓
LangChain Document Loaders
     ↓
Cleaning
     ↓
Metadata
     ↓
Configured Chunking
     ↓
Configured Embeddings
     ↓
Configured Pinecone Index
```

Cleaning should:

- normalize whitespace
- remove irrelevant boilerplate
- preserve headings
- preserve dates
- preserve lists
- preserve event names
- preserve action items
- preserve deadlines
- preserve identifiers
- retain source information

Dates and deadlines are especially important because temporal questions are central to this application.

---

# 20. Baseline Chunking

V1:

```text
RecursiveCharacterTextSplitter

chunk_size = 500
chunk_overlap = 75
```

These are baseline settings, not assumed optimal values.

Later compare:

```text
250
500
750
1000
```

while holding other variables constant.

---

# 21. Embedding Strategy

V1 will use a free local embedding model through Ollama:

```text
nomic-embed-text
```

Architecture:

```text
Documents
    ↓
Chunking
    ↓
nomic-embed-text
    ↓
Dense Vectors
    ↓
Pinecone
```

Later compare:

```text
all-minilm
      vs
nomic-embed-text
      vs
mxbai-embed-large
```

Only the embedding configuration should change during embedding experiments.

---

# 22. Pinecone Vector Database

Pinecone will store:

- embedding vectors
- chunk IDs
- document IDs
- source information
- domain metadata
- person metadata
- event metadata
- obligation metadata

### V1

```text
Dense Embeddings
       +
Metadata
       ↓
Pinecone
```

### Query Flow

```text
User Question
      ↓
Embedding Model
      ↓
Query Vector
      ↓
Pinecone Search
      ↓
Top 5 Chunks
```

---

# 23. Pinecone Index & Namespace Strategy

## Chunking Experiments

If the embedding model remains the same, use one compatible index with separate namespaces.

```text
igotthis-nomic

├── baseline
├── chunk-250
├── chunk-500
├── chunk-750
└── chunk-1000
```

This keeps experimental vectors isolated without creating unnecessary indexes.

## Embedding Experiments

Different embedding models may produce different vector dimensions.

Use separate indexes when required:

```text
igotthis-nomic
igotthis-minilm
igotthis-mxbai
```

Never mix incompatible vector dimensions in the same index.

---

# 24. Week 2 Baseline Architecture

```text
                    CONFIGURATION
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Embedding       Chunking       Pinecone
       Model          Settings        Index
          │              │              │
          └──────────────┼──────────────┘
                         ▼

                     OFFLINE

                   20 Documents
                         ↓
                LangChain Loaders
                         ↓
                      Clean
                         ↓
                      Chunk
                         ↓
                    Embeddings
                         ↓
                     Pinecone


                      ONLINE

                  User Question
                         ↓
                  Query Embedding
                         ↓
                Pinecone Retrieval
                         ↓
                   Top 5 Chunks
                         ↓
                    RAG Prompt
                         ↓
                        LLM
                         ↓
                 Grounded Answer
                         ↓
                    Citations
```

Do **not** introduce LangGraph into V1.

The purpose of V1 is to establish a measurable baseline before adding advanced techniques.

---

# 25. Generation Requirements

The generation model must:

- use retrieved context as the primary source
- avoid adding unsupported personal facts
- produce concise, readable answers
- identify relevant upcoming dates or obligations
- include source citations
- distinguish known information from missing information

Prompt principle:

> **Answer using only the supplied personal knowledge context. If the context does not contain enough information to answer reliably, say so explicitly.**

---

# 26. “I Don't Know” Requirement

The system must never invent personal information.

If sufficient information is unavailable:

> **I couldn't find that information in your knowledge base.**

Example:

**User:**  
What social events do we have next summer?

**I Got This:**  
I couldn't find information about next summer's social schedule in your knowledge base.

The evaluation dataset will include deliberately unanswerable questions to test this behavior.

---

# 27. Evaluation Dataset

Create **15 evaluation questions before optimizing the pipeline**.

| Category | Questions |
|---|---:|
| Exact lookup | 3 |
| Semantic | 3 |
| Date/deadline | 3 |
| Cross-document | 2 |
| Cross-domain | 2 |
| Unanswerable | 2 |
| **Total** | **15** |

---

# 28. Evaluation Questions

## Exact Lookup

1. What time does Saturday's activity start?
2. What should students bring for the field trip?
3. What do we need to bring to the neighborhood potluck?

## Semantic

4. What should we prepare for the beginning of school?
5. What responsibilities do I have for volunteer mentoring?
6. What do I need to complete for my course?

## Date / Deadline

7. When is elementary picture day?
8. When is my next course assignment due?
9. Which invitations still need an RSVP?

## Cross-Document

10. What important school events are coming up?
11. Which birthdays or social events do I need to prepare for?

## Cross-Domain

12. **What's next for everyone this week?**

13. **What do I need to prepare for this weekend across school, kids' activities, learning, volunteering, household, and social commitments?**

## Unanswerable

14. When is next year's graduation?
15. What social events do we have scheduled for next summer?

Questions 12 and 13 represent the application's signature RAG experience.

---

# 29. Future Signature Question — “What Am I Forgetting?”

Once the application supports better obligation reasoning, introduce:

> **What am I forgetting for this weekend?**

This is intentionally harder than:

> What's happening this weekend?

The system must connect an event to unfinished actions.

Example:

```text
Saturday Birthday Party
       │
       └── Gift not purchased

Sunday Potluck
       │
       └── Dessert required

Monday School Trip
       │
       └── Permission form pending

Sunday Course Deadline
       │
       └── Assignment incomplete
```

The desired answer should surface unfinished responsibilities rather than simply list events.

---

# 30. Evaluation Schema

Each evaluation question should have a structured record.

```json
{
  "question_id": "Q012",
  "question": "What's next for everyone this week?",
  "expected_answer": "...",
  "expected_sources": [
    "Elementary School Newsletter",
    "Family Schedule",
    "Course Schedule",
    "Social Commitments"
  ],
  "category": "cross_domain",
  "answerable": true
}
```

---

# 31. Evaluation Metrics

## Retrieval Metrics

Measure:

- Recall@5
- Precision@5
- expected-source rank
- retrieval latency

## Generation Metrics

Measure:

- faithfulness
- answer correctness
- answer relevance
- citation correctness

## Operational Metrics

Measure:

- retrieval latency
- reranking latency
- generation latency
- total response latency

Every incorrect answer should be diagnosed as:

```text
              WRONG ANSWER
                   │
             ┌─────┴─────┐
             ▼           ▼
        RETRIEVAL     GENERATION
          FAILURE       FAILURE
```

The first debugging question should always be:

> **Did the correct evidence reach the LLM?**

---

# 32. Experiment Configuration

Every experiment must explicitly record the configuration that produced the results.

Example:

```json
{
  "experiment_id": "E001",
  "experiment_name": "dense_nomic_chunk500",
  "vector_store": "pinecone",
  "pinecone_index": "igotthis-nomic",
  "pinecone_namespace": "chunk-500",
  "embedding_model": "nomic-embed-text",
  "chunk_size": 500,
  "chunk_overlap": 75,
  "retrieval_strategy": "dense",
  "candidate_k": 5,
  "top_k": 5,
  "reranker_enabled": false,
  "evaluation_dataset": "questions_v1"
}
```

---

# 33. Experiment Reproducibility

Every evaluation result must be tied to its configuration.

```text
evaluation/
│
├── questions.json
│
└── results/
    │
    ├── E001_dense_nomic_chunk500/
    │   ├── config.json
    │   └── results.json
    │
    ├── E002_dense_nomic_chunk250/
    │   ├── config.json
    │   └── results.json
    │
    └── E003_dense_nomic_chunk750/
        ├── config.json
        └── results.json
```

The system should always be able to answer:

> **Exactly what configuration produced this score?**

---

# 34. Technology Stack

| Layer | Technology |
|---|---|
| Programming language | **Python 3.12** |
| RAG framework | **LangChain** |
| Agent/workflow orchestration | **LangGraph — later** |
| Vector database | **Pinecone** |
| Initial embedding model | **nomic-embed-text via Ollama** |
| Embedding experiments | all-minilm / mxbai-embed-large |
| V1 retrieval | Dense vector retrieval |
| Advanced retrieval | Sparse / hybrid |
| Reranking | Cross-encoder or compatible reranker |
| LLM | Configurable through LangChain |
| Evaluation | LangSmith + custom Python |
| UI | Streamlit |
| Configuration | Pydantic / typed configuration |
| Testing | pytest |
| Package management | `uv` |
| Version control | Git |

---

# 35. Python Project Structure

```text
i-got-this-ai/
│
├── app.py
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
│
├── config/
│   ├── baseline.yaml
│   └── experiments/
│       ├── chunk_250.yaml
│       ├── chunk_500.yaml
│       ├── chunk_750.yaml
│       └── chunk_1000.yaml
│
├── data/
│   ├── sample/
│   │   ├── school/
│   │   ├── activities/
│   │   ├── learning/
│   │   ├── volunteer/
│   │   ├── household/
│   │   ├── social/
│   │   └── family/
│   │
│   └── private/
│
├── src/
│   ├── config/
│   │   └── settings.py
│   │
│   ├── ingestion/
│   │   ├── loaders.py
│   │   ├── cleaner.py
│   │   └── chunker.py
│   │
│   ├── embeddings/
│   │   └── embeddings.py
│   │
│   ├── vectorstore/
│   │   └── pinecone_store.py
│   │
│   ├── retrieval/
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   │
│   ├── rag/
│   │   ├── prompts.py
│   │   ├── generator.py
│   │   └── citations.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   │
│   └── evaluation/
│       ├── evaluator.py
│       ├── experiment.py
│       └── metrics.py
│
├── evaluation/
│   ├── questions.json
│   └── results/
│
└── tests/
```

Modules should be added progressively rather than over-engineering V1.

---

# 36. Development Roadmap

## Phase 0 — Dataset Creation

### Goal

Create the controlled RAG laboratory before implementing retrieval.

### Deliverables

- 20 synthetic documents
- seven knowledge domains
- meaningful metadata
- overlapping information
- 15 evaluation questions
- expected answers
- expected source documents
- two deliberately unanswerable questions

### Prompt

> Create a controlled synthetic dataset for a privacy-first personal and family RAG application called **I Got This — What's Next?** Generate 20 realistic documents covering school, children's activities, personal learning, volunteer work, household responsibilities, social responsibilities and obligations, and family planning. Include overlapping dates, deadlines, events, responsibilities, RSVPs, gifts, action items, and related information across documents so retrieval is non-trivial. Also create 15 evaluation questions with expected answers, expected sources, categories, and answerable flags. Do not implement RAG yet.

---

# 37. Phase 1 — Baseline Dense RAG

### Goal

Build the simplest complete RAG pipeline.

```text
Load
 ↓
Chunk
 ↓
Embed
 ↓
Pinecone
 ↓
Retrieve Top 5
 ↓
LLM
 ↓
Answer + Citations
```

### Requirements

The following must be configurable:

- embedding model
- chunk size
- overlap
- Pinecone index
- namespace
- Top-K
- LLM

### Prompt

> Build the V1 Python RAG pipeline for **I Got This — What's Next?** using LangChain and Pinecone. Load Markdown, TXT, and PDF documents, preserve metadata, chunk the documents, generate embeddings, store vectors and metadata in Pinecone, retrieve the top five relevant chunks, and generate answers strictly from retrieved context with source citations.
>
> Design the pipeline so embedding model, chunk size, chunk overlap, Pinecone index, Pinecone namespace, retrieval Top-K, and LLM model are configurable rather than hard-coded. Use environment variables for secrets and a typed Python configuration layer for RAG parameters.
>
> Validate that the configured embedding model is compatible with the selected Pinecone index before indexing or retrieval. Keep ingestion, embeddings, vector storage, retrieval, and generation modular.
>
> Do not add LangGraph, hybrid retrieval, reranking, query rewriting, or calendar integration yet.

---

# 38. Phase 2 — Baseline Evaluation

### Goal

Measure V1 before improving it.

For every question record:

- question
- expected source
- retrieved chunks
- similarity scores
- expected-source rank
- generated answer
- citations
- retrieval latency
- LLM latency
- total latency

### Prompt

> Build an evaluation runner for **I Got This — What's Next?** that executes all 15 evaluation questions against the current Pinecone-based RAG pipeline. Record the active configuration, retrieved document IDs, chunk IDs, similarity scores, expected-source ranking, generated answer, citations, retrieval latency, generation latency, and total latency. Calculate Recall@5 and save both configuration and results for future comparison.

---

# 39. Phase 3 — Chunking Experiments

### Goal

Understand how chunk size affects retrieval.

Compare:

```text
250
500
750
1000 tokens
```

Only change:

- chunk size
- overlap where appropriate
- Pinecone namespace

Keep constant:

- documents
- embedding model
- Pinecone index
- retrieval strategy
- Top-K
- LLM
- evaluation questions

### Prompt

> Create controlled chunking experiments for **I Got This — What's Next?** Compare chunk sizes of 250, 500, 750, and 1000 tokens using the same 20 documents, embedding model, Pinecone index, dense retrieval strategy, Top-K, LLM, and 15 evaluation questions. Use separate Pinecone namespaces for each configuration. Measure Recall@5, expected-source rank, latency, and failure patterns by question category.

---

# 40. Phase 4 — Embedding Experiments

### Goal

Understand embedding-model impact.

Compare:

```text
all-minilm
nomic-embed-text
mxbai-embed-large
```

Keep everything else constant where possible.

Use separate Pinecone indexes if embedding dimensions differ.

### Prompt

> Make the embedding model configurable and compare multiple local Ollama embedding models using the same documents, selected chunking configuration, Top-K, retrieval strategy, LLM, and evaluation dataset. Create or use Pinecone indexes with compatible dimensions for each model. Compare Recall@5, source ranking, retrieval latency, and failure cases.

---

# 41. Phase 5 — Dense vs. Sparse vs. Hybrid Retrieval

### Goal

Understand the difference between semantic and lexical retrieval.

```text
                    Query
                      │
             ┌────────┼────────┐
             ▼        ▼        ▼
           Dense    Sparse   Hybrid
```

Exact dates, names, RSVP terms, event titles, and activity names may benefit from lexical retrieval.

Semantic questions may favor dense retrieval.

### Prompt

> Add configurable dense, sparse, and hybrid retrieval strategies to **I Got This — What's Next?** while keeping the generation model unchanged. Run the same 15 evaluation questions against each retrieval strategy and compare Recall@5, expected-source ranking, latency, and failures by category. Pay special attention to exact dates/names versus semantic and cross-domain questions.

---

# 42. Phase 6 — Reranking

### Goal

Understand two-stage retrieval.

```text
Pinecone Retrieval
       ↓
     Top 20
       ↓
    Reranker
       ↓
      Top 5
       ↓
       LLM
```

Compare:

```text
Retrieve Top 5

vs.

Retrieve Top 20
      ↓
Rerank
      ↓
Top 5
```

### Prompt

> Add an optional reranking stage to the RAG pipeline. Retrieve the top 20 candidates from Pinecone, rerank them against the original user question, and pass only the top 5 to generation. Compare retrieval quality, faithfulness, and end-to-end latency against retrieval without reranking.

---

# 43. Phase 7 — Metadata-Aware Retrieval

### Goal

Use structured information to improve retrieval.

Potential filters:

```text
domain
person
document_type
event_type
event_date
status
rsvp_status
gift_status
```

Example:

> Which birthday invitations still need an RSVP?

could become:

```text
domain = social
event_type = birthday
rsvp_status = pending
```

### Prompt

> Add metadata-aware retrieval to **I Got This — What's Next?** Analyze questions for domain, person, document type, event type, dates, RSVP status, gift status, and other structured constraints. Translate appropriate constraints into Pinecone metadata filters while preserving semantic retrieval. Compare filtered versus unfiltered retrieval on the existing evaluation set.

---

# 44. Phase 8 — Query Transformation

### Goal

Understand how an LLM can improve the retrieval query itself.

Start with:

- query rewriting
- multi-query retrieval

Potential later techniques:

- HyDE
- query decomposition

Example:

```text
User:
"What do I need to take care of before Saturday?"

Rewritten:
"Saturday upcoming events preparation requirements
RSVP gifts forms items to bring deadlines"
```

### Prompt

> Add an LLM-based query rewriting component that converts conversational questions into retrieval-focused queries while preserving exact people, dates, event names, deadlines, RSVP status, gift requirements, and important terminology. Compare retrieval performance with and without rewriting and record cases where query rewriting reduces quality.

---

# 45. Phase 9 — LangGraph Agentic RAG

### Goal

Introduce conditional orchestration after the retrieval pipeline is understood.

```text
Question
   ↓
Analyze Intent
   ↓
Rewrite Query
   ↓
Build Metadata Filters
   ↓
Retrieve
   ↓
Rerank
   ↓
Grade Evidence
  /          \
Good          Weak
 │             │
 │         Retry Once
 ▼
Generate
   ↓
Check Grounding
  /          \
PASS          FAIL
 │             │
 ▼             ▼
Answer      Refuse
```

### Proposed Graph State

```text
original_query
search_query
intent
metadata_filters
retrieved_docs
reranked_docs
retrieval_attempts
evidence_sufficient
answer
citations
grounded
```

### Prompt

> Refactor the existing **I Got This — What's Next?** RAG pipeline into a LangGraph StateGraph without replacing the working retrieval modules. Create nodes for query analysis, query rewriting, metadata construction, retrieval, reranking, evidence grading, generation, and grounding verification. Use conditional edges so insufficient retrieval triggers at most one retry and unsupported answers return an explicit insufficient-information response.

---

# 46. Phase 10 — Final Evaluation

Compare major system versions.

| Version | Recall@5 | Faithfulness | Avg. Latency |
|---|---:|---:|---:|
| Baseline dense RAG | TBD | TBD | TBD |
| Best chunking | TBD | TBD | TBD |
| Best embedding | TBD | TBD | TBD |
| Hybrid retrieval | TBD | TBD | TBD |
| Hybrid + reranker | TBD | TBD | TBD |
| Metadata-aware | TBD | TBD | TBD |
| Query rewriting | TBD | TBD | TBD |
| LangGraph workflow | TBD | TBD | TBD |

Do not populate values before actually running experiments.

The important question is not simply:

> Which configuration scored highest?

It is:

> **Why did it perform better, what did it make worse, and what did the improvement cost?**

---

# 47. Experiment Dashboard

The Streamlit developer view may eventually include:

| Experiment | Embedding | Chunk | Retrieval | Rerank | Recall@5 | Faithfulness | Latency |
|---|---|---:|---|---|---:|---:|---:|
| E001 | nomic | 250 | Dense | No | TBD | TBD | TBD |
| E002 | nomic | 500 | Dense | No | TBD | TBD | TBD |
| E003 | nomic | 750 | Dense | No | TBD | TBD | TBD |
| E004 | MiniLM | Best | Dense | No | TBD | TBD | TBD |
| E005 | nomic | Best | Hybrid | No | TBD | TBD | TBD |
| E006 | nomic | Best | Hybrid | Yes | TBD | TBD | TBD |

Each experiment should answer:

- What changed?
- What stayed constant?
- What improved?
- What became worse?
- Why?
- What latency did it add?
- Was the improvement worth it?

---

# 48. Streamlit User Interface

V1 should stay simple.

```text
╭────────────────────────────────────────────╮
│                                            │
│             I GOT THIS                     │
│             What's next?                   │
│                                            │
├────────────────────────────────────────────┤
│                                            │
│ Ask about your family knowledge...         │
│                                            │
│ [ What should I prepare for this week? ]  │
│                                            │
│                   Ask                      │
│                                            │
├────────────────────────────────────────────┤
│ Answer                                     │
│                                            │
│ • Field trip form due Friday               │
│ • Birthday Saturday — gift still needed    │
│ • Course assignment due Sunday             │
│                                            │
│ Sources                                    │
│ • Elementary School Newsletter             │
│ • Social Commitments                       │
│ • Course Schedule                          │
╰────────────────────────────────────────────╯
```

---

# 49. RAG Developer / Debug Mode

A developer view should eventually expose:

- active configuration
- original question
- rewritten query
- metadata filters
- retrieval strategy
- Pinecone index
- Pinecone namespace
- Top-K
- retrieved chunks
- similarity scores
- reranked chunks
- reranking scores
- context sent to LLM
- generated answer
- citations
- evidence grade
- grounding result
- retrieval latency
- generation latency
- total latency

This makes the RAG system observable rather than a black box.

---

# 50. Week 2 Definition of Done

- [ ] 20 synthetic documents created
- [ ] Seven knowledge domains represented
- [ ] Social obligations explicitly represented
- [ ] Metadata added
- [ ] 15 evaluation questions created
- [ ] Expected answers and sources defined
- [ ] Python/LangChain ingestion works
- [ ] Documents are chunked
- [ ] Local `nomic-embed-text` embeddings work
- [ ] Pinecone index created
- [ ] Embedding/index compatibility validated
- [ ] Vectors and metadata upsert successfully
- [ ] Dense Top-5 retrieval works
- [ ] LLM answers from retrieved evidence
- [ ] Citations are displayed
- [ ] “I don't know” path works
- [ ] Baseline evaluation runs all 15 questions
- [ ] RAG parameters are configuration-driven
- [ ] Secrets are separate from experiment settings
- [ ] Experiment configuration is saved with results
- [ ] Simple Streamlit UI works

### Stretch Goal

- [ ] Run at least two chunking configurations without modifying application source code.

---

# 51. Non-Goals for Week 2

Do not build yet:

- Gmail integration
- Google Calendar integration
- automatic reminders
- automatic task creation
- calendar scheduling
- autonomous multi-agent workflows
- hundreds of documents
- GraphRAG
- production authentication
- mobile application
- production infrastructure

The Week 2 objective is:

> **Understand RAG deeply before expanding the product.**

---

# 52. Future Phase — Calendar Integration

Once document RAG works reliably:

```text
                 USER QUESTION
                       │
                       ▼
                LangGraph Router
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      Knowledge     Schedule     Combined
       Question      Question     Question
          │            │            │
         RAG        Calendar    RAG + Calendar
          │            │            │
          └────────────┼────────────┘
                       ▼
                      LLM
                       │
                       ▼
                    Answer
```

Examples:

> What does the school newsletter say about the field trip?

Use RAG.

> What do we have scheduled Saturday?

Use calendar information.

> What should we prepare before everything happening Saturday?

Use both.

---

# 53. Future Phase — “What’s Next?” Engine

Eventually the product should organize information temporally:

```text
TODAY
 ↓
TOMORROW
 ↓
THIS WEEK
 ↓
THIS WEEKEND
 ↓
COMING UP
```

Example:

```text
Saturday

Birthday party — 3 PM
→ Gift still needed

Sunday

Neighborhood potluck — 6 PM
→ Bring dessert

Course assignment
→ Due Sunday
```

This moves the product from document Q&A toward a true command center.

---

# 54. Future Phase — “What Am I Forgetting?” Engine

The advanced system should reason over:

```text
Known Event
     +
Required Action
     +
Action Status
     +
Deadline
     ↓
Outstanding Obligation
```

Examples:

- Birthday + gift needed → surface gift
- Invitation + RSVP pending → surface RSVP
- Potluck + dessert required → surface preparation
- School trip + form pending → surface form
- Course assignment + due date → surface coursework
- Hosting commitment + shopping required → surface shopping/preparation

This becomes one of the product's defining capabilities.

---

# 55. Future Phase — Personal Planning Agent

A later LangGraph system may combine:

```text
Personal Knowledge Base
          +
       Calendar
          +
      Deadlines
          +
   Outstanding Tasks
          ↓
       LangGraph
          ↓
      Planning Agent
```

Example:

> I need two hours to finish my assignment before Sunday. When can I do it?

The system could:

1. Retrieve the assignment requirements.
2. Retrieve the deadline.
3. Check scheduled commitments.
4. Find available time.
5. Recommend suitable blocks.

This evolves the project from:

**Personal RAG → Agentic RAG → Personal AI Command Center**

---

# 56. Long-Term User Experience

The eventual application should feel less like opening a chatbot and more like opening a personal command center.

```text
╭─────────────────────────────────────────────╮
│                                             │
│               I GOT THIS                    │
│               What's next?                  │
│                                             │
├─────────────────────────────────────────────┤
│ GOOD MORNING                                │
│                                             │
│ TODAY                                       │
│                                             │
│ 🏫 School                                   │
│ • Picture Day                               │
│                                             │
│ 🎻 Kids                                     │
│ • Activity — 4:30 PM                        │
│                                             │
│ 📚 Learning                                 │
│ • Complete Week 3 videos                    │
│                                             │
│ NEXT UP                                     │
│                                             │
│ Friday                                      │
│ • Field trip form due                       │
│                                             │
│ Saturday                                    │
│ • Birthday party — 3 PM                     │
│ • Gift still needed                         │
│                                             │
│ Sunday                                      │
│ • Neighborhood potluck — bring dessert      │
│ • Course assignment due                     │
│                                             │
├─────────────────────────────────────────────┤
│ Ask me anything...                          │
│                                             │
│ [ What am I forgetting this weekend? ]     │
╰─────────────────────────────────────────────╯
```

---

# 57. Portfolio Description

> **I Got This — What’s Next?** is a privacy-first personal knowledge and family command center built with Python, LangChain, Pinecone, local Ollama embeddings, and eventually LangGraph. It transforms scattered school, activity, household, volunteer, social, family, and learning information into a searchable RAG knowledge base that produces grounded answers with source citations.
>
> Rather than treating RAG as a black box, the project is developed experimentally. Chunking strategies, embedding models, dense and sparse retrieval, hybrid search, reranking, metadata filtering, query transformation, context construction, grounding, and LangGraph orchestration are evaluated against a controlled dataset while preserving reproducible experiment configurations.

---

# 58. Core Engineering Principle

A major technical goal of **I Got This — What’s Next?** is not simply to build a working RAG application.

It is to understand how each design choice affects the system.

Therefore:

> **Change one variable → run the same evaluation → measure the result → understand why.**

```text
                 CONFIGURE
                     ↓
                   INDEX
                     ↓
                  RETRIEVE
                     ↓
                  EVALUATE
                     ↓
                   MEASURE
                     ↓
                   COMPARE
                     ↓
                 UNDERSTAND
                     ↓
               NEXT EXPERIMENT
```

This turns the initial **20-document + 15-question** project into a controlled RAG laboratory.

---

# 59. Final Product North Star

**I Got This — What’s Next?** should eventually answer four questions exceptionally well:

> **What's happening?**

> **What's coming next?**

> **What do I need to do about it?**

> **What am I forgetting?**

If the application can reliably answer these questions using grounded personal knowledge without requiring the user to remember where the information originally came from, the product is delivering on its core promise:

> **Less remembering. Less searching. More “I got this.”**