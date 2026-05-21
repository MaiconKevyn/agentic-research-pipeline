# Research Agent with Continuous Evaluation

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135.3-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1.6-purple.svg)](https://github.com/langchain-ai/langgraph)
[![React](https://img.shields.io/badge/React-19.2.5-61DAFB.svg)](https://react.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg)](https://www.postgresql.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991.svg)](https://openai.com/)

> An agentic research pipeline built for project-scoped question answering. It combines LangGraph orchestration, PostgreSQL/pgvector retrieval, OpenAI-powered synthesis, structured outputs, reranking, and evaluation in a single end-to-end system.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
- [Data Ingestion](#data-ingestion)
- [Evaluation](#evaluation)
- [System Design](#system-design)
- [Guardrails](#guardrails)
- [Contributing](#contributing)
- [License](#license)

## Overview
**Research Agent with Continuous Evaluation** is a project-focused research assistant designed to answer questions about its indexed domain: internal documents, RAG systems, LangGraph, LangChain, FastAPI, pgvector, evaluation, and the project architecture itself.

The system uses a controlled multi-step workflow rather than a single prompt. It retrieves evidence from an internal PDF corpus and from web search when appropriate, reranks sources globally, generates a structured answer, and returns traceable evidence and heuristic evaluation metrics.

### Key Capabilities
- **Agentic Workflow**: Explicit multi-step orchestration with LangGraph
- **Hybrid Retrieval**: Dense pgvector search plus PostgreSQL full-text search with web escalation when evidence is weak
- **Structured Outputs**: Pydantic-validated contracts across critical steps
- **Project Scope Guardrails**: Rejects out-of-domain questions before retrieval
- **Evidence Attribution**: Responses include structured sources and execution trace
- **Continuous Evaluation**: Heuristic quality checks on the final answer

## Features
- **FastAPI Backend**: `POST /research` and `GET /health`
- **React Frontend**: Professional SPA built with Vite and TypeScript
- **PDF Ingestion Pipeline**: Structural chunking and embedding generation for real PDFs
- **PostgreSQL + pgvector**: Persistent internal knowledge base with semantic search
- **OpenAI Integration**: Uses `gpt-4o-mini` for synthesis and OpenAI embeddings for retrieval
- **Global Reranking**: Reorders vector and web evidence in a shared candidate pool
- **Execution Trace**: Surfaces planning, retrieval, reranking, and synthesis steps
- **Scope Guardrail**: Declines unrelated questions instead of answering everything from the web

## Architecture
The current architecture is implemented as a deterministic, stateful workflow:

```text
User
  |
  v
React + Vite frontend
  |
  v
FastAPI API
  |
  v
LangGraph workflow
  1. Classify question
     - research
     - operational
     - off_topic
  2. Plan retrieval
  3. Collect evidence
     - vector_search  -> dense + lexical hybrid retrieval
     - web_search     -> corrective OpenAI web search when evidence is weak
     - sql_query      -> PostgreSQL corpus stats
  4. Grade retrieval quality
  5. Deduplicate + global rerank
  6. Structured answer synthesis or insufficient-evidence abstention
  7. Heuristic evaluation
  |
  v
Structured API response
  - answer
  - sources
  - evaluation
  - execution_trace
```

The repository contains image files in `docs/`, but they are not currently the source of truth for this project's architecture. The workflow above reflects the implementation in code.

## Technology Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend API** | FastAPI 0.135.3 | API endpoints and request handling |
| **Workflow Engine** | LangGraph 1.1.6 | Multi-step agent orchestration |
| **Validation** | Pydantic 2.12.5 | Structured contracts and schema enforcement |
| **Database** | PostgreSQL + pgvector | Internal corpus storage and vector search |
| **Language Model** | OpenAI GPT-4o Mini | Answer synthesis and web search |
| **Embeddings** | OpenAI text-embedding-3-small | Semantic retrieval and reranking |
| **Frontend** | React 19 + Vite + TypeScript | Interactive web UI |
| **PDF Processing** | LangChain community loaders + PyMuPDF4LLM | PDF extraction and chunking |
| **Containerization** | Docker Compose | Local PostgreSQL/pgvector environment |
| **Testing** | Pytest + HTTPX | Backend and pipeline validation |

## Project Structure
```bash
agentic-research-pipeline/
├── agent/                             # LangGraph workflow and tool orchestration
│   ├── graph.py                       # Graph definition and runner
│   ├── nodes.py                       # Classification, retrieval, synthesis, evaluation
│   ├── state.py                       # Shared workflow state
│   └── tools/                         # vector_search, web_search, sql_query
├── backend/
│   ├── app/
│   │   ├── api/                       # FastAPI routes
│   │   ├── core/                      # Config and logging
│   │   ├── db/                        # DB connection helpers
│   │   ├── schemas/                   # Pydantic schemas
│   │   └── services/                  # LLM, embeddings, reranking, ingestion, repository
│   └── sql/
│       └── init_pgvector.sql          # pgvector schema bootstrap
├── data/
│   ├── corpus/                        # Initial seed corpus
│   └── raw/                           # Real PDFs for ingestion
├── evaluation/
│   ├── datasets/                      # Legacy sample prompts and docs
│   ├── golden/                        # JSONL golden evaluation cases
│   ├── reports/                       # Generated evaluation reports
│   ├── run_eval.py                    # CI-ready evaluation CLI
│   ├── metrics.py                     # Evaluation metrics
│   └── runner.py                      # Evaluation runner
├── frontend/
│   ├── src/                           # React UI
│   ├── index.html                     # Frontend entry HTML
│   └── package.json                   # Frontend dependencies and scripts
├── scripts/
│   ├── bootstrap_db.sh                # Starts PostgreSQL and seeds initial data
│   ├── ingest_raw_pdfs.py             # Ingests PDFs from data/raw
│   └── seed_documents.py              # Seeds the initial JSON corpus
├── tests/                             # Health, research, rerank, LLM, and ingestion tests
├── docker-compose.yml                 # Local PostgreSQL + pgvector service
├── project.md                         # Project scope and implementation status
├── requirements.txt                   # Backend dependencies
└── README.md                          # This file
```

## Getting Started
### Prerequisites
- **Python**: 3.12 or higher
- **Node.js**: 20+ recommended for the frontend toolchain
- **Docker**: required for local PostgreSQL + pgvector
- **OpenAI API Key**: required for synthesis, embeddings, and web search

### Installation
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/agentic-research-pipeline.git
cd agentic-research-pipeline
```

2. **Set up the Python environment**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. **Install frontend dependencies**
```bash
cd frontend
npm install
cp .env.example .env
cd ..
```

### Configuration
Create a `.env` file in the project root:
```env
APP_NAME=Research Agent
APP_ENV=development
API_HOST=127.0.0.1
API_PORT=8000

OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=

EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
REQUEST_TIMEOUT_SECONDS=60
DEFAULT_TOP_K=5

FRONTEND_ORIGINS=http://127.0.0.1:5173,http://localhost:5173

DATABASE_URL=
DATABASE_HOST=127.0.0.1
DATABASE_PORT=55432
DATABASE_NAME=research_agent
DATABASE_USER=postgres
DATABASE_PASSWORD=postgres
```

You can also start from:
```bash
cp .env.example .env
```

### Running the Application
1. **Start PostgreSQL + pgvector and seed the initial JSON corpus**
```bash
./scripts/bootstrap_db.sh
```

2. **Replace the initial seed with the real PDF corpus**
```bash
./.venv/bin/python scripts/ingest_raw_pdfs.py
```

3. **Run the FastAPI backend**
```bash
./.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

4. **Run the React frontend**
```bash
cd frontend
npm run dev
```

The frontend will be available at [http://127.0.0.1:5173](http://127.0.0.1:5173) and the API at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Usage
### Example Queries
**1. Ask about the internal corpus**
```text
What are the main challenges of RAG according to the internal corpus?
```

**2. Compare frameworks**
```text
What is the difference between LangChain and LangGraph in the indexed documents?
```

**3. Ask an operational corpus question**
```text
How many source documents and indexed chunks are stored in the project corpus?
```

**4. Ask about the project architecture**
```text
How does PostgreSQL + pgvector fit into the workflow of this project?
```

### API Example
```bash
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main challenges of RAG according to the internal corpus?",
    "top_k": 4
  }'
```

### Response Shape
Each response includes:
- **`run_id`**: unique identifier for the research run
- **`corpus_version_id`**: corpus snapshot used by the run
- **`corpus_stats`**: source document and indexed chunk counts
- **`answer`**: final synthesized answer
- **`sources`**: structured evidence used by the system
- **`evaluation`**: heuristic metrics for the answer
- **`execution_trace`**: step-by-step trace of the agent workflow

For out-of-scope questions, the system returns a structured refusal instead of retrieving unrelated web results.

## Data Ingestion
The ingestion pipeline performs the following steps:
1. **Load PDFs** from `data/raw`
2. **Extract text** using LangChain-compatible PDF tooling
3. **Normalize sections** and split content into structural chunks
4. **Add contextual headers** with document, section, and page location before embedding
5. **Generate embeddings** from contextualized text while preserving the raw chunk text
6. **Store source documents, pages, chunks, and embeddings** in PostgreSQL + pgvector

The ingestion service now produces deterministic chunk IDs, checksum-based source document identities, parent-child chunk metadata, and per-document JSON reports under `ingestion_reports/` by default. Reports include pages processed, chunks created, skipped chunks, extraction warnings, token distribution, duplicate percentage, and raw-vs-contextual embedding text size.

### Indexed Corpus
The current internal corpus is focused on:
- Python
- RAG
- Agentic workflows
- LangGraph / LangChain
- Evaluation

The corpus is built from the PDFs in `data/raw`. The exact source document and indexed chunk counts may change as documents are added, removed, or re-ingested. See [project.md](project.md) for the latest recorded project status.

## Evaluation
The project includes a golden-set evaluation harness intended to grow over time.

Current runtime metrics include:
- `groundedness`
- `answer_completeness`
- `evidence_sufficiency`
- `schema_validity`
- `scope_compliance` for out-of-scope refusals

The golden harness stores JSONL cases in `evaluation/golden/` with required sources, expected facts, forbidden claims, rubrics, difficulty, and query category. It writes JSONL reports to `evaluation/reports/` and checks the initial product thresholds for:
- `schema_validity`
- `scope_compliance`
- `citation_precision`
- `groundedness`
- `answer_relevance`
- `retrieval_recall_at_10`
- `retrieval_mrr`
- `retrieval_ndcg_at_10`
- `abstention_accuracy`

Current tests cover:
- health endpoint
- research endpoint
- PDF ingestion
- reranking
- structured output validation
- scope guardrail behavior
- run and corpus provenance
- golden evaluation scoring

Run the test suite with:
```bash
./.venv/bin/python -m pytest -q
```

Run the CI-safe evaluation harness with:
```bash
./.venv/bin/python -m evaluation.run_eval --mode mock --fail-on-threshold
```

Run the live graph over the golden set with:
```bash
./.venv/bin/python -m evaluation.run_eval --mode live --output evaluation/reports/latest.jsonl
```

## System Design
### Workflow
The agent follows a fixed high-level flow:
1. **Classify** the question
2. **Plan** the retrieval strategy
3. **Collect** evidence from hybrid internal retrieval, corrective web search, or SQL
4. **Grade** retrieval quality as sufficient, partial, weak, or irrelevant
5. **Rerank** evidence globally
6. **Synthesize** an answer with structured output or abstain on weak evidence
7. **Evaluate** the response before returning it

If the question is outside the project domain, the workflow exits early with a scope refusal instead of running retrieval.

### Retrieval Strategy
The system uses:
- **Hybrid internal search** that fuses dense pgvector results and PostgreSQL full-text results with Reciprocal Rank Fusion
- **Web search** as a corrective step when internal evidence is partial or weak
- **SQL query** for simple operational questions about the indexed corpus

The storage model now separates source documents, document pages, document chunks, corpus versions, research runs, run sources, and claim-evidence links while preserving the original `research_documents` compatibility table.

### Structured Outputs
Critical workflow steps use explicit Pydantic schemas, including:
- `QuestionClassification`
- `ResearchPlan`
- `EvidenceCollection`
- `SynthesisOutput`
- `EvaluationResult`

This keeps the pipeline predictable and easier to validate.

## Guardrails
The project already includes a scope guardrail:
- it answers only questions about the project and its indexed domain;
- it rejects unrelated questions before retrieval;
- it returns a structured refusal instead of hallucinating an answer from generic web results.

Example of an out-of-scope question:
```text
What is the most beautiful animal?
```

The system will refuse that query because it is unrelated to the project domain.

## Contributing
Contributions are welcome if they improve:
- retrieval quality
- evaluation rigor
- observability
- frontend usability
- pipeline reliability

Recommended workflow:
1. Fork the repository
2. Create a feature branch
3. Run tests locally
4. Open a pull request with a focused change set

## License
This project is licensed under the **MIT License**.

From the [LICENSE](LICENSE) file:

> Copyright (c) 2026 Maicon Kevyn
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions.
