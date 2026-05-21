# Research Agent with Continuous Evaluation

## Objective

Build a small but technically solid project to learn how to develop a research agent with:

- explicit orchestration;
- tool calling;
- structured validation;
- automated evaluation;
- an API and professional web interface;
- practical use of a relational database and vector search.

The goal of the first phase is not to assemble the biggest stack possible. The goal is to learn how to build an agent that is reliable, observable, and easy to evolve.

## Implementation Status

Completed:

- base folder structure for `backend`, `agent`, `frontend`, `evaluation`, and `tests`;
- centralized configuration with `.env`, `gpt-4o-mini`, OpenAI embeddings, and database settings;
- `POST /research` and `GET /health` endpoints;
- real orchestration with `LangGraph`;
- professional web frontend with `React + Vite + TypeScript`;
- local frontend/backend integration through `CORS`;
- selective use of `LangChain` for PDF ingestion and structural chunking;
- Pydantic contracts for classification, planning, evidence collection, synthesis, and evaluation;
- structured output for final synthesis with schema validation;
- answer synthesis through the OpenAI Responses API;
- embeddings through the OpenAI Embeddings API;
- real `web_search` through the OpenAI Responses API;
- `docker-compose` and PostgreSQL bootstrap with `pgvector`;
- working `PostgreSQL + pgvector` infrastructure, vector search, and SQL corpus counting;
- initial seed script for the internal corpus;
- local database running successfully;
- real ingestion of PDFs from `data/raw` into the vector database;
- source enrichment with metadata such as `file`, `section`, `page`, `rank`, and `distance`;
- global evidence reranking before `sources_kept`, considering candidates from `vector_search` and `web_search`;
- scope guardrail for out-of-domain questions, rejecting them before retrieval and synthesis;
- more detailed execution trace per tool;
- tests for `health`, `research`, PDF ingestion, reranking, and structured output;
- UI copy, project documentation, and user-facing prompts translated to English.
- repository README rewritten in a full project format with setup, usage, architecture, and license sections.
- README corrected to match the actual implementation and no longer rely on an unrelated architecture image.
- normalized source document, page, chunk, corpus version, research run, and run-source tables added while preserving the original `research_documents` path.
- each `/research` response now includes `run_id`, `corpus_version_id`, source document count, and chunk count.
- `sql_query` now reports source document count and chunk count separately for operational corpus questions.
- golden-set evaluation harness added with JSONL cases, score reports, thresholds, and a CI-safe CLI mode.
- PDF ingestion now uses deterministic checksum-based document identity, contextualized embedding text, parent-child chunk metadata, and per-document JSON ingestion reports.
- hybrid retrieval added with PostgreSQL full-text search, dense + lexical Reciprocal Rank Fusion, retrieval quality grading, corrective web escalation, and weak-evidence abstention.
- retrieval evaluation metrics now include `Recall@K`, `MRR`, and `nDCG`.

### Current Agent Status

The agent already works end to end for phase 1.

- it receives a question through FastAPI;
- it classifies the question and selects tools;
- it plans execution with controlled `top_k`;
- it queries hybrid internal search, corrective `web_search`, and `sql_query` when applicable;
- it grades retrieval quality before synthesis and abstains when support stays weak;
- it rejects out-of-scope questions before calling retrieval tools;
- it deduplicates evidence, applies global reranking, and keeps the most relevant sources;
- it synthesizes the answer with `gpt-4o-mini` using validated structured output;
- it returns an answer, structured sources, heuristic evaluation, and execution trace.

The critical outputs in the workflow already have explicit schemas:

- `QuestionClassification`
- `ResearchPlan`
- `EvidenceCollection`
- `SynthesisOutput`
- `EvaluationResult`

### Current Interface Status

- SPA built with `React + Vite + TypeScript`;
- layout starts directly with `composer + output`, without a top status strip;
- composer for the question and `top_k` control;
- rich display for answer, metrics, sources, and trace;
- responsive design for desktop and mobile;
- frontend ready to talk to the local FastAPI API.

### Current Tool Status

- `vector_search`: operational as dense + lexical hybrid retrieval over `OpenAI embeddings + PostgreSQL/pgvector + PostgreSQL full-text search`;
- `web_search`: operational through the OpenAI Responses API and used as corrective retrieval when internal evidence is weak or partial;
- `sql_query`: operational for simple corpus statistics.

### Current Internal Corpus Status

- corpus theme: `Python / RAG / Agentic / LangGraph / evaluation`;
- source: PDFs stored in `data/raw`;
- real ingestion implemented through `scripts/ingest_raw_pdfs.py`;
- current chunking strategy: structural chunking by page and section, contextual headers for embeddings, deterministic chunk IDs, and parent-child chunk metadata;
- ingestion reports: generated under `ingestion_reports/` with pages processed, chunks created, skipped chunks, warnings, token distribution, duplicate percentage, and contextualization size comparison;
- currently indexed corpus: `155 chunks` from the real PDFs.

### What the Agent Already Returns

- `answer`: synthesized answer;
- `sources`: structured source list;
- `evaluation`: initial heuristic metrics;
- `execution_trace`: flow steps and counts.

Each internal source may currently include:

- `source_file`;
- `section_title`;
- `source_document_id`;
- `parent_chunk_id`;
- `contextual_header`;
- `page_start` and `page_end`;
- `retrieval_rank`;
- `lexical_rank`;
- `hybrid_rank`;
- `hybrid_score`;
- `retrieval_distance`;
- `global_rerank_score`;
- `global_rerank_rank`.

### Current State of Structured Validation and Guardrails

- Pydantic on API input and output;
- Pydantic on critical node outputs;
- `extra="forbid"` on the main schemas;
- structured output for the final LLM synthesis;
- validation of `cited_source_ids` against the actually available sources;
- scope rejection for questions unrelated to the project or its indexed domain;
- explicit fallback when the model or a tool fails.

### Validation Already Performed

- module compilation through `compileall`;
- automated tests passing for `health`, `research`, PDF ingestion, reranking, and structured output;
- PostgreSQL bootstrap with `pgvector`;
- local database seed;
- reingestion of the real corpus;
- manual validation of vector retrieval and the `/research` pipeline against the real internal corpus;
- manual validation of global reranking between internal and web sources.

Pending in phase 1:

- strengthen evaluation and pipeline tests;
- improve per-step observability;
- refine reranking and hybrid retrieval between the internal corpus and web sources;
- decide whether the next iteration should prioritize richer reranking, better evaluation, or conversation memory.

## What This Project Should Train

This project was designed to exercise the most important skills from the target profile:

- `Agentic AI`: multi-step flow with state, tools, and automatic review;
- `Deterministic workflows`: each agent step has explicit input, output, and contract;
- `Guardrails`: structured outputs backed by `Pydantic`;
- `Evaluation framework`: automated tests for answer quality;
- `Logical consistency`: checks on final answer, sources, and coherence;
- `Tool calling`: controlled use of web search, vector search, and SQL queries;
- `FastAPI + React`: lightweight backend and frontend for fast iteration;
- `SQL + vector database`: hands-on use of `PostgreSQL + pgvector`.

## What Stays Out of Phase 1

To keep the project lean, these items remain out of scope for the first phase:

- fine-tuning, LoRA, and QLoRA;
- MongoDB;
- Redis;
- Weaviate;
- WebSockets;
- a full MCP server;
- complex evaluation with many frameworks at once.

None of this is forbidden. It simply does not belong in the initial phase.

## Phase 1 Scope

### Product

An application where the user asks a research question and receives:

- a synthesized answer;
- the sources used;
- a simple quality score;
- a record of the execution process.

### Phase 1 Stack

- `Frontend`: React + Vite + TypeScript
- `Backend`: FastAPI
- `Orchestration`: LangGraph
- `Validation`: Pydantic
- `Database`: PostgreSQL with `pgvector`
- `External search`: 1 web search tool
- `Evaluation`: custom evaluation suite
- `Observability`: simple structured logs

### Intentional Complexity Cut

In phase 1, the agent should not try to do everything. It should answer a narrower set of questions well through a controlled flow.

Examples:

- questions that require consolidating 2 to 5 sources;
- questions about a previously indexed document set;
- questions where the answer must cite evidence;
- questions where answer quality can be at least partially checked.

## Proposed Phase 1 Architecture

```text
User (React)
      |
      v
FastAPI
      |
      v
LangGraph
  1. Classify question
  2. Plan retrieval
  3. Execute tools
     - Web Search
     - Vector Search (pgvector)
     - SQL Query
  4. Consolidate evidence
  5. Generate structured answer
  6. Evaluate answer quality
      |
      v
PostgreSQL + pgvector
```

## Agent Flow

The flow should matter more than the prompt.

### Steps

1. `Classification`
   - identifies the type of question;
   - decides whether it needs web search, vector search, SQL, or a combination.

2. `Planning`
   - defines a short retrieval plan;
   - limits the number of searches and tool calls.

3. `Collection`
   - executes the selected tools;
   - normalizes results into a shared schema.

4. `Consolidation`
   - removes duplicates;
   - marks weak evidence when needed;
   - organizes candidate sources.

5. `Answer`
   - generates a structured answer;
   - includes summary, evidence, and references.

6. `Evaluation`
   - measures minimum quality before returning;
   - records scores and rationales.

## Structured Contracts

Each graph node should produce objects validated by `Pydantic`.

Suggested schemas:

- `QuestionClassification`
- `ResearchPlan`
- `ToolCallResult`
- `EvidenceItem`
- `SynthesisResponse`
- `EvaluationResult`

This point is central for training guardrails and making the flow more deterministic.

## Technical Components

### 1. FastAPI

Responsible for:

- `POST /research`;
- `GET /health`;
- integration with the graph;
- structured response output.

### 2. React Frontend

Responsible for:

- a clean interface for sending questions;
- displaying the final answer;
- displaying the sources;
- displaying evaluation results.

### 3. LangGraph

Responsible for:

- coordinating agent steps;
- routing across tools;
- keeping the flow auditable.

### 4. PostgreSQL + pgvector

Responsible for:

- storing documents and embeddings;
- enabling semantic retrieval;
- enabling simple SQL queries over metadata.

### 5. Custom Evaluation Suite

Responsible for:

- running test questions;
- comparing expected behavior against generated answers;
- measuring quality over time.

## Continuous Evaluation

The project should exercise evaluation from the start, not as an afterthought.

### Minimum Metrics

- `citation_coverage`: does the answer cite sources?
- `groundedness`: is the answer supported by the collected evidence?
- `answer_completeness`: did it answer what was asked?
- `schema_validity`: did the final output respect the contract?
- `tool_efficiency`: did it overuse or underuse tools?

### Initial Evaluation Dataset

Create a small set of 10 to 20 questions with:

- question;
- expected context;
- expected tool type;
- acceptance criteria;
- keywords or facts that must appear;
- reference sources when applicable.

### Practical Goal

Every change in the agent should be testable against this set to avoid regressions.

## Skills This Project Covers Well

High coverage:

- Agentic AI platforms;
- structured workflows;
- guardrails and validation;
- automated evaluation;
- backend API development;
- SQL and vector search;
- tool integration.

Partial coverage:

- mathematical consistency, if numeric cases are added to the dataset;
- MCP, if a dedicated later phase is added;
- more advanced lifecycle management, once deployment and monitoring exist.

Low coverage in phase 1:

- fine-tuning;
- training dynamics;
- loss functions;
- model optimization.

## Roadmap by Phase

### Phase 1 - Reliable Foundation

Goal: build the smallest research agent that is useful and testable.

Deliverables:

- FastAPI with a research endpoint;
- a lightweight React frontend;
- a graph with 4 to 6 steps;
- vector search in PostgreSQL + pgvector;
- one web search tool;
- structured output with Pydantic;
- a basic evaluation suite with fixed cases.

### Phase 2 - Reliability and Observability

Goal: make the agent less fragile.

Deliverables:

- retries and failure handling;
- structured logs by step;
- richer scores;
- a larger evaluation dataset;
- comparisons between prompts or planning strategies.

### Phase 3 - Architecture Expansion

Goal: move closer to the more advanced target profile.

Deliverables:

- streaming responses;
- Redis for caching;
- session history;
- MCP introduced in at least one tool;
- more sophisticated evaluation, possibly with RAGAS.

## Prioritized Backlog

### Priority 1

- define Pydantic schemas;
- implement graph state;
- create `POST /research`;
- index a small corpus in PostgreSQL + pgvector;
- assemble 10 evaluation questions.

### Priority 2

- integrate web search;
- display sources and score in the UI;
- register simple traces per execution;
- add automated flow tests.

### Priority 3

- compare retrieval strategies;
- add numeric and logical cases to the benchmark;
- add caching;
- introduce streaming.

## Phase 1 Success Criteria

At the end of phase 1, the project should allow us to say:

- "I can build an agent with an explicit workflow, not just a large prompt."
- "I can validate each step with schemas."
- "I can measure whether a change improved or degraded quality."
- "I can integrate API, UI, SQL, and vector search into a single system."

## Executive Summary

This project should start small. The best initial version is not the one with the most components, but the one that:

- has a clear flow;
- has strong contracts;
- has repeatable evaluation;
- is easy to debug.

If phase 1 is done well, it already trains a large part of the core skills in the target profile and creates the right foundation for later additions such as `Redis`, `MongoDB`, `WebSockets`, `MCP`, and other advanced elements.
