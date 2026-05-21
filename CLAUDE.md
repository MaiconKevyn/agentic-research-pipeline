# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research agent (Phase 1) with an explicit LangGraph workflow, structured Pydantic contracts, PostgreSQL + pgvector retrieval, and a React/Vite frontend. See `project.md` for the full phased roadmap and success criteria — it is the source of truth for scope decisions.

## Commands

All Python commands assume `PYTHONPATH=.` (run from repo root) and the venv at `.venv`.

- Run API: `PYTHONPATH=. .venv/bin/uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000`
- Run tests: `PYTHONPATH=. .venv/bin/pytest`
- Run a single test: `PYTHONPATH=. .venv/bin/pytest tests/test_research.py::test_research_route_returns_structured_response`
- Start Postgres + seed: `./scripts/bootstrap_db.sh` (brings up the `pgvector/pgvector:pg16` container and runs `scripts/seed_documents.py`)
- Ingest PDFs from `data/raw`: `PYTHONPATH=. .venv/bin/python scripts/ingest_raw_pdfs.py` (replaces existing corpus)
- Run the evaluation benchmark: `PYTHONPATH=. .venv/bin/python -c "from evaluation.runner import run_benchmark; print(run_benchmark())"`
- Frontend dev server: `cd frontend && npm run dev` (expects backend at the origin listed in `FRONTEND_ORIGINS`)
- Frontend build: `cd frontend && npm run build`

Environment comes from `.env` at repo root (see `.env.example`). `OPENAI_API_KEY` is required for synthesis, embeddings, and web search — without it the LLM node raises `LLMServiceError` and the pipeline returns a fallback answer.

## Architecture

The request path is **FastAPI → `run_research_pipeline` → `agent.graph.run_research` → compiled LangGraph → `ResearchResponse`**. Keep it this thin: the route in `backend/app/api/routes/research.py` only validates `ResearchRequest` and delegates; all orchestration lives in the graph.

### LangGraph state machine (`agent/graph.py`, `agent/nodes.py`)

Linear pipeline over a `ResearchState` TypedDict: `classify_question → plan_research → collect_evidence → synthesize_answer → evaluate_answer`. Each node returns a partial state dict that LangGraph merges; `execution_trace` is append-only and is how per-step observability is surfaced to the API response.

Two cross-cutting behaviors shape every node:

- **Scope guardrail**: `classify_question` inspects the lowered question against `PROJECT_SCOPE_KEYWORDS`. If none match, `query_kind` becomes `off_topic`, `selected_tools` is emptied, and every downstream node short-circuits (no retrieval, no LLM call, canned refusal, `scope_compliance` evaluation). When adding features, preserve this short-circuit — the tests in `tests/test_research.py` assert that retrieval mocks are never called for off-topic questions.
- **Operational queries**: questions mentioning the internal corpus + count keywords (`count`, `total`, `how many`, `quantos`) append `sql_query` to the tool list so `agent/tools/sql_query.py` can return chunk-count evidence.

### Tools (`agent/tools/`)

Three tools all return `list[SourceItem]` and swallow their own errors into empty lists (logged as warnings) so the graph stays resilient:

- `vector_search` → `generate_embedding` → `search_similar_documents` (pgvector cosine via `embedding <=> query`).
- `web_search` → OpenAI Responses API with the `web_search` tool enabled (`backend/app/services/web_search_service.py`).
- `sql_query` → currently only `count_documents()` for operational stats.

After collection, `collect_evidence` deduplicates by `source_id`, calls `rerank_sources_global` (global rerank across internal + web), and keeps the top `top_k`. Rerank failures degrade gracefully — they trace `global_rerank_error=...` instead of failing the request.

### Structured contracts (`backend/app/schemas/research.py`)

All schemas use `ConfigDict(extra="forbid")`. The ones that matter for flow control: `QuestionClassification`, `ResearchPlan`, `EvidenceCollection`, `SynthesisOutput`, `EvaluationResult`, plus the public `ResearchRequest` / `ResearchResponse`. `SynthesisOutput.cited_source_ids` is revalidated against the actual kept `source_id`s in `llm_service._validate_cited_source_ids` before returning — do not bypass this.

### LLM synthesis (`backend/app/services/llm_service.py`)

Uses the OpenAI **Responses API** (not Chat Completions) with `text.format.type = "json_schema"` and `SynthesisOutput.model_json_schema()` enforcing strict structured output. When you touch the prompt or schema, remember the API expects `role/content` items with `input_text` type, and the response is parsed via `_extract_output_text` which handles both `output_text` and nested `output[].content[]` shapes.

### PDF ingestion (`backend/app/services/pdf_ingestion_service.py`)

Uses `PyMuPDF4LLMLoader` (LangChain community) → `MarkdownHeaderTextSplitter` for structural sections → `RecursiveCharacterTextSplitter.from_tiktoken_encoder` (800-token chunks, 100 overlap) → OpenAI embeddings → `upsert_documents`. Chunks under `MIN_CHUNK_CHARS` (120) are dropped. Metadata carried through to retrieval includes `source_file`, `section_title`, `section_index`, `page_start/end`, `total_pages`.

### Database (`backend/app/db/client.py`, `backend/sql/init_pgvector.sql`)

Single table `research_documents` with an HNSW `vector_cosine_ops` index. The schema SQL templates `__EMBEDDING_DIMENSIONS__` — `ensure_schema()` substitutes `settings.embedding_dimensions` at runtime, so changing the embedding model requires dropping the table (or at minimum the column) before the next ingest.

## Conventions that are easy to miss

- Imports assume the repo root is on `PYTHONPATH` (`from backend.app...`, `from agent...`). Tests rely on this too — there is no `src/` layout or installed package.
- The scope guardrail is keyword-based and English+Portuguese aware. When adding new in-scope topics, extend `PROJECT_SCOPE_KEYWORDS` in `agent/nodes.py` or the guardrail will reject valid questions.
- `execution_trace` strings are asserted in tests (e.g. `"global_rerank_applied"`, `"scope_guardrail_triggered"`). Treat them as a semi-public contract; don't rename without updating tests.
- Phase 1 intentionally excludes Redis, MongoDB, Weaviate, WebSockets, full MCP, and fine-tuning (see `project.md` → "What Stays Out of Phase 1"). Don't pull them in without checking scope.
