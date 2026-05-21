# Deployment and Operations

## Production Compose Profile

Create a `.env` file with production values before starting the profile:

```env
APP_ENV=production
API_HOST=0.0.0.0
API_PORT=8000
API_AUTH_TOKEN=change-me
DEFAULT_WORKSPACE_ID=default
DATABASE_NAME=research_agent
DATABASE_USER=postgres
DATABASE_PASSWORD=change-me
OPENAI_API_KEY=sk-...
```

Start the API, PostgreSQL, and background ingestion worker:

```bash
docker compose --profile production up --build -d
```

The production profile runs:

- `backend`: FastAPI served by Uvicorn from `Dockerfile.backend`;
- `postgres`: PostgreSQL 16 with pgvector;
- `ingestion-worker`: periodic PDF ingestion from `data/raw`.

## Health and Readiness

- `GET /health` reports process liveness, environment, and version.
- `GET /ready` checks database/schema availability and corpus counts.

Use `/ready` for container health checks because it verifies dependencies.

## Authentication and Workspaces

If `API_AUTH_TOKEN` is set, protected API routes require:

```http
X-API-Key: <API_AUTH_TOKEN>
```

Research requests include `workspace_id`, defaulting to `DEFAULT_WORKSPACE_ID`. This keeps the API contract ready for multi-workspace isolation while preserving a simple default deployment.

## Worker-Based Ingestion

The ingestion worker runs:

```bash
python -m backend.app.workers.ingestion_worker
```

It polls `RAW_PDF_DIR` for PDFs and calls the same production ingestion pipeline used by `scripts/ingest_raw_pdfs.py`. Reports are written to `ingestion_reports/`.

## Operator Metrics

Operators can inspect run health through:

```http
GET /ops/run-metrics?days=30
```

The endpoint returns run count, failure count, average latency, average estimated cost, average evaluation scores, runs by day, and quality trend data. The React workspace displays the same summary in the Operations panel.

## Tracing

Each research run records OpenTelemetry-style trace fields in `execution_trace`:

- `otel_service`
- `otel_span`
- `otel_span_id`
- `otel_duration_ms`
- `otel_attr_*`

These fields make persisted runs inspectable even before an external collector is configured.
