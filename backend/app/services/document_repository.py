from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.db.client import get_connection
from backend.app.schemas.research import (
    AnswerClaim,
    ClaimEvidence,
    CorpusStats,
    EvaluationScore,
    FeedbackEvalCase,
    FeedbackItem,
    FeedbackRequest,
    ResearchRequest,
    RunSourceDetail,
    RunMetricsSummary,
    RunSummary,
    SourceItem,
    WorkspaceRun,
)


logger = get_logger(__name__)
SQL_PATH = Path(__file__).resolve().parents[2] / "sql" / "init_pgvector.sql"


class DocumentRepositoryError(RuntimeError):
    """Raised when PostgreSQL document operations fail."""


@dataclass(frozen=True)
class ResearchRunRecord:
    run_id: str
    workspace_id: str
    corpus_version_id: str
    question: str
    answer_mode: str
    classification: str | None
    selected_tools: list[str]
    model: str | None
    answer: str
    claims: list[AnswerClaim]
    sources: list[SourceItem]
    evaluation_scores: list[EvaluationScore]
    execution_trace: list[str]
    latency_ms: int | None = None
    cost_estimate_usd: float | None = None
    error: str | None = None


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _source_document_id(document: dict[str, Any], document_id: str) -> str:
    metadata = document.get("metadata", {})
    explicit_id = metadata.get("source_document_id")
    if explicit_id:
        return str(explicit_id)

    source_file = metadata.get("source_file")
    checksum = metadata.get("checksum") or metadata.get("file_checksum")
    if source_file and checksum:
        return f"{source_file}:{str(checksum)[:16]}"
    if source_file:
        return str(source_file)
    return document_id


def _normalize_metadata(value: Any) -> dict[str, Any]:
    return dict(value or {})


def ensure_schema() -> None:
    try:
        sql = SQL_PATH.read_text(encoding="utf-8").replace(
            "__EMBEDDING_DIMENSIONS__",
            str(settings.embedding_dimensions),
        )
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                _ensure_workspace_schema(cursor)
                _backfill_normalized_schema(cursor)
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to ensure pgvector schema: {exc}") from exc


def _ensure_workspace_schema(cursor: Any) -> None:
    cursor.execute(
        """
        ALTER TABLE research_runs
        ADD COLUMN IF NOT EXISTS answer_mode TEXT NOT NULL DEFAULT 'detailed'
        """
    )
    cursor.execute(
        """
        ALTER TABLE research_runs
        ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default'
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_research_runs_workspace_created_at
        ON research_runs (workspace_id, created_at DESC)
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS run_feedback (
            feedback_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
            rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
            comment TEXT NULL,
            corrected_answer TEXT NULL,
            add_to_eval BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def _backfill_normalized_schema(cursor: Any) -> None:
    cursor.execute(
        """
        WITH normalized AS (
            SELECT
                document_id,
                title,
                content,
                source_type,
                source_url,
                metadata,
                embedding,
                COALESCE(
                    metadata->>'source_document_id',
                    metadata->>'source_file',
                    document_id
                ) AS source_document_id
            FROM research_documents
        ),
        source_rows AS (
            SELECT DISTINCT ON (source_document_id)
                source_document_id,
                COALESCE(metadata->>'document_title', title) AS source_title,
                source_url,
                source_type,
                metadata
            FROM normalized
            ORDER BY source_document_id, document_id
        )
        INSERT INTO source_documents (
            source_document_id,
            title,
            source_url,
            source_type,
            metadata
        )
        SELECT
            source_document_id,
            source_title,
            source_url,
            source_type,
            metadata || jsonb_build_object('source_document_id', source_document_id)
        FROM source_rows
        ON CONFLICT (source_document_id) DO NOTHING
        """
    )
    cursor.execute(
        """
        WITH normalized AS (
            SELECT
                document_id,
                content,
                metadata,
                COALESCE(
                    metadata->>'source_document_id',
                    metadata->>'source_file',
                    document_id
                ) AS source_document_id
            FROM research_documents
        )
        INSERT INTO document_pages (
            source_document_id,
            page_number,
            extracted_text,
            metadata
        )
        SELECT DISTINCT ON (source_document_id, (metadata->>'page_start')::INTEGER)
            source_document_id,
            (metadata->>'page_start')::INTEGER AS page_number,
            content,
            metadata || jsonb_build_object('source_document_id', source_document_id)
        FROM normalized
        WHERE metadata->>'page_start' ~ '^[0-9]+$'
        ORDER BY source_document_id, (metadata->>'page_start')::INTEGER, document_id
        ON CONFLICT (source_document_id, page_number) DO NOTHING
        """
    )
    cursor.execute(
        """
        WITH normalized AS (
            SELECT
                document_id,
                content,
                metadata,
                embedding,
                COALESCE(
                    metadata->>'source_document_id',
                    metadata->>'source_file',
                    document_id
                ) AS source_document_id
            FROM research_documents
        )
        INSERT INTO document_chunks (
            chunk_id,
            source_document_id,
            section_title,
            page_start,
            page_end,
            raw_text,
            contextualized_embedding_text,
            metadata,
            embedding
        )
        SELECT
            document_id,
            source_document_id,
            metadata->>'section_title',
            CASE WHEN metadata->>'page_start' ~ '^[0-9]+$' THEN (metadata->>'page_start')::INTEGER ELSE NULL END,
            CASE WHEN metadata->>'page_end' ~ '^[0-9]+$' THEN (metadata->>'page_end')::INTEGER ELSE NULL END,
            content,
            content,
            metadata || jsonb_build_object(
                'source_document_id', source_document_id,
                'chunk_id', document_id
            ),
            embedding
        FROM normalized
        ON CONFLICT (chunk_id) DO NOTHING
        """
    )


def upsert_documents(documents: list[dict[str, Any]]) -> int:
    if not documents:
        return 0

    try:
        ensure_schema()
        inserted = 0
        with get_connection() as connection:
            with connection.cursor() as cursor:
                for document in documents:
                    document_id = document.get("document_id") or str(uuid.uuid4())
                    metadata = _normalize_metadata(document.get("metadata"))
                    source_document_id = _source_document_id(document, document_id)
                    metadata["source_document_id"] = source_document_id
                    metadata["chunk_id"] = document_id
                    page_start = metadata.get("page_start")
                    page_end = metadata.get("page_end")
                    section_title = metadata.get("section_title")
                    contextualized_text = document.get("contextualized_embedding_text") or document["content"]
                    checksum = metadata.get("checksum") or metadata.get("file_checksum")
                    quality_flags = metadata.get("quality_flags") or {}

                    cursor.execute(
                        """
                        INSERT INTO source_documents (
                            source_document_id,
                            title,
                            source_url,
                            source_type,
                            checksum,
                            parser_version,
                            metadata,
                            quality_flags
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (source_document_id) DO UPDATE
                        SET
                            title = EXCLUDED.title,
                            source_url = EXCLUDED.source_url,
                            source_type = EXCLUDED.source_type,
                            checksum = EXCLUDED.checksum,
                            parser_version = EXCLUDED.parser_version,
                            metadata = EXCLUDED.metadata,
                            quality_flags = EXCLUDED.quality_flags,
                            updated_at = NOW()
                        """,
                        (
                            source_document_id,
                            metadata.get("document_title") or document["title"],
                            document.get("source_url"),
                            document.get("source_type", "internal"),
                            checksum,
                            metadata.get("parser_version", "prototype"),
                            json.dumps(metadata),
                            json.dumps(quality_flags),
                        ),
                    )
                    if isinstance(page_start, int):
                        last_page = page_end if isinstance(page_end, int) else page_start
                        for page_number in range(page_start, last_page + 1):
                            cursor.execute(
                                """
                                INSERT INTO document_pages (
                                    source_document_id,
                                    page_number,
                                    extracted_text,
                                    metadata
                                )
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (source_document_id, page_number) DO UPDATE
                                SET
                                    extracted_text = EXCLUDED.extracted_text,
                                    metadata = EXCLUDED.metadata,
                                    updated_at = NOW()
                                """,
                                (
                                    source_document_id,
                                    page_number,
                                    document["content"],
                                    json.dumps(metadata),
                                ),
                            )
                    cursor.execute(
                        """
                        INSERT INTO document_chunks (
                            chunk_id,
                            source_document_id,
                            parent_chunk_id,
                            section_title,
                            page_start,
                            page_end,
                            raw_text,
                            contextualized_embedding_text,
                            metadata,
                            embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, (%s)::vector)
                        ON CONFLICT (chunk_id) DO UPDATE
                        SET
                            source_document_id = EXCLUDED.source_document_id,
                            parent_chunk_id = EXCLUDED.parent_chunk_id,
                            section_title = EXCLUDED.section_title,
                            page_start = EXCLUDED.page_start,
                            page_end = EXCLUDED.page_end,
                            raw_text = EXCLUDED.raw_text,
                            contextualized_embedding_text = EXCLUDED.contextualized_embedding_text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding,
                            updated_at = NOW()
                        """,
                        (
                            document_id,
                            source_document_id,
                            metadata.get("parent_chunk_id"),
                            section_title,
                            page_start,
                            page_end,
                            document["content"],
                            contextualized_text,
                            json.dumps(metadata),
                            _vector_literal(document["embedding"]),
                        ),
                    )
                    cursor.execute(
                        """
                        INSERT INTO research_documents (
                            document_id,
                            title,
                            content,
                            source_type,
                            source_url,
                            metadata,
                            embedding
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, (%s)::vector
                        )
                        ON CONFLICT (document_id) DO UPDATE
                        SET
                            title = EXCLUDED.title,
                            content = EXCLUDED.content,
                            source_type = EXCLUDED.source_type,
                            source_url = EXCLUDED.source_url,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding,
                            updated_at = NOW()
                        """,
                        (
                            document_id,
                            document["title"],
                            document["content"],
                            document.get("source_type", "internal"),
                            document.get("source_url"),
                            json.dumps(metadata),
                            _vector_literal(document["embedding"]),
                        ),
                    )
                    inserted += 1
        return inserted
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to upsert documents: {exc}") from exc


def search_similar_documents(query_embedding: list[float], top_k: int) -> list[SourceItem]:
    try:
        ensure_schema()
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        document_id,
                        title,
                        source_type,
                        source_url,
                        metadata,
                        LEFT(content, 500) AS snippet,
                        embedding <=> (%s)::vector AS distance
                    FROM research_documents
                    ORDER BY distance
                    LIMIT %s
                    """,
                    (_vector_literal(query_embedding), top_k),
                )
                rows = cursor.fetchall()
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to search documents: {exc}") from exc

    sources: list[SourceItem] = []
    for rank, row in enumerate(rows, start=1):
        metadata = dict(row["metadata"] or {})
        metadata["retrieval_rank"] = rank
        metadata["retrieval_distance"] = round(float(row["distance"]), 6)
        metadata["retrieval_path"] = "dense"
        sources.append(
            SourceItem(
                source_id=row["document_id"],
                title=row["title"],
                snippet=row["snippet"],
                source_type=row["source_type"],
                url=row["source_url"],
                metadata=metadata,
            )
        )
    return sources


def search_lexical_documents(query: str, top_k: int) -> list[SourceItem]:
    try:
        ensure_schema()
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    WITH query AS (
                        SELECT websearch_to_tsquery('english', %s) AS tsquery
                    )
                    SELECT
                        document_id,
                        title,
                        source_type,
                        source_url,
                        metadata,
                        LEFT(content, 500) AS snippet,
                        ts_rank_cd(
                            to_tsvector('english', title || ' ' || content),
                            query.tsquery
                        ) AS lexical_score
                    FROM research_documents, query
                    WHERE to_tsvector('english', title || ' ' || content) @@ query.tsquery
                    ORDER BY lexical_score DESC, document_id
                    LIMIT %s
                    """,
                    (query, top_k),
                )
                rows = cursor.fetchall()
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to search lexical documents: {exc}") from exc

    sources: list[SourceItem] = []
    for rank, row in enumerate(rows, start=1):
        metadata = dict(row["metadata"] or {})
        metadata["lexical_rank"] = rank
        metadata["lexical_score"] = round(float(row["lexical_score"] or 0.0), 6)
        metadata["retrieval_path"] = "lexical"
        sources.append(
            SourceItem(
                source_id=row["document_id"],
                title=row["title"],
                snippet=row["snippet"],
                source_type=row["source_type"],
                url=row["source_url"],
                metadata=metadata,
            )
        )
    return sources


def count_documents() -> int:
    try:
        ensure_schema()
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) AS total FROM research_documents")
                row = cursor.fetchone()
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to count documents: {exc}") from exc

    return int(row["total"]) if row else 0


def get_or_create_current_corpus_version_id() -> str:
    try:
        ensure_schema()
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        COUNT(DISTINCT source_document_id) AS source_document_count,
                        COUNT(*) AS chunk_count,
                        COALESCE(
                            STRING_AGG(chunk_id || ':' || EXTRACT(EPOCH FROM updated_at)::TEXT, ',' ORDER BY chunk_id),
                            ''
                        ) AS fingerprint
                    FROM document_chunks
                    """
                )
                row = cursor.fetchone() or {}
                source_document_count = int(row.get("source_document_count") or 0)
                chunk_count = int(row.get("chunk_count") or 0)
                checksum = hashlib.sha256(
                    f"{source_document_count}:{chunk_count}:{row.get('fingerprint') or ''}".encode("utf-8")
                ).hexdigest()
                corpus_version_id = f"corpus-{checksum[:16]}"
                cursor.execute(
                    """
                    INSERT INTO corpus_versions (
                        corpus_version_id,
                        source_document_count,
                        chunk_count,
                        checksum
                    )
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (corpus_version_id) DO NOTHING
                    """,
                    (corpus_version_id, source_document_count, chunk_count, checksum),
                )
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to resolve corpus version: {exc}") from exc

    return corpus_version_id


def get_corpus_stats() -> CorpusStats:
    try:
        corpus_version_id = get_or_create_current_corpus_version_id()
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) AS total FROM source_documents")
                source_row = cursor.fetchone() or {}
                cursor.execute("SELECT COUNT(*) AS total FROM document_chunks")
                chunk_row = cursor.fetchone() or {}
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to get corpus stats: {exc}") from exc

    return CorpusStats(
        source_document_count=int(source_row.get("total") or 0),
        chunk_count=int(chunk_row.get("total") or 0),
        corpus_version_id=corpus_version_id,
    )


def record_research_run(record: ResearchRunRecord) -> None:
    try:
        ensure_schema()
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO research_runs (
                        run_id,
                        workspace_id,
                        corpus_version_id,
                        question,
                        classification,
                        selected_tools,
                        model,
                        answer_mode,
                        answer,
                        evaluation_scores,
                        execution_trace,
                        latency_ms,
                        cost_estimate_usd,
                        error
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE
                    SET
                        workspace_id = EXCLUDED.workspace_id,
                        corpus_version_id = EXCLUDED.corpus_version_id,
                        question = EXCLUDED.question,
                        classification = EXCLUDED.classification,
                        selected_tools = EXCLUDED.selected_tools,
                        model = EXCLUDED.model,
                        answer_mode = EXCLUDED.answer_mode,
                        answer = EXCLUDED.answer,
                        evaluation_scores = EXCLUDED.evaluation_scores,
                        execution_trace = EXCLUDED.execution_trace,
                        latency_ms = EXCLUDED.latency_ms,
                        cost_estimate_usd = EXCLUDED.cost_estimate_usd,
                        error = EXCLUDED.error
                    """,
                    (
                        record.run_id,
                        record.workspace_id,
                        record.corpus_version_id,
                        record.question,
                        record.classification,
                        record.selected_tools,
                        record.model,
                        record.answer_mode,
                        record.answer,
                        json.dumps([score.model_dump() for score in record.evaluation_scores]),
                        json.dumps(record.execution_trace),
                        record.latency_ms,
                        record.cost_estimate_usd,
                        record.error,
                    ),
                )
                for source in record.sources:
                    source_metadata = dict(source.metadata)
                    if source.url:
                        source_metadata["url"] = source.url
                    cursor.execute(
                        """
                        INSERT INTO run_sources (
                            run_id,
                            source_id,
                            source_document_id,
                            chunk_id,
                            source_type,
                            title,
                            snippet,
                            score,
                            metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (run_id, source_id) DO UPDATE
                        SET
                            source_document_id = EXCLUDED.source_document_id,
                            chunk_id = EXCLUDED.chunk_id,
                            source_type = EXCLUDED.source_type,
                            title = EXCLUDED.title,
                            snippet = EXCLUDED.snippet,
                            score = EXCLUDED.score,
                            metadata = EXCLUDED.metadata
                        """,
                        (
                            record.run_id,
                            source.source_id,
                            source.metadata.get("source_document_id"),
                            source.metadata.get("chunk_id") or source.source_id,
                            source.source_type,
                            source.title,
                            source.snippet,
                            source.metadata.get("global_rerank_score"),
                            json.dumps(source_metadata),
                        ),
                    )
                for claim in record.claims:
                    for quote in claim.supporting_quotes:
                        cursor.execute(
                            """
                            INSERT INTO claim_evidence_links (
                                claim_evidence_link_id,
                                run_id,
                                claim_text,
                                source_id,
                                quote,
                                support_label,
                                metadata
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (claim_evidence_link_id) DO UPDATE
                            SET
                                claim_text = EXCLUDED.claim_text,
                                source_id = EXCLUDED.source_id,
                                quote = EXCLUDED.quote,
                                support_label = EXCLUDED.support_label,
                                metadata = EXCLUDED.metadata
                            """,
                            (
                                str(uuid.uuid5(
                                    uuid.NAMESPACE_URL,
                                    f"{record.run_id}:{claim.claim_text}:{quote.source_id}:{quote.quote}",
                                )),
                                record.run_id,
                                claim.claim_text,
                                quote.source_id,
                                quote.quote,
                                claim.support_status,
                                json.dumps(
                                    {
                                        "confidence": claim.confidence,
                                        "limitations": claim.limitations,
                                        "conflicts": claim.conflicts,
                                    }
                                ),
                            ),
                        )
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to record research run: {exc}") from exc


def list_research_runs(limit: int = 20) -> list[RunSummary]:
    try:
        ensure_schema()
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        r.run_id,
                        r.workspace_id,
                        r.question,
                        LEFT(r.answer, 220) AS answer_preview,
                        r.answer_mode,
                        r.created_at,
                        r.latency_ms,
                        COUNT(s.source_id) AS source_count
                    FROM research_runs r
                    LEFT JOIN run_sources s ON s.run_id = r.run_id
                    GROUP BY r.run_id
                    ORDER BY r.created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to list research runs: {exc}") from exc

    return [
        RunSummary(
            run_id=row["run_id"],
            workspace_id=row.get("workspace_id") or settings.default_workspace_id,
            question=row["question"],
            answer_preview=row["answer_preview"] or "",
            answer_mode=row.get("answer_mode") or "detailed",
            source_count=int(row.get("source_count") or 0),
            created_at=_isoformat(row.get("created_at")),
            latency_ms=row.get("latency_ms"),
        )
        for row in rows
    ]


def get_research_run(run_id: str) -> WorkspaceRun | None:
    try:
        ensure_schema()
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        r.run_id,
                        r.workspace_id,
                        r.corpus_version_id,
                        r.question,
                        r.answer,
                        r.answer_mode,
                        r.evaluation_scores,
                        r.execution_trace,
                        r.created_at,
                        r.latency_ms,
                        c.source_document_count,
                        c.chunk_count
                    FROM research_runs r
                    LEFT JOIN corpus_versions c ON c.corpus_version_id = r.corpus_version_id
                    WHERE r.run_id = %s
                    """,
                    (run_id,),
                )
                run_row = cursor.fetchone()
                if not run_row:
                    return None
                sources = _fetch_run_sources(cursor, run_id)
                claims = _fetch_run_claims(cursor, run_id)
                feedback = _fetch_run_feedback(cursor, run_id)
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to get research run: {exc}") from exc

    answer_mode = run_row.get("answer_mode") or "detailed"
    return WorkspaceRun(
        run_id=run_row["run_id"],
        workspace_id=run_row.get("workspace_id") or settings.default_workspace_id,
        corpus_version_id=run_row["corpus_version_id"],
        corpus_stats=CorpusStats(
            source_document_count=int(run_row.get("source_document_count") or 0),
            chunk_count=int(run_row.get("chunk_count") or 0),
            corpus_version_id=run_row["corpus_version_id"],
        ),
        question=run_row["question"],
        answer=run_row["answer"],
        answer_mode=answer_mode,
        claims=claims,
        sources=sources,
        evaluation=_parse_evaluation_scores(run_row.get("evaluation_scores")),
        execution_trace=list(run_row.get("execution_trace") or []),
        created_at=_isoformat(run_row.get("created_at")),
        latency_ms=run_row.get("latency_ms"),
        feedback=feedback,
        replay_request=ResearchRequest(
            question=run_row["question"],
            top_k=max(len(sources), 1),
            answer_mode=answer_mode,
        ),
    )


def get_research_run_source(run_id: str, source_id: str) -> RunSourceDetail | None:
    try:
        ensure_schema()
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        s.run_id,
                        s.source_id,
                        s.title,
                        s.source_type,
                        s.snippet,
                        s.source_document_id,
                        s.chunk_id,
                        s.metadata,
                        c.raw_text,
                        c.page_start,
                        c.page_end,
                        c.section_title,
                        p.extracted_text
                    FROM run_sources s
                    LEFT JOIN document_chunks c ON c.chunk_id = s.chunk_id
                    LEFT JOIN document_pages p
                        ON p.source_document_id = c.source_document_id
                        AND p.page_number = c.page_start
                    WHERE s.run_id = %s AND s.source_id = %s
                    """,
                    (run_id, source_id),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                cursor.execute(
                    """
                    SELECT quote
                    FROM claim_evidence_links
                    WHERE run_id = %s AND source_id = %s
                    ORDER BY created_at, quote
                    """,
                    (run_id, source_id),
                )
                quote_rows = cursor.fetchall()
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to get run source: {exc}") from exc

    metadata = dict(row.get("metadata") or {})
    text = row.get("raw_text") or row.get("extracted_text") or row.get("snippet") or ""
    highlights = _deduplicate_strings(
        [row.get("snippet") or ""]
        + [quote_row.get("quote") or "" for quote_row in quote_rows]
    )
    return RunSourceDetail(
        run_id=row["run_id"],
        source_id=row["source_id"],
        title=row["title"],
        source_type=row["source_type"],
        snippet=row.get("snippet") or "",
        text=text,
        source_document_id=row.get("source_document_id") or metadata.get("source_document_id"),
        chunk_id=row.get("chunk_id") or metadata.get("chunk_id"),
        page_start=row.get("page_start") or metadata.get("page_start"),
        page_end=row.get("page_end") or metadata.get("page_end"),
        section_title=row.get("section_title") or metadata.get("section_title"),
        highlights=highlights,
        metadata=metadata,
    )


def record_run_feedback(run_id: str, feedback: FeedbackRequest) -> str:
    feedback_id = str(uuid.uuid4())
    try:
        ensure_schema()
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO run_feedback (
                        feedback_id,
                        run_id,
                        rating,
                        comment,
                        corrected_answer,
                        add_to_eval
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        feedback_id,
                        run_id,
                        feedback.rating,
                        feedback.comment,
                        feedback.corrected_answer,
                        feedback.add_to_eval,
                    ),
                )
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to record run feedback: {exc}") from exc
    return feedback_id


def list_feedback_as_eval_cases(limit: int = 100) -> list[FeedbackEvalCase]:
    try:
        ensure_schema()
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        f.feedback_id,
                        f.corrected_answer,
                        f.comment,
                        r.question,
                        COUNT(s.source_id) AS source_count,
                        ARRAY_REMOVE(ARRAY_AGG(s.source_id ORDER BY s.source_id), NULL) AS source_ids
                    FROM run_feedback f
                    JOIN research_runs r ON r.run_id = f.run_id
                    LEFT JOIN run_sources s ON s.run_id = r.run_id
                    WHERE f.add_to_eval = TRUE
                    GROUP BY f.feedback_id, r.question
                    ORDER BY f.created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to list feedback eval cases: {exc}") from exc

    cases: list[FeedbackEvalCase] = []
    for row in rows:
        corrected_answer = row.get("corrected_answer") or ""
        expected_facts = _extract_expected_facts(corrected_answer)
        cases.append(
            FeedbackEvalCase(
                id=row["feedback_id"],
                question=row["question"],
                expected_answer_type="factual" if int(row.get("source_count") or 0) else "insufficient_evidence",
                required_sources=list(row.get("source_ids") or []),
                expected_facts=expected_facts,
                forbidden_claims=[],
                answer_rubric=row.get("comment") or "Use corrected researcher feedback.",
                difficulty="feedback",
                query_category="feedback",
                top_k=max(min(int(row.get("source_count") or 5), 10), 1),
            )
        )
    return cases


def get_run_metrics_summary(days: int = 30) -> RunMetricsSummary:
    try:
        ensure_schema()
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) AS run_count,
                        COUNT(*) FILTER (WHERE error IS NOT NULL) AS failure_count,
                        AVG(latency_ms)::FLOAT AS average_latency_ms,
                        CAST(
                            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)
                            FILTER (WHERE latency_ms IS NOT NULL)
                            AS FLOAT
                        ) AS p95_latency_ms,
                        AVG(cost_estimate_usd)::FLOAT AS average_cost_estimate_usd
                    FROM research_runs
                    WHERE created_at >= NOW() - (%s || ' days')::INTERVAL
                    """,
                    (days,),
                )
                summary_row = cursor.fetchone() or {}
                cursor.execute(
                    """
                    SELECT
                        DATE(created_at)::TEXT AS date,
                        COUNT(*) AS count
                    FROM research_runs
                    WHERE created_at >= NOW() - (%s || ' days')::INTERVAL
                    GROUP BY DATE(created_at)
                    ORDER BY date
                    """,
                    (days,),
                )
                runs_by_day = [
                    {"date": row["date"], "count": int(row["count"] or 0)}
                    for row in cursor.fetchall()
                ]
                cursor.execute(
                    """
                    SELECT evaluation_scores, DATE(created_at)::TEXT AS date
                    FROM research_runs
                    WHERE created_at >= NOW() - (%s || ' days')::INTERVAL
                    """,
                    (days,),
                )
                score_rows = cursor.fetchall()
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to get run metrics: {exc}") from exc

    average_scores = _aggregate_score_rows(score_rows)
    quality_trend = _quality_trend(score_rows)
    return RunMetricsSummary(
        run_count=int(summary_row.get("run_count") or 0),
        failure_count=int(summary_row.get("failure_count") or 0),
        average_latency_ms=summary_row.get("average_latency_ms"),
        p95_latency_ms=summary_row.get("p95_latency_ms"),
        average_cost_estimate_usd=summary_row.get("average_cost_estimate_usd"),
        average_scores=average_scores,
        runs_by_day=runs_by_day,
        quality_trend=quality_trend,
    )


def _fetch_run_sources(cursor: Any, run_id: str) -> list[SourceItem]:
    cursor.execute(
        """
        SELECT source_id, title, source_type, snippet, metadata
        FROM run_sources
        WHERE run_id = %s
        ORDER BY score DESC NULLS LAST, created_at, source_id
        """,
        (run_id,),
    )
    return [
        SourceItem(
            source_id=row["source_id"],
            title=row["title"],
            snippet=row["snippet"],
            source_type=row["source_type"],
            url=(row.get("metadata") or {}).get("url"),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in cursor.fetchall()
    ]


def _fetch_run_claims(cursor: Any, run_id: str) -> list[AnswerClaim]:
    cursor.execute(
        """
        SELECT claim_text, source_id, quote, support_label, metadata
        FROM claim_evidence_links
        WHERE run_id = %s
        ORDER BY created_at, claim_text, source_id
        """,
        (run_id,),
    )
    grouped: dict[str, dict[str, Any]] = {}
    for row in cursor.fetchall():
        claim = grouped.setdefault(
            row["claim_text"],
            {
                "supporting_source_ids": [],
                "supporting_quotes": [],
                "metadata": dict(row.get("metadata") or {}),
                "support_status": _normalize_support_status(row.get("support_label")),
            },
        )
        if row["source_id"] not in claim["supporting_source_ids"]:
            claim["supporting_source_ids"].append(row["source_id"])
        claim["supporting_quotes"].append(
            ClaimEvidence(source_id=row["source_id"], quote=row["quote"])
        )

    claims: list[AnswerClaim] = []
    for claim_text, payload in grouped.items():
        metadata = payload["metadata"]
        claims.append(
            AnswerClaim(
                claim_text=claim_text,
                supporting_source_ids=payload["supporting_source_ids"],
                supporting_quotes=payload["supporting_quotes"],
                confidence=metadata.get("confidence", "medium"),
                limitations=metadata.get("limitations", []),
                conflicts=metadata.get("conflicts", []),
                support_status=payload["support_status"],
            )
        )
    return claims


def _fetch_run_feedback(cursor: Any, run_id: str) -> list[FeedbackItem]:
    cursor.execute(
        """
        SELECT feedback_id, run_id, rating, comment, corrected_answer, add_to_eval, created_at
        FROM run_feedback
        WHERE run_id = %s
        ORDER BY created_at DESC
        """,
        (run_id,),
    )
    return [
        FeedbackItem(
            feedback_id=row["feedback_id"],
            run_id=row["run_id"],
            rating=row["rating"],
            comment=row.get("comment"),
            corrected_answer=row.get("corrected_answer"),
            add_to_eval=bool(row.get("add_to_eval")),
            created_at=_isoformat(row.get("created_at")),
        )
        for row in cursor.fetchall()
    ]


def _parse_evaluation_scores(value: Any) -> list[EvaluationScore]:
    payload = value or []
    if isinstance(payload, str):
        payload = json.loads(payload)
    return [EvaluationScore.model_validate(item) for item in payload]


def _normalize_support_status(value: Any) -> str:
    if value in {"supported", "unsupported", "conflicting"}:
        return str(value)
    if value == "conflict":
        return "conflicting"
    return "supported"


def _isoformat(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value or "")


def _deduplicate_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduplicated: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(normalized)
    return deduplicated


def _extract_expected_facts(text: str) -> list[str]:
    words = [word.strip(".,;:!?()[]{}").lower() for word in text.split()]
    return [word for word in words if len(word) >= 6][:5]


def _aggregate_score_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for score in _score_payload(row.get("evaluation_scores")):
            metric = score.get("metric")
            value = score.get("score")
            if isinstance(metric, str) and isinstance(value, int | float):
                totals[metric] = totals.get(metric, 0.0) + float(value)
                counts[metric] = counts.get(metric, 0) + 1
    return {
        metric: round(total / counts[metric], 4)
        for metric, total in sorted(totals.items())
        if counts.get(metric)
    }


def _quality_trend(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_day: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_day.setdefault(row.get("date") or "", []).extend(_score_payload(row.get("evaluation_scores")))
    trend: list[dict[str, Any]] = []
    for date, scores in sorted(by_day.items()):
        if not date:
            continue
        day_payload = {"date": date}
        for metric, value in _aggregate_score_rows([{"evaluation_scores": scores}]).items():
            day_payload[metric] = value
        trend.append(day_payload)
    return trend


def _score_payload(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def delete_all_documents() -> None:
    try:
        ensure_schema()
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM claim_evidence_links")
                cursor.execute("DELETE FROM run_sources")
                cursor.execute("DELETE FROM research_runs")
                cursor.execute("DELETE FROM corpus_versions")
                cursor.execute("DELETE FROM document_chunks")
                cursor.execute("DELETE FROM document_pages")
                cursor.execute("DELETE FROM source_documents")
                cursor.execute("DELETE FROM research_documents")
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to delete documents: {exc}") from exc
