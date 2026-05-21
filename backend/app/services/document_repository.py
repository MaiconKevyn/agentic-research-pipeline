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
from backend.app.schemas.research import CorpusStats, EvaluationScore, SourceItem


logger = get_logger(__name__)
SQL_PATH = Path(__file__).resolve().parents[2] / "sql" / "init_pgvector.sql"


class DocumentRepositoryError(RuntimeError):
    """Raised when PostgreSQL document operations fail."""


@dataclass(frozen=True)
class ResearchRunRecord:
    run_id: str
    corpus_version_id: str
    question: str
    classification: str | None
    selected_tools: list[str]
    model: str | None
    answer: str
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
                _backfill_normalized_schema(cursor)
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to ensure pgvector schema: {exc}") from exc


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
                        corpus_version_id,
                        question,
                        classification,
                        selected_tools,
                        model,
                        answer,
                        evaluation_scores,
                        execution_trace,
                        latency_ms,
                        cost_estimate_usd,
                        error
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE
                    SET
                        corpus_version_id = EXCLUDED.corpus_version_id,
                        question = EXCLUDED.question,
                        classification = EXCLUDED.classification,
                        selected_tools = EXCLUDED.selected_tools,
                        model = EXCLUDED.model,
                        answer = EXCLUDED.answer,
                        evaluation_scores = EXCLUDED.evaluation_scores,
                        execution_trace = EXCLUDED.execution_trace,
                        latency_ms = EXCLUDED.latency_ms,
                        cost_estimate_usd = EXCLUDED.cost_estimate_usd,
                        error = EXCLUDED.error
                    """,
                    (
                        record.run_id,
                        record.corpus_version_id,
                        record.question,
                        record.classification,
                        record.selected_tools,
                        record.model,
                        record.answer,
                        json.dumps([score.model_dump() for score in record.evaluation_scores]),
                        json.dumps(record.execution_trace),
                        record.latency_ms,
                        record.cost_estimate_usd,
                        record.error,
                    ),
                )
                for source in record.sources:
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
                            json.dumps(source.metadata),
                        ),
                    )
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to record research run: {exc}") from exc


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
