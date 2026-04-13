from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.db.client import get_connection
from backend.app.schemas.research import SourceItem


logger = get_logger(__name__)
SQL_PATH = Path(__file__).resolve().parents[2] / "sql" / "init_pgvector.sql"


class DocumentRepositoryError(RuntimeError):
    """Raised when PostgreSQL document operations fail."""


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def ensure_schema() -> None:
    try:
        sql = SQL_PATH.read_text(encoding="utf-8").replace(
            "__EMBEDDING_DIMENSIONS__",
            str(settings.embedding_dimensions),
        )
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to ensure pgvector schema: {exc}") from exc


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
                            json.dumps(document.get("metadata", {})),
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


def delete_all_documents() -> None:
    try:
        ensure_schema()
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM research_documents")
    except Exception as exc:
        raise DocumentRepositoryError(f"Failed to delete documents: {exc}") from exc
