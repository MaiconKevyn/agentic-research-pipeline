from backend.app.core.logging import get_logger
from backend.app.schemas.research import SourceItem
from backend.app.services.document_repository import (
    DocumentRepositoryError,
    get_corpus_stats,
)


logger = get_logger(__name__)


def query_structured_data(question: str) -> list[SourceItem]:
    lowered = question.lower()
    if not any(keyword in lowered for keyword in ("count", "total", "sum", "quantos", "how many")):
        return []

    try:
        corpus_stats = get_corpus_stats()
    except DocumentRepositoryError as exc:
        logger.warning("SQL query unavailable: %s", exc)
        return []

    return [
        SourceItem(
            source_id="sql-corpus-stats",
            title="Indexed corpus counts",
            snippet=(
                f"There are currently {corpus_stats.source_document_count} source documents and "
                f"{corpus_stats.chunk_count} indexed chunks stored in the project's corpus."
            ),
            source_type="sql",
            url=None,
            metadata={
                "query_type": "count_indexed_corpus",
                "source_document_count": corpus_stats.source_document_count,
                "chunk_count": corpus_stats.chunk_count,
                "corpus_version_id": corpus_stats.corpus_version_id,
            },
        )
    ]
