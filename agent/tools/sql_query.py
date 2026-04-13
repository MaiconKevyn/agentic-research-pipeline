from backend.app.core.logging import get_logger
from backend.app.schemas.research import SourceItem
from backend.app.services.document_repository import (
    DocumentRepositoryError,
    count_documents,
)


logger = get_logger(__name__)


def query_structured_data(question: str) -> list[SourceItem]:
    lowered = question.lower()
    if not any(keyword in lowered for keyword in ("count", "total", "sum", "quantos", "how many")):
        return []

    try:
        total_documents = count_documents()
    except DocumentRepositoryError as exc:
        logger.warning("SQL query unavailable: %s", exc)
        return []

    return [
        SourceItem(
            source_id="sql-doc-count",
            title="Indexed chunk count",
            snippet=(
                f"There are currently {total_documents} indexed chunks stored in the project's "
                "vector corpus."
            ),
            source_type="sql",
            url=None,
            metadata={
                "query_type": "count_indexed_chunks",
                "chunk_total": total_documents,
            },
        )
    ]
