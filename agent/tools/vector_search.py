from backend.app.core.logging import get_logger
from backend.app.schemas.research import SourceItem
from backend.app.services.document_repository import (
    DocumentRepositoryError,
)
from backend.app.services.embedding_service import (
    EmbeddingServiceError,
)
from backend.app.services.hybrid_search import search_hybrid_documents


logger = get_logger(__name__)


def search_documents(question: str, top_k: int) -> list[SourceItem]:
    try:
        return search_hybrid_documents(question=question, top_k=top_k)
    except (EmbeddingServiceError, DocumentRepositoryError) as exc:
        logger.warning("Hybrid search unavailable: %s", exc)
        return []
